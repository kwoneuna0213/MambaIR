import os
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('bcm_opt'):
            self.cri_bcm = build_loss(train_opt['bcm_opt']).to(self.device)
        else:
            self.cri_bcm = None

        if self.cri_pix is None and self.cri_perceptual is None and self.cri_bcm is None:
            raise ValueError('pixel_opt, perceptual_opt and bcm_opt are all None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        if self.cri_bcm:
            l_bcm = self.cri_bcm(self.lq, self.output)
            l_total += l_bcm
            loss_dict['l_bcm'] = l_bcm

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def test_selfensemble_hv(self):
        """4-way TTA: original + hflip + vflip + hflip&vflip.
        transpose를 제외해 비정방형 이미지(720×480 등)에서도 안전하게 동작."""

        def _flip(v, op):
            v2np = v.data.cpu().numpy()
            if op == 'h':
                out = v2np[:, :, ::-1, :].copy()
            else:  # 'v'
                out = v2np[:, :, :, ::-1].copy()
            return torch.tensor(out).to(self.device)

        lq_list = [
            self.lq,
            _flip(self.lq, 'h'),
            _flip(self.lq, 'v'),
            _flip(_flip(self.lq, 'h'), 'v'),
        ]

        net = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
        was_training = net.training
        net.eval()
        with torch.no_grad():
            out_list = [net(aug) for aug in lq_list]
        if was_training:
            net.train()

        # 역변환 후 평균
        out_list[1] = _flip(out_list[1], 'h')
        out_list[2] = _flip(out_list[2], 'v')
        out_list[3] = _flip(_flip(out_list[3], 'v'), 'h')

        self.output = torch.stack(out_list, dim=0).mean(dim=0)

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, epoch=None, total_epochs=None):
        # 모든 rank가 참여하는 분산 validation:
        #   rank 0 : inference + 저장 + metric 계산
        #   rank 1~N : rank 0과 함께 inference 루프를 돌되 저장/metric은 skip
        # → 어느 rank도 NCCL 통신 없이 오래 대기하지 않으므로 deadlock 방지
        self.nondist_validation(
            dataloader, current_iter, tb_logger,
            save_img=(save_img and self.opt['rank'] == 0),
            epoch=epoch, total_epochs=total_epochs,
            log_metric=(self.opt['rank'] == 0),
        )

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, epoch=None, total_epochs=None, log_metric=True):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        skip_existing = self.opt['val'].get('skip_existing', False)
        val_loss_sum = 0.0
        val_loss_count = 0
        save_img_max_per_folder = self.opt['val'].get('save_img_max_per_folder', None)  # 폴더당 최대 N장만 저장

        # save_img_sample: first/mid/last 중 선택 저장 (save_img_max_per_folder보다 우선)
        # yml 예시: save_img_sample: [first, mid, last]
        save_img_sample = self.opt['val'].get('save_img_sample', None)
        # 폴더별 이미지 순서 집계 - dataloader 순회 없이 dataset.paths에서 직접 계산
        _save_idx_set = set()
        if save_img_sample:
            _folder_indices = {}
            for _idx, p in enumerate(dataloader.dataset.paths):
                _fk = osp.basename(osp.dirname(p['lq_path']))
                _folder_indices.setdefault(_fk, []).append(_idx)
            for _fk, _idxs in _folder_indices.items():
                n = len(_idxs)
                picks = []
                if 'first' in save_img_sample:
                    picks.append(_idxs[0])
                if 'mid' in save_img_sample:
                    picks.append(_idxs[n // 2])
                if 'last' in save_img_sample:
                    picks.append(_idxs[-1])
                _save_idx_set.update(picks)

        saved_per_folder = dict()

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            folder_key = osp.basename(osp.dirname(val_data['lq_path'][0]))  # e.g. 1_2_high

            # Compute expected save path and skip if already exists (resume)
            if save_img and skip_existing:
                if self.opt['is_train']:
                    _save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                              f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        _save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                  f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        _save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                  f'{img_name}_{self.opt["name"]}.png')
                if os.path.isfile(_save_img_path):
                    if use_pbar:
                        pbar.update(1)
                        pbar.set_description(f'Skip {img_name}')
                    continue

            self.feed_data(val_data)
            if self.opt['val'].get('self_ensemble', False):
                self.test_selfensemble_hv()
            else:
                self.test()

            # val loss (L1) — gt/output 삭제 전에 계산
            if hasattr(self, 'gt') and hasattr(self, 'cri_pix'):
                with torch.no_grad():
                    val_loss_sum += self.cri_pix(self.output, self.gt).item()
                    val_loss_count += 1

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                do_save = True
                if save_img_sample:
                    # first/mid/last 인덱스에 해당할 때만 저장
                    do_save = idx in _save_idx_set
                elif save_img_max_per_folder is not None:
                    n = saved_per_folder.get(folder_key, 0)
                    do_save = n < save_img_max_per_folder
                    if do_save:
                        saved_per_folder[folder_key] = n + 1
                if do_save:
                    if self.opt['is_train']:
                        # folder_key(e.g. 1_2_high) 포함: visualization/1_2_high/wind2_000_4000.png
                        save_img_path = osp.join(self.opt['path']['visualization'], folder_key,
                                                 f'{img_name}_{current_iter}.png')
                    else:
                        if self.opt['val']['suffix']:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["val"]["suffix"]}.png')
                        else:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["name"]}.png')
                    imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics and log_metric:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger, epoch, total_epochs)

        # val loss 로깅
        if log_metric and val_loss_count > 0:
            val_loss_avg = val_loss_sum / val_loss_count
            epoch_str = f'epoch:{epoch}/{total_epochs}, ' if epoch is not None else ''
            logger = get_root_logger()
            logger.info(f'Validation {dataset_name} [{epoch_str}iter:{current_iter:,}]'
                        f'\t # val_loss (L1): {val_loss_avg:.6f}')
            if tb_logger:
                tb_logger.add_scalar(f'losses/{dataset_name}/val_loss', val_loss_avg, current_iter)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger, epoch=None, total_epochs=None):
        epoch_str = f'epoch:{epoch}/{total_epochs}, ' if epoch is not None else ''
        iter_str = f'{current_iter:,}' if isinstance(current_iter, int) else str(current_iter)
        log_str = f'Validation {dataset_name} [{epoch_str}iter:{iter_str}]\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
