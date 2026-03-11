import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel


@MODEL_REGISTRY.register()
class MambaIRv2Model(SRModel):
    """MambaIRv2 model for image restoration."""

    @staticmethod
    def _make_weight_mask(size_h, size_w, sh_start, sh_end, sw_start, sw_end, device):
        """Linear-ramp weight mask to blend tile boundaries smoothly."""
        weight = torch.ones(1, 1, size_h, size_w, device=device)
        if sh_start > 0:
            ramp = torch.linspace(0.0, 1.0, sh_start, device=device)
            weight[:, :, :sh_start, :] *= ramp.view(-1, 1)
        if sh_end > 0:
            ramp = torch.linspace(1.0, 0.0, sh_end, device=device)
            weight[:, :, -sh_end:, :] *= ramp.view(-1, 1)
        if sw_start > 0:
            ramp = torch.linspace(0.0, 1.0, sw_start, device=device)
            weight[:, :, :, :sw_start] *= ramp.view(1, -1)
        if sw_end > 0:
            ramp = torch.linspace(1.0, 0.0, sw_end, device=device)
            weight[:, :, :, -sw_end:] *= ramp.view(1, -1)
        return weight

    def _merge_outputs(self, outputs, ral, row, split_h, split_w, shave_h, shave_w, H, W, C, scale):
        """Weighted blending merge — eliminates hard seam lines."""
        device = outputs[0].device
        _img    = torch.zeros(1, C, H * scale, W * scale, device=device)
        _weight = torch.zeros(1, 1, H * scale, W * scale, device=device)

        for i in range(ral):
            for j in range(row):
                # Placement slice in full output (mirrors input slice, × scale)
                if i == 0 and i == ral - 1:
                    out_top = slice(0, split_h * scale)
                    sh_s, sh_e = 0, 0
                elif i == 0:
                    out_top = slice(0, (split_h + shave_h) * scale)
                    sh_s, sh_e = 0, shave_h * scale
                elif i == ral - 1:
                    out_top = slice((i * split_h - shave_h) * scale, H * scale)
                    sh_s, sh_e = shave_h * scale, 0
                else:
                    out_top = slice((i * split_h - shave_h) * scale, ((i + 1) * split_h + shave_h) * scale)
                    sh_s, sh_e = shave_h * scale, shave_h * scale

                if j == 0 and j == row - 1:
                    out_left = slice(0, split_w * scale)
                    sw_s, sw_e = 0, 0
                elif j == 0:
                    out_left = slice(0, (split_w + shave_w) * scale)
                    sw_s, sw_e = 0, shave_w * scale
                elif j == row - 1:
                    out_left = slice((j * split_w - shave_w) * scale, W * scale)
                    sw_s, sw_e = shave_w * scale, 0
                else:
                    out_left = slice((j * split_w - shave_w) * scale, ((j + 1) * split_w + shave_w) * scale)
                    sw_s, sw_e = shave_w * scale, shave_w * scale

                out = outputs[i * row + j]
                oh, ow = out.shape[-2], out.shape[-1]
                w_mask = self._make_weight_mask(oh, ow, sh_s, sh_e, sw_s, sw_e, device)
                _img[..., out_top, out_left]    += out * w_mask
                _weight[..., out_top, out_left] += w_mask

        return _img / (_weight + 1e-8)

    # test by partitioning
    def test(self):
        _, C, h, w = self.lq.size()
        split_token_h = h // 200 + 1  # number of horizontal cut sections
        split_token_w = w // 200 + 1  # number of vertical cut sections
        # padding
        mod_pad_h, mod_pad_w = 0, 0
        if h % split_token_h != 0:
            mod_pad_h = split_token_h - h % split_token_h
        if w % split_token_w != 0:
            mod_pad_w = split_token_w - w % split_token_w
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, H, W = img.size()
        split_h = H // split_token_h  # height of each partition
        split_w = W // split_token_w  # width of each partition
        # overlapping — 1/4 of tile size (was 1/10) for smoother blending
        shave_h = split_h // 4
        shave_w = split_w // 4
        scale = self.opt.get('scale', 1)
        ral = H // split_h
        row = W // split_w
        slices = []  # list of partition borders
        for i in range(ral):
            for j in range(row):
                if i == 0 and i == ral - 1:
                    top = slice(i * split_h, (i + 1) * split_h)
                elif i == 0:
                    top = slice(i*split_h, (i+1)*split_h+shave_h)
                elif i == ral - 1:
                    top = slice(i*split_h-shave_h, (i+1)*split_h)
                else:
                    top = slice(i*split_h-shave_h, (i+1)*split_h+shave_h)
                if j == 0 and j == row - 1:
                    left = slice(j*split_w, (j+1)*split_w)
                elif j == 0:
                    left = slice(j*split_w, (j+1)*split_w+shave_w)
                elif j == row - 1:
                    left = slice(j*split_w-shave_w, (j+1)*split_w)
                else:
                    left = slice(j*split_w-shave_w, (j+1)*split_w+shave_w)
                temp = (top, left)
                slices.append(temp)
        img_chops = []  # list of partitions
        for temp in slices:
            top, left = temp
            img_chops.append(img[..., top, left])
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                outputs = []
                for chop in img_chops:
                    out = self.net_g_ema(chop)
                    outputs.append(out)
                self.output = self._merge_outputs(
                    outputs, ral, row, split_h, split_w, shave_h, shave_w, H, W, C, scale)
        else:
            self.net_g.eval()
            with torch.no_grad():
                outputs = []
                for chop in img_chops:
                    out = self.net_g(chop)
                    outputs.append(out)
                self.output = self._merge_outputs(
                    outputs, ral, row, split_h, split_w, shave_h, shave_w, H, W, C, scale)
            self.net_g.train()
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
