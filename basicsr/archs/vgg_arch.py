import torch
import torch.nn as nn
from torchvision import models


# VGG19 features layer index → canonical name mapping
_VGG19_LAYER_MAP = {
    'conv1_1': 0,  'relu1_1': 1,
    'conv1_2': 2,  'relu1_2': 3,
    'pool1':   4,
    'conv2_1': 5,  'relu2_1': 6,
    'conv2_2': 7,  'relu2_2': 8,
    'pool2':   9,
    'conv3_1': 10, 'relu3_1': 11,
    'conv3_2': 12, 'relu3_2': 13,
    'conv3_3': 14, 'relu3_3': 15,
    'conv3_4': 16, 'relu3_4': 17,
    'pool3':   18,
    'conv4_1': 19, 'relu4_1': 20,
    'conv4_2': 21, 'relu4_2': 22,
    'conv4_3': 23, 'relu4_3': 24,
    'conv4_4': 25, 'relu4_4': 26,
    'pool4':   27,
    'conv5_1': 28, 'relu5_1': 29,
    'conv5_2': 30, 'relu5_2': 31,
    'conv5_3': 32, 'relu5_3': 33,
    'conv5_4': 34, 'relu5_4': 35,
    'pool5':   36,
}

_VGG_MEAN = [0.485, 0.456, 0.406]
_VGG_STD  = [0.229, 0.224, 0.225]


class VGGFeatureExtractor(nn.Module):
    """Extract intermediate feature maps from a pretrained VGG network.

    Args:
        layer_name_list (list[str]): Names of layers whose outputs are returned,
            e.g. ['conv3_4', 'conv4_4'].
        vgg_type (str): 'vgg19' or 'vgg16'. Default: 'vgg19'.
        use_input_norm (bool): If True, normalise input with ImageNet statistics.
        range_norm (bool): If True, map input from [-1, 1] → [0, 1] before norm.
    """

    def __init__(self,
                 layer_name_list,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False):
        super().__init__()
        self.layer_name_list = layer_name_list
        self.use_input_norm  = use_input_norm
        self.range_norm      = range_norm

        if vgg_type == 'vgg19':
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            layer_map = _VGG19_LAYER_MAP
        elif vgg_type == 'vgg16':
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            raise NotImplementedError('vgg16 layer map not yet defined; use vgg19.')
        else:
            raise ValueError(f'Unsupported vgg_type: {vgg_type}')

        max_idx = max(layer_map[n] for n in layer_name_list)
        self.features = nn.Sequential(*list(vgg.features.children())[:max_idx + 1])
        self._layer_indices = {n: layer_map[n] for n in layer_name_list}

        for param in self.parameters():
            param.requires_grad = False

        if use_input_norm:
            mean = torch.tensor(_VGG_MEAN).view(1, 3, 1, 1)
            std  = torch.tensor(_VGG_STD).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std',  std)

    def forward(self, x):
        if self.range_norm:
            x = (x + 1) / 2
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        outputs = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            for name, idx in self._layer_indices.items():
                if i == idx:
                    outputs[name] = x.clone()
            if len(outputs) == len(self._layer_indices):
                break
        return outputs
