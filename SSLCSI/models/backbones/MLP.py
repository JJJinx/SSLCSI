import torch
import torch.nn as nn
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcls.models.backbones.base_backbone import BaseBackbone

from ..builder import BACKBONES


@BACKBONES.register_module()
class MLP(BaseBackbone):
    """MLP backbone.Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a tuple which contains a two-dimensional tensor ((`B`, `C_out`)).

    Please refer to the `paper <https://arxiv.org/pdf/1901.10738.pdf>`__ for
    details.

    Args:
        in_channels (int): Number of input channels. Defaults to 90.
        channels(int): channels Number of channels manipulated in the causal CNN. Defaults to 40.
        depth (int): Network depth,. Defaults to 1.
        reduced_size (int):Fixed length to which the output time series of the causal CNN is reduced.
            Default to 160.
        out_channels Number of output channels. Defaults to 320.
        kernel_size Kernel size of the applied non-residual convolutions. Defaults to 4.
        out_indices (Sequence | int): Output from which stages.
            Defaults to (-1,), means the last stage.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    Example:
        >>> from mmselfsup.models import CausaulNet
        >>> import torch
        >>> self = CausaulNet()
        >>> self.eval()
        >>> inputs = torch.rand(3, 90, 100)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
    """

    def __init__(self, 
                 in_channels = 90, 
                 channels = 40, 
                 depth = 10, 
                 reduced_size = 160,
                 out_channels = 320,
                 kernel_size = 4,
                 out_indices=(-1, ),
                 frozen_stages=-1,
                 init_cfg = None,
                 ):
        super(MLP, self).__init__()

        layers = []  # List of causal convolution blocks
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.init_cfg = init_cfg

        self.network = nn.Sequential(
            nn.Linear(3*30*500,1024),
            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
        )


    def forward(self, x):
        self._freeze_stages()
        x = x.reshape(-1,3*30*500)
        x = self.network(x)      
        out = []
        out.append(x)
        return tuple(out)

    def _freeze_stages(self):
        """Freeze patch_embed layer, some parameters and stages."""
        for i in range(0, self.frozen_stages):
            m = self.network[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False


