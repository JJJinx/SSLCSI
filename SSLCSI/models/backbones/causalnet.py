import torch
import torch.nn as nn
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcls.models.backbones.base_backbone import BaseBackbone
from ..builder import BACKBONES

class Chomp1d(torch.nn.Module):
    """
    Removes the last elements of a time series.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
    is the number of elements to remove.
    @param chomp_size Number of elements to remove.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class SqueezeChannels(torch.nn.Module):
    """
    Squeezes, in a three-dimensional tensor, the third dimension.
    """
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)


class CausalConvolutionBlock(torch.nn.Module):
    """
    Causal convolution block, composed sequentially of two causal convolutions
    (with leaky ReLU activation functions), and a parallel residual connection.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).
    @param in_channels Number of input channels.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param dilation Dilation parameter of non-residual convolutions.
    @param final Disables, if True, the last activation function.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 final=False):
        super(CausalConvolutionBlock, self).__init__()

        # Computes left padding so that the applied convolutions are causal
        padding = (kernel_size - 1) * dilation

        # First causal convolution
        conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        # The truncation makes the convolution causal
        chomp1 = Chomp1d(padding)
        relu1 = torch.nn.LeakyReLU()

        # Second causal convolution
        conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU()

        # Causal network
        self.causal = torch.nn.Sequential(
            conv1, chomp1, relu1, conv2, chomp2, relu2
        )

        # Residual connection
        self.upordownsample = torch.nn.Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None

        # Final activation function
        self.relu = torch.nn.LeakyReLU() if final else None

    def forward(self, x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.relu is None:
            return out_causal + res
        else:
            return self.relu(out_causal + res)



@BACKBONES.register_module()
class CausalNet(BaseBackbone):
    """CausalNet backbone.Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
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
        >>> from mmselfsup.models import CausalNet
        >>> import torch
        >>> self = CausalNet()
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
                 finetune=True,
                 init_cfg = None,
                 ):
        super(CausalNet, self).__init__()

        layers = []  # List of causal convolution blocks
        dilation_size = 1  # Initial dilation size
        self.out_indices = out_indices
        self.init_cfg = init_cfg

        for i in range(depth):
            in_channels_block = in_channels if i == 0 else channels
            layers += [CausalConvolutionBlock(
                in_channels_block, channels, kernel_size, dilation_size
            )]
            dilation_size *= 2  # Doubles the dilation size at each step

        # Last layer
        layers += [CausalConvolutionBlock(
            channels, reduced_size, kernel_size, dilation_size
        )]
        reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        squeeze = SqueezeChannels()  # Squeezes the third dimension (time)
        linear = torch.nn.Linear(reduced_size, out_channels)
        self.network = torch.nn.Sequential(*layers, reduce_size, squeeze,linear) #depth +3 layers

        self.finetune = finetune
        if not self.finetune: # when finetune is false, freeze the para
            self._freeze_stages()


    def forward(self, x):
        self._freeze_stages()
        x = self.network(x)      
        out = []
        out.append(x)
        return tuple(out)

    def _freeze_stages(self):
        """Freeze params in backbone when linear probing."""
        for _, param in self.named_parameters():
            param.requires_grad = False

