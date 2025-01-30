import torch
import torch.nn as nn
from ..builder import BACKBONES


@BACKBONES.register_module()
class LSTMcsi(nn.Module):
    """LSTM backbone.Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).

    Args:
        in_channels (int): Number of input channels. Defaults to 90.
        out_channels Number of output channels and hidden state channels. Defaults to 128.
        depth (int): Network depth,. Defaults to 2.
        out_indices (Sequence | int): Output from which stages.
            Defaults to (-1,), means the last stage.
    """

    def __init__(self, 
                 in_channels = 90, 
                 depth = 2, 
                 out_channels = 128,
                 out_indices=(-1,),
                 ):
        super(LSTMcsi, self).__init__()
        self.out_indices = out_indices
        self.network = nn.LSTM(in_channels, out_channels, num_layers=depth,batch_first=True) # take the input size of (B, L ,C)

    def forward(self, x):
        x, (h, c) = self.network(x.permute(0,2,1))
        x = x.permute(0,2,1)
        outs = []
        outs.append(x)        
        return tuple(outs)

    def _freeze_stages(self):
        """Freeze patch_embed layer, some parameters and stages."""
        pass
        # for i in range(0, self.frozen_stages):
        #     m = self.network[i]
        #     m.eval()
        #     for param in m.parameters():
        #         param.requires_grad = False
