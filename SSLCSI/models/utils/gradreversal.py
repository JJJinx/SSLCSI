from torch.autograd import Function
from torch.nn import Module
from torch import tensor

# borrowed from https://github.com/janfreyberg/pytorch-revgrad/blob/master/src/pytorch_revgrad/functional.py
# Thanks janfreyberg!
class RevGradF(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None

revgrad = RevGradF.apply

#from .functional import revgrad
class RevGrad(Module):
    def __init__(self, alpha=1., *args, **kwargs):
        """
        A gradient reversal layer.

        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self._alpha = tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return revgrad(input_, self._alpha)