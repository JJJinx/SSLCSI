# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import random
from typing import Tuple

import numpy as np
from sympy import cosine_transform
import torch
from mmcv.utils import build_from_cfg
from PIL import Image, ImageFilter
from timm.data import create_transform
from torchvision import transforms as _transforms
import scipy.interpolate
from scipy.signal import butter, lfilter, freqz



from ..builder import PIPELINES

# register all existing transforms in torchvision
_EXCLUDED_TRANSFORMS = ['GaussianBlur']
for m in inspect.getmembers(_transforms, inspect.isclass):
    if m[0] not in _EXCLUDED_TRANSFORMS:
        PIPELINES.register_module(m[1])


@PIPELINES.register_module()
class BlockwiseMaskGenerator(object):
    """Generate random block mask for each Image.

    Args:
        input_size (int): Size of input image. Defaults to 192.
        mask_patch_size (int): Size of each block mask. Defaults to 32.
        model_patch_size (int): Patch size of each token. Defaults to 4.
        mask_ratio (float): The mask ratio of image. Defaults to 0.6.
    """

    def __init__(self,
                 input_size: int = 192,
                 mask_patch_size: int = 32,
                 model_patch_size: int = 4,
                 mask_ratio: float = 0.6) -> None:
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        mask = torch.from_numpy(mask)  # H X W

        return img, mask


@PIPELINES.register_module()
class RandomAug(object):
    """RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation
    with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.

    This code is borrowed from <https://github.com/pengzhiliang/MAE-pytorch>
    """

    def __init__(self,
                 input_size=None,
                 color_jitter=None,
                 auto_augment=None,
                 interpolation=None,
                 re_prob=None,
                 re_mode=None,
                 re_count=None,
                 mean=None,
                 std=None):

        self.trans = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=color_jitter,
            auto_augment=auto_augment,
            interpolation=interpolation,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
            mean=mean,
            std=std,
        )

    def __call__(self, img):
        return self.trans(img)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class RandomAppliedTrans(object):
    """Randomly applied transformations.

    Args:
        transforms (list[dict]): List of transformations in dictionaries.
        p (float, optional): Probability. Defaults to 0.5.
    """

    def __init__(self, transforms, p=0.5):
        t = [build_from_cfg(t, PIPELINES) for t in transforms]
        self.trans = _transforms.RandomApply(t, p=p)
        self.prob = p

    def __call__(self, img):
        return self.trans(img)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'prob = {self.prob}'
        return repr_str


# custom transforms
@PIPELINES.register_module()
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise).

    Args:
        alphastd (float, optional): The parameter for Lighting.
            Defaults to 0.1.
    """

    _IMAGENET_PCA = {
        'eigval':
        torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec':
        torch.Tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])
    }

    def __init__(self, alphastd=0.1):
        self.alphastd = alphastd
        self.eigval = self._IMAGENET_PCA['eigval']
        self.eigvec = self._IMAGENET_PCA['eigvec']

    def __call__(self, img):
        assert isinstance(img, torch.Tensor), \
            f'Expect torch.Tensor, got {type(img)}'
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'alphastd = {self.alphastd}'
        return repr_str


@PIPELINES.register_module()
class GaussianBlur(object):
    """GaussianBlur augmentation refers to `SimCLR.

    <https://arxiv.org/abs/2002.05709>`_.

    Args:
        sigma_min (float): The minimum parameter of Gaussian kernel std.
        sigma_max (float): The maximum parameter of Gaussian kernel std.
        p (float, optional): Probability. Defaults to 0.5.
    """

    def __init__(self, sigma_min, sigma_max, p=0.5):
        assert 0 <= p <= 1.0, \
            f'The prob should be in range [0,1], got {p} instead.'
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.prob = p

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'sigma_min = {self.sigma_min}, '
        repr_str += f'sigma_max = {self.sigma_max}, '
        repr_str += f'prob = {self.prob}'
        return repr_str


@PIPELINES.register_module()
class Solarization(object):
    """Solarization augmentation refers to `BYOL.

    <https://arxiv.org/abs/2006.07733>`_.

    Args:
        threshold (float, optional): The solarization threshold.
            Defaults to 128.
        p (float, optional): Probability. Defaults to 0.5.
    """

    def __init__(self, threshold=128, p=0.5):
        assert 0 <= p <= 1.0, \
            f'The prob should be in range [0, 1], got {p} instead.'

        self.threshold = threshold
        self.prob = p

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        img = np.array(img)
        img = np.where(img < self.threshold, img, 255 - img)
        return Image.fromarray(img.astype(np.uint8))

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'threshold = {self.threshold}, '
        repr_str += f'prob = {self.prob}'
        return repr_str

@PIPELINES.register_module()
class NoAction(object):
    """No augmentation for test
    """

    def __init__(self):
        pass

    def __call__(self, csi):
        return csi

    def __repr__(self):
        return None


@PIPELINES.register_module()
class JitterCSI(object):
    """Add Gaussian Noise to original CSI data.

    Args:
        sigma_min (float): The minimum parameter of Gaussian kernel std.
        sigma_max (float): The maximum parameter of Gaussian kernel std.
    This code is borrowed from <https://github.com/iantangc/ContrastiveLearningHAR>
    """

    def __init__(self, sigma=0.002):

        #self.sigma_min = sigma_min
        #self.sigma_max = sigma_max
        self.sigma = sigma

    def __call__(self, csi):
        # input csi shape [A,C,T]
        sigma = self.sigma#np.random.uniform(self.sigma_min, self.sigma_max)
        csi_transfromed = csi + torch.from_numpy(np.random.normal(loc=0, scale=sigma, size=csi.shape)).type(csi.dtype)
        return csi_transfromed

    def __repr__(self):
        repr_str = self.__class__.__name__
        #repr_str += f'sigma_min = {self.sigma_min}, '
        #repr_str += f'sigma_max = {self.sigma_max}, '
        repr_str += f'sigma = {self.sigma}, '
        return repr_str


@PIPELINES.register_module()
class PermutationCSI(object):
    """First slices the input CSI data into multiple segments along the time axis, 
    and then permutes the segments to generate a new data example.

    Args:
        num_segments(int): The number of segments for the input signal
    This code is borrowed from <https://github.com/iantangc/ContrastiveLearningHAR>
    """

    def __init__(self, num_segments):

        self.num_segments = num_segments

    def __call__(self, csi):
        # input csi shape [A, C T] or [C,T]
        time_length = csi.shape[-1]
        segment_points_permuted = np.random.choice(time_length, size=(self.num_segments))
        segment_points = np.sort(segment_points_permuted, axis=0)

        csi_permutated = torch.empty(size=csi.shape)
        splitted = np.split(csi, segment_points,axis=2)
        random.shuffle(splitted)
        concat = np.concatenate(splitted, axis=2)
        csi_permutated = torch.from_numpy(concat).type(csi.dtype)

        return csi_permutated

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'num_segments = {self.num_segments}, '
        return repr_str



@PIPELINES.register_module()
class TimeWarpCSI(object):
    """This transformation generates a new data example with the same label 
    by stretching and warping the time intervals between the values of the input timeseries. 

    Args:
        sigma(float): The parameter of Gaussian kernel std.
        num_knots(int): number of knots for cubic spline.
    This code is borrowed from <https://github.com/iantangc/ContrastiveLearningHAR>
    """

    def __init__(self, sigma,num_knots):

        self.sigma = sigma
        self.num_knots = num_knots

    def __call__(self, csi):
        if len(csi.shape)==2:  #data shape [C,T]
            reshape_flag = False
            shape = csi.shape
            time_stamp = np.arange(csi.shape[1])
        else:                   #data shape [A,C,T]
            reshape_flag = True
            shape = csi.shape
            time_length = csi.shape[2]
            time_stamp = np.arange(csi.shape[2])
            csi = csi.reshape(-1,time_length) # became [AC,T]

        knot_xs = np.arange(0, self.num_knots + 2, dtype=float) * (csi.shape[1] - 1) / (self.num_knots + 1)
        spline_ys = np.random.normal(loc=1.0, scale=self.sigma, size=(csi.shape[0], self.num_knots + 2))
        spline_values = np.array([self.get_cubic_spline_interpolation(time_stamp, knot_xs, spline_ys_individual) for spline_ys_individual in spline_ys])
        cumulative_sum = np.cumsum(spline_values, axis=1)
        distorted_time_stamps_all = cumulative_sum / cumulative_sum[:, -1][:, np.newaxis] * (csi.shape[1] - 1)
        csi_transformed = np.empty(shape=csi.shape)
        for i, distorted_time_stamps in enumerate(distorted_time_stamps_all):
            csi_transformed[i,:] = np.interp(time_stamp, distorted_time_stamps, csi[i,:])
        if reshape_flag == True:
            csi_transformed = csi_transformed.reshape(shape)
        csi_transformed = torch.from_numpy(csi_transformed).type(csi.dtype)
        return csi_transformed
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'sigma = {self.sigma}, '
        repr_str += f'num_knots = {self.num_knots}, '
        return repr_str
    
    def get_cubic_spline_interpolation(self,x_eval, x_data, y_data):
        cubic_spline = scipy.interpolate.CubicSpline(x_data, y_data)
        return cubic_spline(x_eval)


@PIPELINES.register_module()
class ScalingCSI(object):
    """Scaling the csi signal with a random generated scaling factor 

    Args:
        sigma(float): The parameter of Gaussian kernel std.
    This code is borrowed from <https://github.com/iantangc/ContrastiveLearningHAR>
    """

    def __init__(self, sigma):

        self.sigma = sigma

    def __call__(self, csi):
        if len(csi.shape)==2:   #data shape [C,T]
            reshape_flag = False
            shape = csi.shape
        else:                   #data shape [A,C,T]
            reshape_flag = True
            shape = csi.shape
            time_length = csi.shape[2]
            csi = csi.reshape(-1,time_length)
        scaling_factor = torch.from_numpy(np.random.normal(loc=1.0, scale=self.sigma, size=(csi.shape[0], 1))).type(csi.dtype)
        csi_transformed = csi*scaling_factor
        if reshape_flag == True:
            csi_transformed = csi_transformed.reshape(shape)
        return csi_transformed

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'sigma = {self.sigma}, '
        return repr_str
    

@PIPELINES.register_module()
class InversionCSI(object):
    """This transformation multiplies the input data with −1. 
    This code is borrowed from <https://github.com/iantangc/ContrastiveLearningHAR>
    """

    def __init__(self):
        pass

    def __call__(self, csi):
        csi_transformed = csi*-1
        return csi_transformed

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class TimeFlippingCSI(object):
    """This transformation reverses the input data along the time-direction. 

    This code is borrowed from <https://github.com/iantangc/ContrastiveLearningHAR>
    """

    def __init__(self):
        pass

    def __call__(self, csi):
        # input csi shape [C T] or [A,C,T]
        csi_transformed = torch.flip(csi,dims=[-1])
        return csi_transformed

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class ChannelShuffleCSI(object):
    """Shuffle the order of channels
    """
    def __init__(self):
        pass

    def __call__(self, csi):
        # input csi shape [C T]
        if len(csi.shape)==2:
            p = np.random.RandomState(seed=21).permutation(csi.shape[0])
            csi_transformed = csi[p,: ]

        # input csi shape [A C T]
        else:
            p = np.random.RandomState(seed=21).permutation(csi.shape[1])
            csi_transformed = csi[:,p,: ]
        return csi_transformed

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class ResampleCSI(object):
    """Up-sample the signal in time axis
    Args:
        up_scale(int): up-sample factor. Default to 3
    This code is borrowed from <https://github.com/Tian0426/CL-HAR>
    """

    def __init__(self,up_scale=3):
        self.up_scale = up_scale

    def __call__(self, csi):
        if len(csi.shape)==2:
            reshape_flag = False
            shape = csi.shape
        else:
            reshape_flag = True
            shape = csi.shape
            time_length = csi.shape[2]
            csi = csi.reshape(-1,time_length)
        orig_steps = np.arange(csi.shape[1])
        interp_steps = np.arange(0, orig_steps[-1]+0.001, 1/self.up_scale)
        Interp = scipy.interpolate.interp1d(orig_steps, csi, axis=1)
        InterpVal = Interp(interp_steps)
        start = random.choice(orig_steps)
        resample_index = np.arange(start, self.up_scale * csi.shape[1], self.up_scale-1)[:csi.shape[1]]
        csi_transformed = torch.from_numpy(InterpVal[:, resample_index]).type(csi.dtype)
        if reshape_flag == True:
            csi_transformed = csi_transformed.reshape(shape)
        return csi_transformed

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'up_scale = {self.up_scale}, '
        return repr_str
    
# frequency domain augmentation
@PIPELINES.register_module()
class LowPassCSI(object):
    """Pass the input signal through a low-pass filter
    Args:
        fpass(float): cutoff frequency. normalized from 0 to 1, where 1 is the Nyquist frequency (1/2 sampling frequency).
        fs(int): The sampling frequency of the digital system.
        order(int): The order of the filter. The higher the order, the steeper the edge of the filter 
    """
    def __init__(self, fpass, fs=None, order=9):

        self.fpass = fpass
        self.fs = fs
        self.order = order

    def __call__(self, csi):
        if len(csi.shape)==2:
            reshape_flag = False
            shape = csi.shape
        else:
            reshape_flag = True
            shape = csi.shape
            time_length = csi.shape[2]
            csi = csi.reshape(-1,time_length)
        b, a = butter(self.order, self.fpass, btype='low', analog=False, fs=self.fs)
        csi_transformed = lfilter(b, a, csi)
        csi_transformed = torch.from_numpy(csi_transformed).type(csi.dtype)
        if reshape_flag == True:
            csi_transformed = csi_transformed.reshape(shape)
        return csi_transformed

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'fpass = {self.fpass}, fs = {self.fs}, order = {self.order}'
        return repr_str


@PIPELINES.register_module()
class HighPassCSI(object):
    """Pass the input signal through a low-pass filter
    Args:
        fpass(float): cutoff frequency. normalized from 0 to 1, where 1 is the Nyquist frequency (1/2 sampling frequency).
        fs(int): The sampling frequency of the digital system.
        order(int): The order of the filter. The higher the order, the steeper the edge of the filter 
    """
    def __init__(self, fpass, fs=None, order=9):

        self.fpass = fpass
        self.fs = fs
        self.order = order

    def __call__(self, csi):
        if len(csi.shape)==2:
            reshape_flag = False
            shape = csi.shape
        else:
            reshape_flag = True
            shape = csi.shape
            time_length = csi.shape[2]
            csi = csi.reshape(-1,time_length)
        b, a = butter(self.order, self.fpass, btype='high', analog=False, fs=self.fs)
        csi_transformed = lfilter(b, a, csi)
        csi_transformed = torch.from_numpy(csi_transformed).type(csi.dtype)
        if reshape_flag == True:
            csi_transformed = csi_transformed.reshape(shape)
        return csi_transformed

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'fpass = {self.fpass}, fs = {self.fs}, order = {self.order}'
        return repr_str


@PIPELINES.register_module()
class PhaseShiftCSI(object):
    """Phase shift augmentation adds a random value 
    between −π and π to the phase values.
    """
    def __init__(self):
        pass

    def __call__(self, csi):
        if len(csi.shape)==2:
            reshape_flag = False
            shape = csi.shape
        else:
            reshape_flag = True
            shape = csi.shape
            time_length = csi.shape[2]
            csi = csi.reshape(-1,time_length)

        fdomain_csi = torch.fft.fft(csi,dim=-1) # apply fft along the time axis
        #fdomain_csi = torch.fft.fftshift(fdomain_csi,dim=-1) # shift the low f in the center

        amp = fdomain_csi.abs()
        phase = fdomain_csi.angle()

        angles = np.random.uniform(low=-np.pi, high=np.pi, size=csi.shape[1]) # for all channels add the same phase shift value
        phase = phase + torch.from_numpy(angles).type(phase.dtype)

        csi_transformed = amp*torch.exp(1j*phase)
        csi_transformed = torch.real(torch.fft.ifft(csi_transformed,dim=-1))
        if reshape_flag == True:
            csi_transformed = csi_transformed.reshape(shape)
        return csi_transformed

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
    
@PIPELINES.register_module()
class AmpPhasePertCSI(object):
    """In this transformation, the amplitude and/or the phase values of randomly 
    selected segments of the frequency domain data are perturbed by Gaussian noise.

    This code is borrowed from <https://github.com/Tian0426/CL-HAR>
    """
    def __init__(self):
        pass

    def __call__(self, csi):
        if len(csi.shape)==2:
            reshape_flag = False
            shape = csi.shape
        else:
            reshape_flag = True
            shape = csi.shape
            time_length = csi.shape[2]
            csi = csi.reshape(-1,time_length)

        fdomain_csi = torch.fft.fft(csi,dim=-1) # apply fft along the time axis
        #fdomain_csi = torch.fft.fftshift(fdomain_csi,dim=-1) # shift the low f in the center
        amp = fdomain_csi.abs()
        phase = fdomain_csi.angle()

        # select a segment to conduct perturbations
        start = np.random.randint(0, int(0.5 * csi.shape[1]))
        end = start + int(0.5 * csi.shape[1])

        #phase shift
        angles = np.repeat(np.expand_dims(np.random.uniform(low=-np.pi, high=np.pi, size=csi.shape[1]), axis=0), csi.shape[0], axis=0)# do the broadcast explicitly
        phase[:, start:end] = phase[:, start:end] + torch.from_numpy(angles[:, start:end]).type(phase.dtype)

        # amp shift
        amp[:, start:end] = amp[:, start:end] + torch.from_numpy(np.random.normal(loc=0., scale=0.8, size=csi.shape)[:, start:end]).type(amp.dtype)

        csi_transformed = amp*torch.exp(1j*phase)
        csi_transformed = torch.real(torch.fft.ifft(csi_transformed,dim=-1))
        if reshape_flag == True:
            csi_transformed = csi_transformed.reshape(shape)
        return csi_transformed

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class AmpPhasePertFullyCSI(object):
    """apply the amplitude and phase perturbation to the whole sequence of frequency response.

    This code is borrowed from <https://github.com/Tian0426/CL-HAR>
    """
    def __init__(self):
        pass

    def __call__(self, csi):
        if len(csi.shape)==2:
            reshape_flag = False
            shape = csi.shape
        else:
            reshape_flag = True
            shape = csi.shape
            time_length = csi.shape[2]
            csi = csi.reshape(-1,time_length)

        fdomain_csi = torch.fft.fft(csi,dim=-1) # apply fft along the time axis
        #fdomain_csi = torch.fft.fftshift(fdomain_csi,dim=-1) # shift the low f in the center
        amp = fdomain_csi.abs()
        phase = fdomain_csi.angle()

        #phase shift
        angles = np.repeat(np.expand_dims(np.random.uniform(low=-np.pi, high=np.pi, size=csi.shape[1]), axis=0), csi.shape[0], axis=0)# do the broadcast explicitly
        phase = phase + torch.from_numpy(angles).type(phase.dtype)

        # amp shift
        amp = amp + torch.from_numpy(np.random.normal(loc=0., scale=0.8, size=csi.shape)).type(amp.dtype)

        csi_transformed = amp*torch.exp(1j*phase)
        csi_transformed = torch.real(torch.fft.ifft(csi_transformed,dim=-1))
        if reshape_flag == True:
            csi_transformed = csi_transformed.reshape(shape)
        return csi_transformed

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str




