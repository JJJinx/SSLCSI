from numpy import *
from math import *
import pywt
import sys
import numpy as np
from scipy.signal import butter, lfilter, freqz,sosfilt



def butter_amp_filter(noisy_signal):
    """
        noisy_signal (tensor): amplitude of CSI signal,has the shape of [T,C]
        return shape is [T,C]
    """
    assert len(noisy_signal.shape)==2
    lb,la = butter(6, 60, 'lowpass', fs=1000, output='ba')
    hb,ha = butter(3, 2, 'highpass', fs=1000, output='ba')
    filtered = lfilter(lb,la, noisy_signal,axis=0)
    filtered = lfilter(hb,ha , filtered,axis=0)
    return filtered_signal

def wden_filter(noisy_signal):
    """
        noisy_signal (tensor): amplitude of CSI signal , has the shae of [T, C]
        the parameter is set to be the same as code used in Falldefi

        return ndarray shape is [T,C]
    """
    assert len(noisy_signal.shape)==2
    wden_filtered = []
    for j in range(noisy_signal.shape[1]):
        sig = noisy_signal[:,j]
        #wden denoising
        wden_f = wden(sig,'sqtwolog','soft','sln',4,'sym3') # ndarray
        wden_filtered.append(wden_f)
    wden_filtered = np.vstack(wden_filtered).T
    return wden_filtered



# WDen contains functions for wavelet denoising The functions are named following 
# the naming conventions used by MATLAB Wavelet Toolbox.
#################################
#
# wden(x, tptr, sorh, scal, n, wname) does wavelet denoising. 
#
# x, input signal to be denoised
# tptr, threshold selection rule. See thselect.
# sorh, threshold type. See wthresh
# scal = 'one', for no threshold rescaling
#      = 'sln', for rescaling using a single estimation of level noise based 
#               on the first detail coefficients
#      = 'mln', for rescaling done using level dependent estimation
# wname, wavelet name
#
def wden(x, tptr, sorh, scal, n, wname):
    
    # epsilon stands for a very small number
    eps = 2.220446049250313e-16
    # decompose the input signal. Symetric padding is given as a default.
    coeffs = pywt.wavedec(x, wname,'sym', n)
    # threshold rescaling coefficients
    if scal == 'one':
        stdc = 1
    elif scal == 'sln':
        stdc = wnoisest(coeffs)
    elif scal == 'mln':
        stdc = wnoisest(coeffs, level = n)
    else: 
        raise ValueError('Invalid value for scale, scal = %s' %(scal))
    
    # wavelet coefficients thresholding
    coeffsd = [coeffs[0]]
    for i in range(0, n):
        if tptr == 'sqtwolog' or tptr == 'minimaxi':
            th = thselect(x, tptr)
        else:
            if len(stdc) == 1:
                s = stdc[0]
                if s < sqrt(eps) * max(coeffs[1+i]): 
                    th = 0
                else: 
                    th = thselect(coeffs[1+i]/s, tptr)
            else:
                s = stdc
                if s[i] < sqrt(eps) * max(coeffs[1+i]): 
                    th = 0
                else: 
                    th = thselect(coeffs[1+i]/s[i], tptr)
        
        ### DEBUG
#        print "threshold before rescaling:", th
        ###
        
        # rescaling
        if len(stdc) == 1: 
            s = stdc[0]
            th = th *s
        else: 
            th = th *stdc[i]
        
        #### DEBUG
#        print "threshold:", th
        ####
        
        coeffsd.append(array(wthresh(coeffs[1+i], sorh, th)))
        
    # wavelet reconstruction 
    xdtemp = pywt.waverec(coeffsd, wname, 'sym')
    
    # get rid of the extended part for wavelet decomposition
    extlen = floor(abs(len(x)-len(xdtemp))/2)
    xd = xdtemp[extlen:len(x)+extlen]
    
    return xd
    
    
#################################
#
# thselect(x, tptr) returns threshold x-adapted value using selection rule defined by string tptr.
# 
# tptr = 'rigrsure', adaptive threshold selection using principle of Stein's Unbiased Risk Estimate.
#        'heursure', heuristic variant of the first option.
#        'sqtwolog', threshold is sqrt(2*log(length(X))).
#        'minimaxi', minimax thresholding. 
        
def thselect(x, tptr):
    x = array(x) # in case that x is not an array, convert it into an array
    l = len(x)
    
    if tptr == 'rigrsure':
        sx2 = [sx*sx for sx in absolute(x)]
        sx2.sort()
        cumsumsx2 = cumsum(sx2)        
        risks = []
        for i in range(0, l):
            risks.append((l-2*(i+1)+(cumsumsx2[i]+(l-1-i)*sx2[i]))/l)
        mini = argmin(risks)
        th = sqrt(sx2[mini])
    if tptr == 'heursure':
        hth = sqrt(2*log(l))
        
        # get the norm of x
        normsqr = dot(x, x)
        eta = 1.0*(normsqr-l)/l
        crit = (log(l,2)**1.5)/sqrt(l)
        
        ### DEBUG
#        print "crit:", crit
#        print "eta:", eta
#        print "hth:", hth
        ###
        
        if eta < crit: th = hth
        else: 
            sx2 = [sx*sx for sx in absolute(x)]
            sx2.sort()
            cumsumsx2 = cumsum(sx2)        
            risks = []
            for i in range(0, l):
                risks.append((l-2*(i+1)+(cumsumsx2[i]+(l-1-i)*sx2[i]))/l)
            mini = argmin(risks)
            
            
            ### DEBUG
#            print "risk:", risks[mini]
#            print "best:", mini
#            print "risks[222]:", risks[222]
            ###
            
            rth = sqrt(sx2[mini])
            th = min(hth, rth)     
    elif tptr == 'sqtwolog':
        th = sqrt(2*log(l))
    elif tptr == 'minimaxi':
        if l <32: th = 0
        else: th = 0.3936 + 0.1829*log(l, 2)
    else:
        raise ValueError('Invalid value for threshold selection rule, tptr = %s' %(tptr))
    
    return th
    
#################################
#
# wthresh(x, sorh, t) returns the soft (sorh = 'soft') or hard (sorh = 'hard') 
# thresholding of x, the given input vector. t is the threshold.
# sorh = 'hard', hard trehsholding
# sorh = 'soft', soft thresholding
#    
    
def wthresh(x, sorh, t):
    
    if sorh == 'hard':
        y = [e*(abs(e) >= t) for e in x]
    elif sorh == 'soft':
        y = [((e<0)*-1.0 + (e>0))*((abs(e)-t)*(abs(e) >= t)) for e in x]
    else:
        raise ValueError('Invalid value for thresholding type, sorh = %s' %(sorh))
    
    return y

#################################
#
# wnoisest(coeffs, level = None) estimates the variance(s) of the given detail(s)
#
# coeffs = [CAn, CDn, CDn-1, ..., CD1], multi-level wavelet coefficients
# level, decomposition level. None is the default.
# return stdc (list)

def wnoisest(coeffs, level= None):
    
    l = len(coeffs) - 1
    
    if level == None:
        sig = [abs(s) for s in coeffs[-1]]
        stdc = median(sig)/0.6745
        stdc = [stdc]
    else:
        stdc = []
        for i in range(0, l):
            sig = [abs(s) for s in coeffs[1+i]]
            stdc.append(median(sig)/0.6745)
    
    return stdc

#################################
#
# median(data) returns the median of data (float)
#
# data, a list of numbers.
#       
 
def median(data):
        
    temp = data[:]
    temp.sort()
    dataLen = len(data)
    if dataLen % 2 == 0: # even number of data points
        med = (temp[int(dataLen/2 -1)] + temp[int(dataLen/2)])/2.0
    else:
        med = temp[int(dataLen/2)]
        
    return med   

