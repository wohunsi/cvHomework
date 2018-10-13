import math
import numpy as np


def rec_filter_horizontal(I, D, sigma):
    a = math.exp(-math.sqrt(2.0) / sigma)

    F = I.copy()
    V = np.power(a, D)

    h, w, num_channels = I.shape

    for i in range(1,w):
        for c in range(num_channels):
            F[:,i,c] = F[:,i,c] + V[:,i] * (F[:,i-1,c] - F[:,i,c])

    for i in range(w-2,-1,-1):
        for c in range(num_channels):
            F[:,i,c] = F[:,i,c] + V[:,i+1] * (F[:,i+1,c] - F[:,i,c])

    return F

def bilateral(I, sigma_s, sigma_r, num_iterations=5, J=None):
    if I.ndim == 3:
        img = I.copy()
    else:
        h, w = I.shape
        img = I.reshape((h, w, 1))

    if J is None:
        J = img

    if J.ndim == 2:
        h, w = J.shape
        J = np.reshape(J, (h, w, 1))

    h, w, num_channels = J.shape

    dIcdx = np.diff(J, n=1, axis=1)
    dIcdy = np.diff(J, n=1, axis=0)

    dIdx = np.zeros((h, w))
    dIdy = np.zeros((h, w))

    for c in range(num_channels):
        dIdx[:,1:] = dIdx[:,1:] + np.abs(dIcdx[:,:,c])
        dIdy[1:,:] = dIdy[1:,:] + np.abs(dIcdy[:,:,c])

    dHdx = (1.0 + sigma_s / sigma_r * dIdx)
    dVdy = (1.0 + sigma_s / sigma_r * dIdy)

    dVdy = dVdy.T

    N = num_iterations
    F = img.copy()

    sigma_H = sigma_s

    for i in range(num_iterations):
        sigma_H_i = sigma_H * math.sqrt(3.0) * (2.0 ** (N - (i + 1))) / math.sqrt(4.0 ** N - 1.0)

        F = rec_filter_horizontal(F, dHdx, sigma_H_i)
        F = np.swapaxes(F, 0, 1)
        F = rec_filter_horizontal(F, dVdy, sigma_H_i)
        F = np.swapaxes(F, 0, 1)

    return F

def lum(img):
    l = 0.2126 * img[:,:,0] + \
        0.7152 * img[:,:,1] + \
        0.0722 * img[:,:,2]
    return l

def remove_specials(img):
    img[np.isinf(img)] = 0.0
    img[np.isnan(img)] = 0.0
    return img

def bilateral_separation(img, sigma_s=0.02, sigma_r=0.4):
    r, c = img.shape

    sigma_s = max(r, c) * sigma_s

    img_log = np.log10(img + 1.0e-6)
    img_fil = bilateral(img_log, sigma_s, sigma_r)

    base = 10.0 ** (img_fil) - 1.0e-6

    base[base <= 0.0] = 0.0

    base = base.reshape((r, c))
    detail = remove_specials(img / base)

    return base, detail

def durand(img, target_contrast=5.0):
    L = lum(img)
    tmp = np.zeros(img.shape)
    for c in range(3):
        tmp[:,:,c] = remove_specials(img[:,:,c] / L)

    Lbase, Ldetail = bilateral_separation(L)

    log_base = np.log10(Lbase)

    max_log_base = np.max(log_base)
    log_detail = np.log10(Ldetail)
    compression_factor = np.log(target_contrast) / (max_log_base - np.min(log_base))
    log_absolute = compression_factor * max_log_base

    log_compressed = log_base * compression_factor + log_detail - log_absolute

    output = np.power(10.0, log_compressed)

    ret = np.zeros(img.shape)
    for c in range(3):
        ret[:,:,c] = tmp[:,:,c] * output

    ret = np.maximum(ret, 0.0)
    ret = np.minimum(ret, 1.0)

    return ret

def gamma(L, g):
    return np.power(L, g)