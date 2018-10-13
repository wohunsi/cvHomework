
import numpy as np
import numpy.linalg
import scipy as sp
import scipy.misc
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import tonemap
from itertools import product
import rawpy

def tls(A, b):
    ATA = np.dot(A.T, A)
    ATb = np.dot(A.T, b)
    return sp.sparse.linalg.bicgstab(ATA, ATb)[0]

def gsolve(Z, B, l, w):
    n = 256
    A = np.zeros((Z.shape[0] * Z.shape[1] + n + 1, n + Z.shape[0]), dtype='float')
    b = np.zeros((A.shape[0], 1), dtype='float')

    k = 0
    for i, j in product(range(Z.shape[0]), range(Z.shape[1])):
        wij = w[Z[i,j]]
        A[k, Z[i, j]] = wij
        A[k, n+i] = -wij
        b[k, 0] = wij * B[j]
        k += 1

    A[k, 128] = 1.0
    k += 1

    for i in range(n-2):
        A[k, i] = l * w[i+1]
        A[k, i+1] = -2.0 * l * w[i+1]
        A[k, i+2] = l * w[i+1]
        k += 1

    x = tls(A, b)
    g = np.exp(x[:n])
    return g

def remove_specials(img):
    img[np.where(np.isnan(img))] = 0.0
    img[np.where(np.isinf(img))] = 0.0
    return img

def weight_function(img, weight_type):

    if weight_type == 'all':
        weight = np.ones(img.shape)
    elif weight_type == 'hat':
        weight = 1.0 - np.power(2.0 * img - 1.0, 12.0)
    elif weight_type == 'Deb97':
        z_min = 0.0
        z_max = 1.0
        tr = (z_min + z_max) / 2.0
        indx1 = np.where(img <= tr)
        indx2 = np.where(img > tr)
        weight = np.zeros(img.shape)
        weight[indx1] = img[indx1] - z_min
        weight[indx2] = z_max - img[indx2]
        weight[np.where(weight < 0.0)] = 0.0
        weight = weight / weight.max()
    else:
        weight = 1.0

    return weight

def tabled_function(img, table):
    for i in range(3):
        work = np.zeros(img[:,:,i].shape)
        for j in range(256):
            indx = np.where(img[:,:,i] == j)
            work[indx] = table[j, i]
        img[:,:,i] = work
    return img

def combine_ldr(stack, exposure_stack, lin_type, lin_fun, weight_type):
    r, c, col, n = stack.shape
    img_out = np.zeros((r, c, col))
    total_weight = np.zeros((r, c, col))

    for i in range(n):
        tmp_stack = []
        if lin_type == 'gamma2.2':
            tmp_stack = np.power(stack[:,:,:,i] / 255.0, 2.2)
        elif lin_type == 'tabledDeb97':
            tmp_stack = tabled_function(stack[:,:,:,i], lin_fun)
        else:
            raise Exception('Unknown linear type: %s' % lin_type)

        tmp_weight = weight_function(tmp_stack, weight_type)
        img_out = img_out + (tmp_weight * tmp_stack) / exposure_stack[i]
        total_weight = total_weight + tmp_weight

    return remove_specials(img_out / total_weight)

def stack_low_res(stack):
    r, c, col, n = stack.shape
    stack_out = []

    for i in range(n):
        tmp_stack = stack[:,:,:,i]
        tmp_stack = np.round(sp.misc.imresize(tmp_stack, 0.01, 'bilinear'))

        r, c, col = tmp_stack.shape

        if i == 0:
            stack_out = np.zeros((r * c, n, col))

        for j in range(col):
            stack_out[:,i,j] = np.reshape(tmp_stack[:,:,j], (r * c))

    return stack_out

def HdrImaging(images, expotimes, weight_type='all', lin_type='gamma2.2'):
    n_img = len(expotimes)
    if n_img == 0:
        raise Exception('Input images and exposure times are invalid')

    h, w, col = images[0].shape
    stack = np.zeros((h, w, col, n_img))
    for i in range(n_img):
        stack[:,:,:,i] = images[i]

    lin_fun = []
    if lin_type == 'tabledDeb97':
        weight = weight_function(np.array([x / 255.0 for x in range(256)]), weight_type)
        stack2 = stack_low_res(stack)
        lin_fun = np.zeros((256, 3))
        for i in range(3):
            g = gsolve(stack2[:,:,i], expotimes, 10.0, weight)
            lin_fun[:,i] = g / g.max()

    return combine_ldr(stack, np.exp(expotimes) + 1.0, lin_type, lin_fun, weight_type)

def HdrShow(imgs, expotimes):
    hdr = HdrImaging(imgs, expotimes)

    tm = tonemap.durand(hdr)
    tm = tonemap.gamma(tm, 1.0 / 2.2)

    return tm
    fig, ax = plt.subplots()
    ax.imshow(tm)
    ax.set_title('Generated HDR')
    ax.axis('off')
    plt.show()

def ProcessJpg():
    imgNum = 16
    imgs = [ None ] * imgNum
    expotimes = [ 0.0 ] * imgNum
    for i in range(imgNum):
        imgPath = "./data/exposure" + str(i+1) + ".jpg"
        imTmp = sp.misc.imread(imgPath)            
        imgs[i] = sp.misc.imresize(imTmp, 0.25)
        expotimes[i] = 1.0 / 2048 * pow(2, i)

    hdrimg = HdrShow(imgs, expotimes)
    sp.misc.imsave("./data/resultJpgHdr.jpg", hdrimg)
    return hdrimg

def ProcessNef():
    imgNum = 16
    imgs = [ None ] * imgNum
    expotimes = [ 0.0 ] * imgNum
    for i in range(imgNum):
        imgPath = "./data/exposure" + str(i+1) + ".nef"         
        raw = rawpy.imread(imgPath)
        imTmp = raw.postprocess()
        imgs[i] = sp.misc.imresize(imTmp, 0.25)
        expotimes[i] = 1.0 / 2048 * pow(2, i)

    hdrimg = HdrShow(imgs, expotimes)
    sp.misc.imsave("./data/resultNefHdr.jpg", hdrimg)
    return hdrimg

if __name__ == '__main__':
    ProcessNef()
    hdrimg = ProcessJpg()

    fig, ax = plt.subplots()
    ax.imshow(hdrimg)
    ax.set_title('Generated HDR')
    ax.axis('off')
    plt.show()
