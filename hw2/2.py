import sys
import cv2
import numpy as np
import os
import scipy.signal as sc

def BGR2YIQ(bgr):
    r = bgr[:, :, 2]
    g = bgr[:, :, 1]
    b = bgr[:, :, 0]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    i = 0.596 * r + -0.274 * g + -0.322 * b
    q = 0.211 * r + -0.523 * g + 0.312 * b
    return cv2.merge([y, i, q])

def YIQ2BGR(yiq):
    y = yiq[:, :, 0]
    i = yiq[:, :, 1]
    q = yiq[:, :, 2]
    r = 1 * y + 0.956 * i + 0.621 * q
    g = 1 * y + -0.272 * i + -0.647 * q
    b = 1 * y + -1.106 * i + 1.703 * q
    return cv2.merge([b, g, r])

def binomialFilter(sz):

    if (sz < 2):
        print ('size argument must be larger than 1')
        return

    kernel = np.asarray(([0.5,0.5],[0.5,0.5]))

    for n in range(1,sz-1):
        kernel = sc.convolve2d(np.asarray(([0.5,0.5],[0.5,0.5])), kernel)
    
    return kernel

def named_filter(name):

    if (name[:5] == 'binom'):
        kernel = np.sqrt(2) * binomialFilter(int(name[5:]))
    elif name == 'qmf5':
        kernel = np.asarray((-0.076103, 0.3535534, 0.8593118, 0.3535534, -0.076103))
    elif name == 'qmf9':
        kernel = np.asarray((0.02807382, -0.060944743, -0.073386624, 0.41472545, 0.7973934,
          0.41472545, -0.073386624, -0.060944743, 0.02807382))
    elif name == 'qmf13':
        kernel = np.asarray((-0.014556438, 0.021651438, 0.039045125, -0.09800052,
          -0.057827797, 0.42995453, 0.7737113, 0.42995453, -0.057827797,
          -0.09800052, 0.039045125, 0.021651438, -0.014556438))
    elif name == 'qmf8':
        kernel = np.sqrt(2) * np.asarray((0.00938715, -0.07065183, 0.06942827, 0.4899808,
        0.4899808, 0.06942827, -0.07065183, 0.00938715))
    elif name == 'qmf12':
        kernel = np.sqrt(2) * np.asarray((-0.003809699, 0.01885659, -0.002710326, -0.08469594,
        0.08846992, 0.4843894, 0.4843894, 0.08846992, -0.08469594, -0.002710326,
        0.01885659, -0.003809699))
    elif name == 'qmf16':
        kernel = np.sqrt(2) * np.asarray((0.001050167, -0.005054526, -0.002589756, 0.0276414, -0.009666376,
        -0.09039223, 0.09779817, 0.4810284, 0.4810284, 0.09779817, -0.09039223, -0.009666376,
        0.0276414, -0.002589756, -0.005054526, 0.001050167))
    elif name == 'haar':
        kernel = np.asarray((1, 1)) / np.sqrt(2)
    elif name == 'daub2':
        kernel = np.asarray((0.482962913145, 0.836516303738, 0.224143868042, -0.129409522551))
    elif name == 'daub3':
        kernel = np.asarray((0.332670552950, 0.806891509311, 0.459877502118, -0.135011020010,
        -0.085441273882,  0.035226291882))
    elif name == 'daub4':
        kernel = np.asarray((0.230377813309, 0.714846570553, 0.630880767930, -0.027983769417,
        -0.187034811719, 0.030841381836, 0.032883011667, -0.010597401785))
    elif name == 'gauss5':  # for backward-compatibility
        kernel = np.sqrt(2) * np.asarray((0.0625, 0.25, 0.375, 0.25, 0.0625))
    elif name == 'gauss3':  # for backward-compatibility
        kernel = np.sqrt(2) * np.asarray((0.25, 0.5, 0.25))
    else:
        print ('Bad filter name: ', name)
        return

    return kernel

   
def shiftdim(x, n):  
    return x.transpose(np.roll(range(x.ndim), -n))


def repmat(a,m):
    #First, pad out a so it has same dimensionality as m
    for i in range(0,m.ndim-a.ndim):
        a = np.expand_dims(a,1)
    #Now just use numpy tile and return result
    return np.tile(a,m.shape)

def ideal_bandpassing(input, dim, wl, wh, samplingRate):

    #if dim is greater than the dimensionality (2d, 3d etc) of the input, quit
    if (dim > len(input.shape)):
        print ('Exceed maximum dimension')
        return
        
    #This has the effect that input_shifted[0] = input[dim]
    input_shifted = shiftdim(input,dim-1)
         
    #Put the dimensions of input_shifted in a 1d array
    Dimensions = np.asarray(input_shifted.shape)
            
    #how many things in the first dimension of input_shifted
    n = Dimensions[0]
    
    #get the dimensionality (eg. 2d, 3d etc) of input_shifted
    dn = input_shifted.ndim
        
    #creates a vector [1,...,n], the same length as the first dimension of input_shifted
    Freq = np.arange(1.0,n+1)
            
    #Equivalent in python: Freq = (Freq-1)/n*samplingRate
    Freq = (Freq-1)/n*samplingRate
           
    #Create boolean mask same size as Freq, true in between the frequency limits wl,wh
    mask = (Freq > wl) & (Freq < wh)
 
    Dimensions[0] = 1
    mask = repmat(mask,np.ndarray(Dimensions))

    #F = fft(X,[],dim) and F = fft(X,n,dim) applies the FFT operation across the dimension dim.
    #Python: F = np.fftn(a=input_shifted,axes=0)
    F = np.fft.fftn( a=input_shifted, axes=[0] )
    
    #So we are indexing array F using boolean not mask, and setting those values of F to zero, so the others pass thru
    #Python: F[ np.logical_not(mask) ]
    F[ np.logical_not(mask) ] = 0
    
    #Get the real part of the inverse fourier transform of the filtered input
    filtered = np.fft.ifftn( a=F, axes=[0] ).real
    
    filtered = filtered.astype(np.float32)
    
    filtered = shiftdim(filtered,dn-(dim-1))
    
    return filtered
    

def corrDn(im, filt, edges='reflect1', step=(1,1), start=(0,0), stop=0):

    #default value of stop is size of image
    if stop==0:
        stop = im.shape

    # Reverse order of taps in filt, to do correlation instead of convolution
    filt = filt[::-1,::-1]

    #convolution here
    tmp = sc.convolve2d(im,filt,mode='valid',boundary = 'symm')
        
    #this is the downsampling line
    res = tmp[start[0]:stop[0]+1:step[1],start[1]:stop[1]+1:step[1]]
    
    return res

def return_next_frame_blurred(vid,level,colourSpace):

	retval,temp = vid.read()
	temp = temp.astype(np.float32)

	if colourSpace == 'yuv':
            temp = cv2.cvtColor(temp,cv2.COLOR_BGR2YUV)
	elif colourSpace == 'yiq':
            temp = BGR2YIQ(temp)

	return blurDnClr(temp,level)

def build_GDown_stack(vidFile, startIndex, endIndex, level, colourSpace = 'rgb'):


    # Read video
    vid = cv2.VideoCapture(vidFile)
    fps = vid.get(cv2.CAP_PROP_FPS)
    framecount = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    vidWidth = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    vidHeight = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))


    #firstFrame
    blurred = return_next_frame_blurred(vid,level,colourSpace)
    
    # create pyr stack
    GDown_stack = np.zeros((endIndex - startIndex +1, blurred.shape[0],blurred.shape[1],blurred.shape[2]))
    GDown_stack[0,:,:,:] = blurred


    for k in range(1,endIndex-startIndex+1):

        #process the video frame and add it to the stack
        GDown_stack[k,:,:,:] = return_next_frame_blurred(vid,level,colourSpace)
        
        #progress indicator
        sys.stdout.write('.')
        sys.stdout.flush()
       
    return GDown_stack
    

    
def blurDn(im, nlevs=1, filt='binom5'):

    #if filt is a string, pass it to namedFilter, which returns a 1d kernel
    if isinstance(filt,str):
        filt = named_filter(filt)

    #Normalize filt. Applying this more than once has no effect - once it's normalized it's normalized
    filt = filt/np.sum(filt)

    #Recursively call BlurDn, passing the normalized filt, and taking one off nlevs
    if nlevs > 1:
        im = blurDn(im,nlevs-1,filt)


    if (nlevs >= 1):
        #if im is 1d
        if (len(im.shape) == 1):
            if not (1 in filt.shape):
                print ('Cant  apply 2D filter to 1D signal')
                return
                
            if (im.shape[1] == 1):
                filt = filt.flatten()
            else:
                filt = numpy.transpose(filt.flatten())
            
            res = corrDn(im,filt,'reflect1',tuple(map(lambda x: int(not x==1)+1,im.shape)))
        
        #else if im is 2d, but the filter is 1d
        elif (len(filt.shape) == 1):
            filt = filt.flatten()
            res = corrDn(im, filt, 'reflect1', (2,1))
            res = corrDn(res,numpy.transpose(filt), 'reflect1', (1,2))
        else:
            res = corrDn(im, filt, 'reflect1', (2,2))
    
    else:
        res = im
    
    return res


def blurDnClr(im, nlevs=1, filt='binom5'):

    tmp = blurDn(im[:,:,0], nlevs, filt);
    out = np.zeros((tmp.shape[0], tmp.shape[1], im.shape[2]));
    out[:,:,0] = tmp;
    for clr in range(1,im.shape[2]):
        out[:,:,clr] = blurDn(im[:,:,clr], nlevs, filt);

    
    return out



def amplify_spatial_Gdown_temporal_ideal(vidFile,outDir, alpha,level,
                     fl,fh, chromAttenuation, colourSpace = 'rgb'):
  
    vidName = os.path.basename(vidFile)
    vidName = vidName[:-4]
    outName = (outDir + vidName + '_' + colourSpace + '.avi')


    vid = cv2.VideoCapture(vidFile)
    fr = vid.get(cv2.CAP_PROP_FPS)
    len = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vidWidth = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    vidHeight = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    startIndex = 0
    endIndex = len-1
    
    
    print ('width, height, fps ', str(vidWidth) + ', ' + str(vidHeight) + ', ' + str(fr))


    # Define the codec and create VideoWriter object
    capSize = (vidWidth,vidHeight)
    fourcc = cv2.VideoWriter_fourcc('j', 'p', 'e', 'g')
    vidOut = cv2.VideoWriter()
    success = vidOut.open(outName,fourcc,fr,capSize,True)
    
    print (outName)
    

    # compute pyramid
    print ('Laplacian pyramid construction...')
    Gdown_stack = build_GDown_stack(vidFile, startIndex, endIndex, level, colourSpace)
    print ('Finished')
    
    
    # Temporal filtering
    print ('Band passing...')
    filtered_stack = ideal_bandpassing(Gdown_stack, 1, fl, fh, fr)
    print ('Finished')
        
    
    ## amplify
    print ('Amplify')
    if   colourSpace == 'yuv':
        filtered_stack[:,:,:,0] = filtered_stack[:,:,:,0] * alpha
        filtered_stack[:,:,:,1] = filtered_stack[:,:,:,1] * alpha * chromAttenuation
        filtered_stack[:,:,:,2] = filtered_stack[:,:,:,2] * alpha * chromAttenuation
    elif colourSpace == 'rgb':
        filtered_stack = filtered_stack * alpha
    elif colourSpace == 'yiq':
        filtered_stack = filtered_stack * alpha



    ## Render on the input video
    print ('Image reconstruction...')
    # output video
    for k in range(0,endIndex-startIndex+1):

        retval,temp = vid.read()
        frame = temp.astype(np.float32)
        
        filtered = np.squeeze(filtered_stack[k,:,:,:])          
        filtered = cv2.resize(filtered,(vidWidth, vidHeight),0,0,cv2.INTER_LINEAR)
        
        if   colourSpace == 'yuv':
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2YUV)
            frame[:,:,1:] = frame[:,:,1:] + filtered[:,:,1:]
            frame = cv2.cvtColor(frame,cv2.COLOR_YUV2BGR)
        elif   colourSpace == 'yiq':
            frame = BGR2YIQ(frame)
            frame[:,:,1:] = frame[:,:,1:] + filtered[:,:,1:]
            frame = YIQ2BGR(frame)
        elif colourSpace == 'rgb':
            frame = frame + filtered
               
        frame = np.clip(frame,0,255)
        frame = cv2.convertScaleAbs(frame)

        vidOut.write(frame)
        sys.stdout.write('.')
        sys.stdout.flush()


    print ('Finished')
    vid.release()
    vidOut.release() 




if __name__=="__main__":
    
    amplify_spatial_Gdown_temporal_ideal('data/face.mp4','data/',50,4,50/60.0,60/60.0, 1,'yiq')
    #amplify_spatial_Gdown_temporal_ideal('data/baby2.mp4','data/',120,4,140/60.0,160/60.0, 1,'yiq')

