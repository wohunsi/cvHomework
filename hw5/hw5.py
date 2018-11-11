# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 20:21:25 2018

@author: admin
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d


def showPic(image,title="img",pixl=10,color=None):
    size = (pixl,pixl)
    if color != "gray":        
        showimage =np.zeros(image.shape,dtype=np.float)
        for ch in range(3):
            showimage[:,:,ch] = image[:,:,2-ch]
    else:
        showimage = image
    plt.figure("Image analysis",figsize=size,dpi=96) # 图像窗口名称
    plt.subplot(1,1,1);plt.title(title);plt.imshow(showimage,cmap=color)
    plt.axis('off') # 关掉坐标轴为 off
    plt.show()


def shifting(img,dx,dy):    
    interImg = np.zeros(img.shape,dtype=np.float)
    x_range = np.arange(0,interImg.shape[1])
    y_range = np.arange(0,interImg.shape[0])
    y_pos,x_pos = np.meshgrid(y_range,x_range,indexing='ij')
    for c in range(img.shape[2]):
        img_val = img[y_pos,x_pos,c]
        func = interp2d(x_range,y_range,img_val)
        x_fullrange = x_range + dx
        y_fullrange = y_range - dy
        interImg[:,:,c] = func(x_fullrange,y_fullrange)
    
    return interImg

def RefocusLightField(lightField,d):
    print("d=",d)
    u,v,s,t,c = lightField.shape
    OutImg = np.zeros([s,t,c])
       
    for m in range(u):
        for n in range(v):
            reI = lightField[m,n,:,:,:].reshape([s,t,3])            
#            print("reI shape:",reI.shape)
            newImg = shifting(reI,d*(n-int(v/2)),d*(m-int(u/2)))
            OutImg = OutImg + (1/(u*v)) * newImg
    
    OutImg[OutImg<0] = 0
    OutImg[OutImg>1] = 1    
    return OutImg
 
def BGR2XYZ(image):
    var_R = image[:,:,2]
    var_G = image[:,:,1]
    var_B = image[:,:,0]
    X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
    Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
    Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505
    return X,Y,Z    

def getWei(img,sigma1,sigma2):

    x,y,z = BGR2XYZ(img)
    gauss1 = cv2.GaussianBlur(y,(5,5),sigma1)
    imgsub = y - gauss1
    w = cv2.GaussianBlur(imgsub**2,(19,19),sigma2)
    return w
    

def getAllfocus(image,picNum,path=""):
    H,W,c = img.shape
    u = 16
    v = 16
    t = int(W/u)
    s = int(H/v)
    L = np.zeros([u,v,s,t,c])
    
    for i in range(t):
        for j in range(s):
            L[:,:,j,i,:] = img[j*16:(j+1)*16,i*16:(i+1)*16,:]
    
    w3 = np.zeros([s,t,c])
    WW = np.zeros([s,t,c])
    Img_allfocus = np.zeros([s,t,c])
    Img_depth = np.zeros([s,t])
    
    images = []
    
    ds = np.arange(-picNum,picNum+1,2) 
    
    for j in range(len(ds)):
        outimg = RefocusLightField(L,ds[j]/10)
        if not path == "":
            cv2.imwrite(path+"lfp_"+str(ds[j]/10)+".png",outimg*255)
        images.append(outimg)
        
        w = getWei(outimg,0.4,5)
        w3[:,:,0] = w
        w3[:,:,1] = w
        w3[:,:,2] = w
#        showPic(w3*255,"w3")
        Img_allfocus = Img_allfocus + w3*outimg
        Img_depth += w*(ds[j]/10)
        WW += w3 
        print("Complete the calculation of img No." + str(j+1))
        
        
    Img_allfocus = Img_allfocus / WW
    Img_depth = Img_depth/WW[:,:,1]
    Img_depth = Img_depth + abs(np.min(Img_depth))
    Img_depth = Img_depth/ np.max(Img_depth)    
    
    showPic(Img_allfocus,"allfocus")
    showPic(Img_depth,"lfp_depth",10,"gray")    
    cv2.imwrite(path+"lfp_allfocus.png",Img_allfocus*255)
    cv2.imwrite(path+"lfp_depth.png",Img_depth*255)
    

if __name__ == "__main__":
    import os
    dirs = "./imgOut/"
    if not os.path.exists(dirs):
        os.makedirs(dirs)
        
    img = cv2.imread("./data/chessboard_lightfield.png",-1)/255.        
    getAllfocus(img,20,dirs)

        
        
    
    
    
    
    
    
    
    
    
    
    

