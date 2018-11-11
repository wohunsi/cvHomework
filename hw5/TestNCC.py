# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:27:00 2018

@author: admin
"""

import cv2
import numpy as np
import hw5

def funNCC(temp,target):
    n,m = target.shape
    n0,m0 = temp.shape
    result = np.zeros((n-n0+1,m-m0+1))    
    temp = temp.reshape(-1) - np.mean(temp[:])
    norm_sub = np.linalg.norm(temp)
    for i in range(m-m0+1):
        for j in range(n-n0+1):
            sub_mat = target[j:j+n0,i:i+m0]
            vec = np.double(sub_mat.reshape(-1))
            vec = vec - np.mean(vec)            

            result[j,i] = np.dot(vec.transpose(),temp)/(np.linalg.norm(vec)*norm_sub+1e-8)

    y,x = np.where(result==np.max(result))
    return x,y

def videoRefocus(cap,frameId):
    cap.set(cv2.CAP_PROP_POS_FRAMES,10)
    ret,frame = cap.read()
    if not ret:
        print("Video is empty!")
        return None
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    refocusImg = np.zeros(frame.shape)
    newImg = np.zeros(frame.shape)
    h,w = gray.shape
    #x=700,y=450 ; w=230,h=70
    tempImg = gray[450:520,710:930].copy()
#    hw5.showPic(tempImg,"tempIm",10,"gray")
    length = len(frameId)
    for i in range(length):
        cap.set(cv2.CAP_PROP_POS_FRAMES,frameId[i])
        ret,img = cap.read()
#        cv2.imwrite("../frame"+str(frameId[i])+".png",img)
        grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        target = grayImg[250:720,510:1130].copy()
        x,y = funNCC(tempImg,target)
        xoffset = int(200-x)
        yoffset = int(200-y)
        if xoffset > 0:
            colS1 = xoffset
            colE1 = w
            colS2 = 0
            colE2 = w - xoffset
        else:
            colS1 = 0
            colE1 = w + xoffset
            colS2 = - xoffset
            colE2 = w
        
        if yoffset > 0:
            rowS1 = yoffset
            rowE1 = h
            rowS2 = 0
            rowE2 = h - yoffset
        else:
            rowS1 = 0 
            rowE1 = h + yoffset
            rowS2 = 0 - yoffset
            rowE2 = h
        
        newImg[rowS1:rowE1,colS1:colE1,:] = np.double(img[rowS2:rowE2,colS2:colE2,:])
        refocusImg += newImg
        print("index=",i)
    return refocusImg / length

if __name__ == "__main__":
    cap = cv2.VideoCapture("../data/1.AVI")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    print(fps,frames_num)
    
    indexs = np.arange(3,int(frames_num),22)
    
    result = videoRefocus(cap,indexs) / 255
    hw5.showPic(result,"VideoResult")
    cv2.imwrite("../result.png",result*255) 
    print("The end!!!")
    
