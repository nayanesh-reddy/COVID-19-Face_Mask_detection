import scipy.io.wavfile as audio
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
[fs,a]=audio.read("E:\images\s3.wav")

a=np.int32(a)
b=a-a.min()
b=np.uint8((b/b.max())*255)

c=0
while c==0:
    h=int(input("Enter the height:"))
    w=int(input("Enter the width:"))
    if h*w*3<len(a)*2:
        print("resolution of image is small to fit the audio")
        print("difference: ",(h*w*3-len(a)*2))
    else:
        c=1
        dif=h*w*3-len(a)*2
        b=list(b)
        for k in range(int(np.ceil(dif/2))):
            b.append([0,0])
        b=np.array(b)
        i=np.reshape(b,(h,w,3))
i=np.uint8(i)
plt.imshow(i)
cv.imwrite("E:\images\s3.png",i)
