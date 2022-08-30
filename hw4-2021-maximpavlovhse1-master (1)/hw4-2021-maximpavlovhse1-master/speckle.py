# -*- coding: utf-8 -*-
"""
Created on Tue May 11 12:47:14 2021

@author: I
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import astropy.io.fits as fits
import scipy.ndimage
from PIL import Image, ImageEnhance

data = fits.open('speckledata.fits')[2].data





datasum = 0
for i in range(0, len(data)):
    datasum = datasum + data[i][::]





datamean = datasum/len(data)
plt.imshow(datamean, vmax = 500)
plt.savefig('mean.png')





#fourier = np.fft.fft2(datamean)
#spectr = abs(fourier)**2



    
fouriermean = np.mean(np.fft.fft2(data), axis = 0)

realfour = np.fft.fftshift(fouriermean)





spectre = abs(realfour)**2

im = plt.imshow(spectre, cmap = "gray",vmin = 267.3308370477036, vmax = 190586964.7048)
#print(np.max(abs(realfour)**2))
#print(np.min(abs(realfour)**2))

#im = Image.open('fourier.png')
#image brightness enhancer
#enhancer = ImageEnhance.Contrast(im)

#factor = 1.3 #gives original image
#im_output = enhancer.enhance(factor)
#im_output.save('original-image.png')
plt.savefig('fourier.png')


im1 = Image.open('fourier.png')
width = 512
height = 512
resized_im1 = im1.resize((width, height), Image.ANTIALIAS)
resized_im1.save('fourier.png')


angle = np.arange(0, 360)

for i in range(0, 360):
    anglespectr = scipy.ndimage.rotate(spectre, angle[i])
    


plt.imshow(anglespectr, cmap = "gray", vmin = 2670000000.33083704770360, vmax = 19058696400.7048)

plt.savefig("rotaver.png")

im2 = Image.open('rotaver.png')
width = 512
height = 512
resized_im2 = im1.resize((width, height))
#resized_im2.save('rotaver.png')



