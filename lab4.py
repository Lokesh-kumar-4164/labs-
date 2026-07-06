import numpy as np
import matplotlib.pyplot as plt
from skimage import data,color
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi

raw = color.rgb2gray(data.astronaut())

freq = 0.1
angle = np.pi/4

kernel = np.real(gabor_kernel(freq,theta=angle, sigma_x=5,sigma_y=5))

op = ndi.convolve(raw,kernel,mode="wrap")

fig,axes = plt.subplots(1,3)

ax = axes.ravel()

ax[0].imshow(raw,cmap="grey")
plt.show()