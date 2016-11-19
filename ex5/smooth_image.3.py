import numpy as np
import matplotlib.pyplot as plt

#Reads a square image in 8-bit/color PPM format from the given file. Note: No checks on valid format are done.
def readImage(filename):
    f = open(filename,"rb")
    
    f.readline()
    s = f.readline()
    f.readline()
    (pixel, pixel) = [t(s) for t,s in zip((int,int),s.split())]
    
    data = np.fromfile(f,dtype=np.uint8,count = pixel*pixel*3)
    img = data.reshape((pixel,pixel,3)).astype(np.double)
    
    f.close()
    
    return img, pixel
    

#Writes a square image in 8-bit/color PPM format.
def writeImage(filename, image):
    f = open(filename,"wb")
    
    pixel = image.shape[0]
    f.write(bytes("P6\n%d %d\n%d\n"%(pixel, pixel, 255),"utf-8"))
    
    image = image.astype(np.uint8)
    
    image.tofile(f)
    
    f.close()
    
def evalKernel(r, h):
    u = r/h
    k = 11.428571428571 / (2 * np.pi * h**2)
    if u > 1:
        return 0
    else: 
        if u > 0.5:
            return 2*k*(1 - u)**3
        else: 
            return k*(1 - 6*u**2 + 6*u**3)
    

img, pixel = readImage("aq-original.ppm")


#Now we set up our desired smoothing kernel. We'll use complex number for it even though it is real. 
kernel_real = np.zeros((pixel,pixel),dtype=np.complex)

# smoothing length
hsml = 10.

#now set the values of the kernel 
for i in np.arange(pixel):
    for j in np.arange(pixel):
        
        #TODO: do something sensible here to set the real part of the kernel
        #kernel_real[i, j] = ....
        kernel_real[i,j] = evalKernel(np.sqrt((i - pixel/2)**2 + (j - pixel/2)**2), hsml)

sumKernel = np.sum(kernel_real)

#Let's calculate the Fourier transform of the kernel
kernel_kspace = np.fft.fft2(kernel_real)


#further space allocations for image transforms
color_real = np.zeros((pixel,pixel),dtype=np.complex)


#we now convolve each color channel with the kernel using FFTs
imgSmooth = img
sumBefore = np.sum(img)
for colindex in np.arange(3):
    #copy input color into complex array
    color_real[:,:].real = img[:,:,colindex]
    
    
    #forward transform
    color_kspace = np.fft.fft2(color_real)
    
    #multiply with kernel in Fourier space
    #TODO: fill in code here
    color_kspace = color_kspace * kernel_kspace
    
    #backward transform
    color_real = np.fft.ifft2(color_kspace)
    # need to shift the image so that the four quarters appear at the right spots
    color_real = np.fft.ifftshift(color_real)
    
    
    #copy real value of complex result back into color array
    imgSmooth[:,:,colindex] = color_real.real
    
sumAfter = np.sum(imgSmooth)    
writeImage("aq-smoothed.ppm", imgSmooth)