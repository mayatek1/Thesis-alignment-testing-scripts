# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 13:52:09 2023

@author: pymte
"""
import numpy as np
import cv2 
import scipy as sp
from scipy import misc, fftpack

from skimage.registration import phase_cross_correlation
import matplotlib.pyplot as plt
from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import fourier_shift
from scipy.signal import fftconvolve, convolve

import functions as fn

import skimage as sk


'''cross correlation coefficient function'''

def mean2(x):
    y = np.sum(x) / np.size(x);
    return y


def corr2(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)

    numerator = (a * b).sum()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())

    if denominator != 0:
        r = numerator / denominator
        return r
    else:
        # Handle the case where the denominator is zero
        return np.nan  # You can return NaN (Not a Number) or handle it in your preferred way


def normxcorr2(template, image, mode="same"):

    """

    Input arrays should be floating point numbers.

   

    Args:

        template::arr

            N-D array, of template or filter you are using for cross-correlation. Must be less or equal dimensions to image. Length of each dimension must be less than length of image.

        image::arr

            N-D array containing nput image being compared to the template

        mode::

            Options, "full", "valid", "same"

            full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs. Output size will be image size + 1/2 template size in each dimension.

            valid: The output consists only of those elements that do not rely on the zero-padding.

            same: The output is the same size as image, centered with respect to the ‘full’ output.

       

   

    Returns:

        CCR_map::arr

            N-D array of same dimensions as image. Size depends on mode parameter.

   

    """

 

    # If this happens, it is probably a mistake

    if np.ndim(template) > np.ndim(image) or len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:

        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

 

    template = template - np.mean(template)

    image = image - np.mean(image)

 

    a1 = np.ones(template.shape)

    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate

    ar = np.flipud(np.fliplr(template))

    out = fftconvolve(image, ar.conj(), mode=mode)

   

    image = fftconvolve(np.square(image), a1, mode=mode) - np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

 

    # Remove small machine precision errors after subtraction

    image[np.where(image < 0)] = 0

 

    template = np.sum(np.square(template))

    out = out / np.sqrt(image * template)

 

    # Remove any divisions by 0 or very close to 0

    out[np.where(np.logical_not(np.isfinite(out)))] = 0

   

    return out

def ind2sub(array_shape, ind):
    rows = (ind.astype('int') / array_shape[1])
    cols = (ind.astype('int') % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return (rows, cols)

def blend(a, b, alpha=0.5):
    """
    Alpha blend two images.

    Parameters
    ----------
    a, b : numpy.ndarray
        Images to blend.
    alpha : float
        Blending factor.

    Returns
    -------
    result : numpy.ndarray
        Blended image.
    """

    a = sk.img_as_float(a)
    b = sk.img_as_float(b)
    return a*alpha+(1-alpha)*b



def NanoLocz_ccr_rot_align(im, imR,a_range,thresh):
 
    """
    Estimate the relative rotational offset between two images using
    rotational cross-correlation.

    This function rotates the reference image through a range of trial
    angles, calculates the correlation coefficient between the rotated
    image and the target image, and identifies the angle producing the
    highest correlation. A spline interpolation and local optimisation
    step are then used to refine the rotation estimate.

    Parameters
    ----------
    im : ndarray
        Target image.

    imR : ndarray
        Reference image to be rotated.

    a_range : tuple or list
        Two-element sequence specifying the minimum and maximum rotation
        angles (degrees) to search.

    thresh : float
        Height threshold. Pixels below this value are set to zero
        before correlation analysis.

    Returns
    -------
    rot_ang_unoptimised : float
        Angle corresponding to the maximum sampled correlation value.

    rot_ang : float
        Refined rotation angle obtained using spline interpolation and
        local optimisation.

    rot_cor : ndarray
        Correlation coefficient calculated at each trial angle.

    """

    
    angles= np.arange(a_range[0], a_range[1]+0.5, 0.5)
    if a_range[1]-a_range[0] <=20:
         angles = np.arange(a_range[0], a_range[1]+0.2, 0.2) # gives angles to check (every 0.5 degrees within the range, or every 0.2 degrees if range is <20)
 
    
        
    im= im * (im>thresh)
    imR=imR * (imR>thresh)

       
    min_rows = min(im.shape[0], imR.shape[0])
    min_cols = min(im.shape[1], imR.shape[1])

    im  = im[:min_rows, :min_cols]
    imR = imR[:min_rows, :min_cols] 
    
    
    rot_cor= np.zeros(len(angles))
    
    for i in range(len(angles)):
        imRi = sk.transform.rotate(imR, angles[i], resize = False) #checked and imRi is 50x50 array
        
        rot_cor[i] = fn.corr2(imRi,im) #plot this to have a look at it for different noise levels. 
  
     
   
    max_index = np.argmax(rot_cor)
    rot_ang_unoptimised = angles[max_index]
   
    #F = sp.interpolate.interp1d(angles,rot_cor,kind='linear', fill_value='extrapolate')
    F = sp.interpolate.InterpolatedUnivariateSpline(angles,rot_cor,k=3)
    
    max_iterations = 50
    rot_ang = sp.optimize.fmin(lambda x: -F(x), angles[max_index], maxiter=max_iterations)
    #rot_ang = list(map(lambda x: sp.optimize.fmin(lambda angles: -F[angles],x),angles[max_index])) #improving precision to find max on graph of correlation factors for each angle
   
    
    if rot_ang >360:
        #rot_ang = [360]
        rot_ang = [np.nan]
    elif rot_ang <-360:
        #rot_ang = [-360]
        rot_ang = [np.nan]
  
    
  
    return [rot_ang_unoptimised, rot_ang, rot_cor]   

'''mean square error'''
def mse(img1, img2):
   h, w = img1.shape
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse

'''add parachuting in just the trace direction'''
def add_para3(img, para_grad):
    
    if para_grad == 0:
         return np.copy(img) 
     
    sobel_x = np.array([-1, 0, 1])
    result = np.copy(img)
    rows, cols = img.shape
    
    for i in range(rows):
        Li = img[i,:]
        x_gradient = convolve(Li, sobel_x, mode='same', method = 'direct')
        Gt = x_gradient>2*para_grad
        Gt[0] = False
        
        for k in np.where(Gt)[0]:
            count = k-1
            while count < cols - 1 and Li[count] >= img[i, count]:
                count +=1
                if count < cols:
                    Li[count] = max(Li[count-1]-para_grad, img[i, count])
                    #update Li only if the new value is greater than the original image value
                    #Use max() to ensure the new value is not less than the original image value
                    
        result[i, :] = Li
    return result


def convert_to_nx3_array(arr):
    n_rows, n_cols = arr.shape
    result = np.zeros((n_rows * n_cols, 3), dtype=float)
    index = 0
    for i in range(n_rows):
        for j in range(n_cols):
            result[index] = [i, j, arr[i][j]]
            index += 1
    return result

def powernoise(alpha, N):
    N2 = N // 2 
    f = np.arange(1, N2 + 1)
    A2 = 1.0 / (f ** (alpha / 2.0))

    p2 = (np.random.rand(N2) - 0.5) * 2 * np.pi
    d2 = A2 * np.exp(1j * p2)

    if N % 2 == 0:
        d = np.concatenate(([1], d2, [1 / (N2 ** (alpha / 2))], np.flipud(np.conj(d2))))
    else:
        d = np.concatenate(([1], d2, [1 / (N2 ** (alpha / 2))], np.flipud(np.conj(d2))))

    x = np.real(np.fft.ifft(d))
    return x[:N]