# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:41:37 2024

@author: pymte

Trying to make it work better by applying summed corrections to start image on each iteration, 
so that it is less blury
"""

import numpy as np
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.filters import window
from scipy.fft import fft2, fftshift
import skimage as sk
import cv2

plt.close('all')

EXAMPLE_FRAMES_SHOWN = 'yes' #want to see example images? input 'yes' or not yes

#load chosen stack 
pix_per_nm = 2.5
image_file = f'Rotate_drift_6J6J_f500_n0_s{pix_per_nm}_z0.5_Imagestack.npy'
drift_file = f'Rotate_drift_6J6J_f500_n0_s{pix_per_nm}_z0.5_Drifts.npy'
angle_file = f'Rotate_drift_6J6J_f500_n0_s{pix_per_nm}_z0.5_Angles.npy'

image_stack = np.load(image_file)
drift_array = np.load(drift_file)
angles_array=np.load(angle_file)

num_frames = image_stack.shape[2]
images = [image_stack[:, :, i] for i in range(num_frames)]
 
#the first frame is the reference image so separate this
  
ref_image = images[0] 
img_no = 253 #choose an image to test
start_image = images[img_no] 
    
AFM = np.load('lutAFM.npy')
AFM = ListedColormap(AFM)
if EXAMPLE_FRAMES_SHOWN == 'yes':
    fig1, ax1 = plt.subplots(1,2)
   
    ax1[0].imshow(ref_image, cmap = AFM)
    ax1[1].imshow(start_image, cmap = AFM)
    ax1[1].title.set_text(f'xshift={drift_array[img_no, 0]}, yshift={drift_array[img_no, 1]}, rotation={angles_array[img_no]}')
    plt.show() 

num_iterations = 2
corr_images=[]
phaseccr_shift_results_x=[]
phaseccr_shift_results_y=[]
angle_results=[]
corr_images_fresh=[]

for i in range(num_iterations):
    #PHASE CCR TRANSLATION..................................................................
    if i==0:
        image = start_image  
    if i >0:
        image = corr_images_fresh[i-1] # Use fresh version of previous corrected image (corrections applied to start image)
    phaseccr_shift, error, phasediff = phase_cross_correlation(ref_image, image, upsample_factor=100, normalization=None)
    phaseccr_shift_matrix = np.array([[1,0,phaseccr_shift[1]],[0,1,phaseccr_shift[0]]]) 
    
    driftcorr_image = cv2.warpAffine(image, phaseccr_shift_matrix, (ref_image.shape[1],ref_image.shape[0]))     
    phaseccr_shift_result_x = -phaseccr_shift[0]
    phaseccr_shift_result_y = -phaseccr_shift[1]
   
    phaseccr_shift_results_x.append(phaseccr_shift_result_x)
    phaseccr_shift_results_y.append(phaseccr_shift_result_y)
    #POLAR ROTATION (using image that has had drift corrected by phaseccr).........................................................................
        
    ref_polar=sk.transform.warp_polar(ref_image, radius=ref_image.shape[0])
    img_polar=sk.transform.warp_polar(driftcorr_image, radius=driftcorr_image.shape[0]) 
     
    angle_range = 30 #set range
    ref_polar_cropped= ref_polar[0:angle_range,:]
    img_polar_cropped= img_polar[0:angle_range,:]
     
    shifts, error, phasediff = phase_cross_correlation(ref_polar_cropped, img_polar_cropped, upsample_factor=100, normalization=None)
    angle_result=-shifts[0]  
    angle_results.append(angle_result)
    
    cols = driftcorr_image.shape[1]
    rows = driftcorr_image.shape[0]
    rotation_matrix = cv2.getRotationMatrix2D((cols/2,rows/2),angle_result,1) 
    corr_image= cv2.warpAffine(driftcorr_image, rotation_matrix,(cols,rows)) 
    corr_images.append(corr_image)
    
    
    #fresh image (apply total shifts so far to start image)
    print('iteration no.',i)
    print('Sum y:', sum(phaseccr_shift_results_y))
    print('Sum x:', sum(phaseccr_shift_results_x))
    print('Sum angle:', sum(angle_results))
    
    phaseccr_shift_matrix_total = np.array([[1,0,-sum(phaseccr_shift_results_y)],[0,1,-sum(phaseccr_shift_results_x)]]) 
    driftcorr_image_fresh = cv2.warpAffine(start_image, phaseccr_shift_matrix_total, (ref_image.shape[1],ref_image.shape[0])) 
    
    rotation_matrix_total = cv2.getRotationMatrix2D((cols/2,rows/2),sum(angle_results),1)
    corr_image_fresh = cv2.warpAffine(driftcorr_image_fresh, rotation_matrix_total,(cols,rows))
    corr_images_fresh.append(corr_image_fresh)

plt.figure()
plt.imshow(corr_images[0], cmap=AFM)
plt.figure()
plt.imshow(corr_images[num_iterations-1], cmap=AFM)
plt.figure()
plt.imshow(corr_images_fresh[num_iterations-1], cmap=AFM)

overall_shift_x = sum(phaseccr_shift_results_x)
overall_shift_y = sum(phaseccr_shift_results_y)
overall_rotation = sum(angle_results)
       
#......................................................................


print(f'number of iterations = {num_iterations}')
print(f'Expected value for x, y shift: {[drift_array[img_no, 0],drift_array[img_no, 1]]} pix')
print(f'Recovered values for x shift : {phaseccr_shift_results_x}. Sum (overall x shift to apply): {overall_shift_x} pixels')
print(f'Recovered values for y shift : {phaseccr_shift_results_y} Sum (overall y shift to apply): {overall_shift_y} pixels')

print('')
print(f'Expected value for rotation in degrees: {angles_array[img_no]} degrees')
print(f'Recovered values for rotation : {angle_results}. Sum (overall rotation to apply) = {overall_rotation} degrees')


print('')

print(f'Expected value for x, y shift: {[drift_array[img_no, 0],drift_array[img_no, 1]]} pix')
print(f'Cumulative recovered x shifts: {np.cumsum(phaseccr_shift_results_x)}')
print(f'Cumulative recovered y shifts: {np.cumsum(phaseccr_shift_results_y)}')
print(f'Overall x shift to apply = {overall_shift_x} pixels')
print(f'Overall y shift to apply = {overall_shift_y} pixels')

print(f'Expected value for rotation in degrees: {angles_array[img_no]} degrees')
print(f'Cumulative recovered rotations: {np.cumsum(angle_results)}')
print(f'Overall rotation to apply = {overall_rotation} degrees')


print(f'number of iterations = {num_iterations}')
print(f'final x error (pix) = {abs(drift_array[img_no, 0]-overall_shift_x):.3f}')
print(f'final y error (pix) = {abs(drift_array[img_no, 1]-overall_shift_y):.3f}')
print(f'final angle error (degrees) = {abs(angles_array[img_no]-overall_rotation):.3f}')
