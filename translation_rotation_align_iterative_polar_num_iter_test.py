# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 16:57:19 2026

@author: pymte
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

EXAMPLE_FRAMES_SHOWN = 'no' #want to see example images? input 'yes' or not yes

#load chosen stack 
pix_per_nm = 7.5
rotate_search_angle = 30 #set range for polar function


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


results={}

for num_iterations in range(1, 11):   
    
    shift_errors=[]
    angle_errors=[]
    for img_no in range(1, 101): # first 100 images after reference 
        start_image = images[img_no]         
        AFM = np.load('lutAFM.npy')
        AFM = ListedColormap(AFM)
        if EXAMPLE_FRAMES_SHOWN == 'yes':
            fig1, ax1 = plt.subplots(1,2)
           
            ax1[0].imshow(ref_image, cmap = AFM)
            ax1[1].imshow(start_image, cmap = AFM)
            ax1[1].title.set_text(f'xshift={drift_array[img_no, 0]}, yshift={drift_array[img_no, 1]}, rotation={angles_array[img_no]}')
            plt.show() 
        
    
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
             
    
            ref_polar_cropped= ref_polar[0:rotate_search_angle,:]
            img_polar_cropped= img_polar[0:rotate_search_angle,:]
             
            shifts, error, phasediff = phase_cross_correlation(ref_polar_cropped, img_polar_cropped, upsample_factor=100, normalization=None)
            angle_result=-shifts[0]  
            angle_results.append(angle_result)
            
            cols = driftcorr_image.shape[1]
            rows = driftcorr_image.shape[0]
            rotation_matrix = cv2.getRotationMatrix2D((cols/2,rows/2),angle_result,1) 
            corr_image= cv2.warpAffine(driftcorr_image, rotation_matrix,(cols,rows)) 
            corr_images.append(corr_image)
            
            #fresh image (apply total shifts so far to start image)
            #print('iteration no.',i)
            #print('Sum y:', sum(phaseccr_shift_results_y))
            #print('Sum x:', sum(phaseccr_shift_results_x))
            #print('Sum angle:', sum(angle_results))
            
            phaseccr_shift_matrix_total = np.array([[1,0,-sum(phaseccr_shift_results_y)],[0,1,-sum(phaseccr_shift_results_x)]]) 
            driftcorr_image_fresh = cv2.warpAffine(start_image, phaseccr_shift_matrix_total, (ref_image.shape[1],ref_image.shape[0])) 
            
            rotation_matrix_total = cv2.getRotationMatrix2D((cols/2,rows/2),sum(angle_results),1)
            corr_image_fresh = cv2.warpAffine(driftcorr_image_fresh, rotation_matrix_total,(cols,rows))
            corr_images_fresh.append(corr_image_fresh)
            
            overall_shift_x = sum(phaseccr_shift_results_x)
            overall_shift_y = sum(phaseccr_shift_results_y)
            overall_rotation = sum(angle_results) 
            
        #x_errors.append(abs(drift_array[img_no, 0] - overall_shift_x))
        #y_errors.append(abs(drift_array[img_no, 1] - overall_shift_y))
        
        shift_error = np.sqrt((drift_array[img_no, 0] - overall_shift_x)**2 +(drift_array[img_no, 1] - overall_shift_y)**2)
        angle_error = abs(angles_array[img_no] - overall_rotation)
        
        shift_errors.append(shift_error)     
        angle_errors.append(angle_error)
            
    
    results[num_iterations] = {
        'shift_mean': np.mean(shift_errors),
        'shift_std': np.std(shift_errors),
        'angle_mean': np.mean(angle_errors),
        'angle_std': np.std(angle_errors)
        }    
            
    print(f'Number of iterations = {num_iterations}')
    print(f'Mean shift error = {np.mean(shift_errors):.3f} ± {np.std(shift_errors):.3f} pixels')
    print(f'Mean angle error = {np.mean(angle_errors):.3f} ± {np.std(angle_errors):.3f} degrees')
    
    
iterations = list(results.keys())

shift_mean = [results[i]['shift_mean'] for i in iterations]
shift_std = [results[i]['shift_std'] for i in iterations]

angle_mean = [results[i]['angle_mean'] for i in iterations]
angle_std = [results[i]['angle_std'] for i in iterations]


fig, ax1 = plt.subplots(figsize=(4,3))
# Left axis: shift error
ax1.errorbar(
iterations,
shift_mean,
yerr=shift_std,
marker='o',
label='Shift error',
color='tab:blue'
)

ax1.set_xlabel('Number of iterations')
ax1.set_ylabel('Mean shift error (pixels)')

# Right axis: rotation error
ax2 = ax1.twinx()

ax2.errorbar(
iterations,
angle_mean,
yerr=angle_std,
marker='^',
label='Angle error',
color='tab:red'
)

ax2.set_ylabel('Mean angle error (degrees)')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines1 + lines2, labels1 + labels2)

plt.tight_layout()
plt.show()