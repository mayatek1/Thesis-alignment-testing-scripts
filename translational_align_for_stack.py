# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 11:27:37 2024

@author: pymte

Translational Align for Stack 4.0

ALigning each frame to the previous image rather than to one reference image
Option for power noise

"""

import numpy as np
import functions as fn
from skimage.registration import phase_cross_correlation
import cv2
import csv
from datetime import datetime
import statistics
from scipy.stats import sem
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os



#description of chosen stack to align
pdb_ID = '6uw4'
molecule_name = 'TRPV3_180'
pix_per_nm = 10#sampling
z_thresh = 0.8 #z thresh used for stack, needed to load file 

#Settings
run = 'Powernoise_prev_image_align'
#run = 'Powernoise_prev_image_align' #input value if repeating alignment run on same stack
EXAMPLE_FRAMES_SHOWN = 'yes' #want to see example images? input 'yes' or not yes

#Alternative image noise settings - put yes for only one. Default is gaussian noise
powernoise = 'yes' #add 0.1sdev powernoise
no_noise = 'no'

noise = 0.1 #standard deviation of gaussian noise in nm

label = f'{pdb_ID}{z_thresh}_{run}'

#create a folder to save the alignment value files into
folder_name = f'Gauss_s{pix_per_nm}_{label}'
folder_path = f'./{folder_name}'
os.makedirs(folder_path, exist_ok=True)

#load chosen stack 
path = f'Drift_stacks_and_values_{molecule_name}_{pdb_ID}'
#image_file = f'{path}/6J6J_f500_n0.0_s{pix_per_nm}_z{z_thresh}_stack.npy'
#drift_file = f'{path}/6J6J_f500_n0.0_s{pix_per_nm}_z{z_thresh}_drifts.npy'
image_file = f'{path}/{pdb_ID}_f500_n0.0_s{pix_per_nm}_z{z_thresh}_imagestack_{molecule_name}.npy'
drift_file = f'{path}/{pdb_ID}_f500_n0.0_s{pix_per_nm}_z{z_thresh}_driftvalues_{molecule_name}.npy'


image_stack = np.load(image_file)
drift_array = np.load(drift_file)
zeros = np.zeros((1, 2))
drift_array = np.vstack((zeros, drift_array)) #drift array imcluding the drift for the first frame

num_frames = image_stack.shape[2]
num_ref_frames = num_frames-1

applied_x_ref = drift_array[0:num_ref_frames, 0] 
applied_x = drift_array[1:num_frames, 0]

applied_y_ref = drift_array[0:num_ref_frames, 1] 
applied_y = drift_array[1:num_frames, 1]

applied_x_difference = applied_x - applied_x_ref
applied_y_difference = applied_y - applied_y_ref


ref_images = [image_stack[:, :, i] for i in range(num_frames-1)] #images excluding final frame
images = [image_stack[:, :,1+i] for i in range(num_frames-1)] #images excluding first frame



AFM = np.load('lutAFM.npy')
AFM = ListedColormap(AFM)

#apply Gaussian filter
#sigma_value_nm = [0, 0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200, 0.225, 0.250, 0.275, 0.300] 
#sigma_value_pix = [0]
sigma_value_pix = [0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]

#now going through certain pixel values
sigma_value_nm = [x/pix_per_nm for x in sigma_value_pix]

phaseccr_MD_list = []   #MD = measured difference
phaseccr_sdev_list = []
phaseccr_sem_list = []

ccr_MD_list = []
ccr_sdev_list = []
ccr_sem_list = []
example_image_list=[]

#image dimensions for power noise
xlen = images[0].shape[0] 
ylen = images[0].shape[1] 
mf100 = 3.4               #multiplication factor for image of line length 100
lldiff = xlen-100         #line length difference
mf = mf100 + lldiff*0.03  #multiplication factor

powernoise_stack = []
for j in range(len(images)): 
    refsizeimg = np.zeros((xlen, ylen))      
    for i in range(0,xlen):
        pnoise = fn.powernoise(1,ylen)
        refsizeimg[i, :] = pnoise * mf
    powernoise_stack.append(refsizeimg)


for s in range(len(sigma_value_pix)):
    images_n = [image + np.random.normal(0, noise, images[0].shape) for image in images] #default gaussian noise 
    ref_images_n = [ref_image + np.random.normal(0, noise, images[0].shape) for ref_image in ref_images]

    if powernoise =='yes':  #power noise
        images_n = [image + powernoiseimg for image, powernoiseimg in zip(images, powernoise_stack)]
    
    if no_noise == 'yes':   #no noise
        images_n = images
    
    images_n_gauss= [gaussian_filter(image_n, sigma_value_pix[s]) for image_n in images_n]    
    ref_images_n_gauss = [gaussian_filter(ref_image_n, sigma_value_pix[s]) for ref_image_n in ref_images_n] 
    
    stacked_images = np.stack(images_n_gauss, axis=2)
    file_name_npy = f'{pdb_ID}_f500_s{pix_per_nm}_g{sigma_value_pix[s]}pix_{sigma_value_nm[s]}nm_noise{noise}_stack_{run}.npy'
    file_path_npy = os.path.join(folder_path, file_name_npy)
    np.save(file_path_npy, stacked_images)  
        

    #Check appearance of noise
    AFM = np.load('lutAFM.npy')
    AFM = ListedColormap(AFM)
    if EXAMPLE_FRAMES_SHOWN == 'yes':
        fig1, ax1 = plt.subplots(2,3)
        plt.suptitle(f'First few frames, check noise. s={pix_per_nm}pix/nm, sigma = {sigma_value_pix[s]}pix')
    
        ax1[0,0].imshow(images_n_gauss[0], cmap = AFM)       
        ax1[0,1].imshow(images_n_gauss[1], cmap = AFM)
        ax1[0,2].imshow(images_n_gauss[2], cmap = AFM)
        ax1[1,0].imshow(images_n_gauss[3], cmap = AFM)
        ax1[1,1].imshow(images_n_gauss[4], cmap = AFM)
        ax1[1,2].imshow(images_n_gauss[5], cmap = AFM)
    
        plt.show()
    
    #calculate drifts. Here, aligning with noise version of the previous image
    
    #1) phase cross correlation method ........................................................................
    measured_drift = []
    correction_images= []
     
    for i in range(len(images)):
         phaseccr_shift, error, phasediff = phase_cross_correlation(ref_images_n_gauss[i], images_n_gauss[i], upsample_factor=100, normalization=None)
         phaseccr_shift_matrix = np.array([[1,0,phaseccr_shift[1]],[0,1,phaseccr_shift[0]]]) 
     
         phaseccr_correction = cv2.warpAffine(images[i], phaseccr_shift_matrix, (images[0].shape[1],images[0].shape[0]))
         correction_images.append(phaseccr_correction)
         
         #phaseccr_shift[0], phaseccr_shift[1] = phaseccr_shift[1], phaseccr_shift[0]
         phaseccr_shift_result = [-x for x in phaseccr_shift]
         
         measured_drift.append(list(phaseccr_shift_result))
         if i % 10 == 0:
            
             print(f'\rNumber of shifts calculated and corrected: {i}', end='', flush=True)
     
    # 2) Cross corrlation method...........................................................................
     
    measured_drift_ccr = []
     
    for i in range(len(images)):
         re =100; w=2 #re should be the size of c. which seems to be 99 not 100? c.shape[0] = 99
         c = fn.normxcorr2(images_n_gauss[i], ref_images_n_gauss[i]) #function from dylan. generates cross correlation image. 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
         
         indices = np.where(c == c.max())
         
         corr_shift_1 = [indices[0][0]-(c.shape[0]/2), indices[1][0]-(c.shape[1]/2)]
         
         #now zoom in to find more precise maximum
         
         clip_c = c[indices[0][0]-w:indices[0][0]+w +1,indices[1][0]-w:indices[1][0]+w +1]  
         
         c_zoom = cv2.resize(clip_c, None, fx = re, fy = re, interpolation = cv2.INTER_CUBIC)
       
         indices2 = np.where(c_zoom == c_zoom.max())
         
         
         ccr_shift_2 = [(indices2[0][0]-c_zoom.shape[0]/2)/re + 1/re, (indices2[1][0]-c_zoom.shape[1]/2)/re + 1/re]    
         
         
         ccr_shift = [(indices[0][0]+ccr_shift_2[0]-images[0].shape[0]/2), (indices[1][0]+ccr_shift_2[1]-images[0].shape[1]/2)]
         ccr_shift_matrix = np.array([[1,0,ccr_shift[1]],[0,1,ccr_shift[0]]]) 
         
         ccr_shift_result = [-x for x in ccr_shift]
         measured_drift_ccr.append(list(ccr_shift_result))
         
    #print(drift_list)
    #print(measured_drift_ccr)   
     
    #put x and y values into separate lists, and find the difference between applied x,y and measured x,y....................................
    
    phaseccr_x, phaseccr_y = zip(*measured_drift)
    ccr_x, ccr_y = zip(*measured_drift_ccr)
     
    phaseccr_diff_x= [applied_x_difference[i] - phaseccr_x[i] for i in range(len(applied_x_difference))]
    phaseccr_diff_y =  [applied_y_difference[i] - phaseccr_y[i] for i in range(len(applied_y_difference))]
    ccr_diff_x=[applied_x_difference[i] - ccr_x[i] for i in range(len(applied_x_difference))]
    ccr_diff_y=[applied_y_difference[i] - ccr_y[i] for i in range(len(applied_y_difference))]
     
    phaseccr_MD = [np.sqrt(phaseccr_diff_x[i]**2 + phaseccr_diff_y[i]**2) for i in range(len(phaseccr_diff_x))]  #measured difference: sqrt(delta x ^2 + delta y ^2)
    ccr_MD = [np.sqrt(ccr_diff_x[i]**2 + ccr_diff_y[i]**2) for i in range(len(ccr_diff_x))]
     
    drift_mse = []
    for i in range(len(correction_images)):
         mean_square_error=fn.mse(correction_images[i], images[i])
         mean_square_error=round(mean_square_error, 4)
         drift_mse.append(mean_square_error)
     
     
    #Save the drift, measured drift, difference and mean square error in an excel file...............................................
     
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    file_name1 = f'drift_{pdb_ID}_s{pix_per_nm}_g{sigma_value_pix[s]}pix_{sigma_value_nm[s]}nm_noise{noise}_{run}.csv'
    file_path1 = os.path.join(folder_path, file_name1)
    with open(file_path1, 'w', newline='') as csvfile:
         fieldnames = ['Applied_X_diff', 'Applied_Y_diff', 'phaseccr_X', 'phaseccr_Y', 'phaseccr_diff_X', 'phaseccr_diff_Y', 'phaseccr_MD', 'MSE', 'ccr_X', 'ccr_Y', 'ccr_diff_X', 'ccr_diff_Y', 'ccr_MD']
         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
     
         writer.writeheader()
         for ax, ay, m1x, m1y, d1x, d1y, MD1, mse1, m2x, m2y, d2x, d2y, MD2 in zip(applied_x_difference, applied_y_difference, phaseccr_x, phaseccr_y, phaseccr_diff_x, phaseccr_diff_y, phaseccr_MD, drift_mse, ccr_x, ccr_y, ccr_diff_x, ccr_diff_y, ccr_MD):
             writer.writerow({'Applied_X_diff': ax, 'Applied_Y_diff': ay, 'phaseccr_X': m1x, 'phaseccr_Y': m1y, 'phaseccr_diff_X': d1x, 'phaseccr_diff_Y': d1y, 'phaseccr_MD' : MD1, 'MSE': mse1, 'ccr_X': m2x, 'ccr_Y': m2y, 'ccr_diff_X': d2x, 'ccr_diff_Y': d2y, 'ccr_MD' : MD2 })
     
    print('\nData saved to', (file_name1))
    
    av_phaseccr_MD =  statistics.mean(phaseccr_MD)
    av_ccr_MD =  statistics.mean(ccr_MD)
    
    #Errors (standard error in the mean) 
    error_av_phaseccr_MD= sem(phaseccr_MD)
    error_av_ccr_MD= sem(ccr_MD)
     
    #standard deviation
    
    std_phaseccr_MD = np.std(phaseccr_MD)
    std_ccr_MD = np.std(ccr_MD)
    
    print(f'Gaussian filter strength = {sigma_value_nm[s]}nm, {sigma_value_pix[s]}pix') 
    print('\nphaseccr average difference=', av_phaseccr_MD, '\nstandard deviation =', std_phaseccr_MD, '\nSEM =', error_av_phaseccr_MD)
    print('\nccr average difference =', av_ccr_MD, '\nstandard deviation =', std_ccr_MD, '\nSEM =', error_av_ccr_MD)
    
    phaseccr_MD_list.append(av_phaseccr_MD)
    phaseccr_sdev_list.append(std_phaseccr_MD)
    phaseccr_sem_list.append(error_av_phaseccr_MD)

    ccr_MD_list.append(av_ccr_MD)
    ccr_sdev_list.append(std_ccr_MD)
    ccr_sem_list.append(error_av_ccr_MD)  
  

folder_path_allcomparison = f'Drift_all_sampling_comparison_{molecule_name}_{pdb_ID}'
if not os.path.exists(folder_path_allcomparison):
        os.makedirs(folder_path_allcomparison, exist_ok=True) 

file_name2 = f'drift_{pdb_ID}_s{pix_per_nm}_noise{noise}_all_filter_strengths_{run}.csv' 
file_path2 = os.path.join(folder_path, file_name2)
file_path_allcomparison = os.path.join(folder_path_allcomparison, file_name2) 

with open(file_path2, 'w', newline='') as csvfile:
     fieldnames = ['Gauss_nm', 'Gauss_pix', 'phaseccr_MD', 'phaseccr_sdev', 'phaseccr_SEM', 'ccr_MD', 'ccr_sdev', 'ccr_SEM']
     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
 
     writer.writeheader()
     for nm, pix, pc, pc_std, pc_sem, c, c_std, c_sem in zip(sigma_value_nm, sigma_value_pix, phaseccr_MD_list, phaseccr_sdev_list, phaseccr_sem_list, ccr_MD_list, ccr_sdev_list, ccr_sem_list):
         writer.writerow({'Gauss_nm': nm, 'Gauss_pix': pix, 'phaseccr_MD': pc, 'phaseccr_sdev': pc_std, 'phaseccr_SEM': pc_sem, 'ccr_MD': c, 'ccr_sdev' : c_std, 'ccr_SEM': c_sem})
 
print('\nData saved to', (file_name2))

with open((file_path_allcomparison), 'w', newline='') as csvfile:
     fieldnames = ['Gauss_nm', 'Gauss_pix', 'phaseccr_MD', 'phaseccr_sdev', 'phaseccr_SEM', 'ccr_MD', 'ccr_sdev', 'ccr_SEM']
     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
 
     writer.writeheader()
     for nm, pix, pc, pc_std, pc_sem, c, c_std, c_sem in zip(sigma_value_nm, sigma_value_pix, phaseccr_MD_list, phaseccr_sdev_list, phaseccr_sem_list, ccr_MD_list, ccr_sdev_list, ccr_sem_list):
         writer.writerow({'Gauss_nm': nm, 'Gauss_pix': pix, 'phaseccr_MD': pc, 'phaseccr_sdev': pc_std, 'phaseccr_SEM': pc_sem, 'ccr_MD': c, 'ccr_sdev' : c_std, 'ccr_SEM': c_sem})


