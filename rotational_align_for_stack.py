# -*- coding: utf-8 -*-

"""
Created on Mon Dec 18 16:44:05 2023

@author: pymte


"""

import numpy as np
from functions import NanoLocz_ccr_rot_align
import functions_update as fn
from skimage.registration import phase_cross_correlation
import csv
import statistics
from scipy.stats import sem
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import skimage as sk
import os


plt.close('all')

label = 'Powernoise_ccr100_polar100' #if repeating measurements put in run number/ a label

sampling_values = [1.25, 2.5, 3.75, 5.0, 6.25, 7.5, 8.75, 10]
for pix_per_nm in sampling_values:
    #information to load chosen files
    pdb_id = '2IWV' #6uw4 6J6J 2IWV
    z_thresh = 0.7 #07 0.5 0.8
    molecule_name = 'OmpG' #Streptavidin TRPV3_180 OmpG
    applied_angle_range = 100
    
    #set angle range for alignment functions!
    polar_angle_range = 180
    ccr_angle_range = 180
    
    #set noise and parachuting
    
    noise = 0.1
    para_angle = 0
    para_grad = 1/(np.tan(np.deg2rad(para_angle))*pix_per_nm)
    
    GAUSS_PIX_OR_NM = 'pix'
    EXAMPLE_FRAMES_SHOWN = 'yes'
    ccr_method='yes'
    powernoise = 'yes'
    noisy_reference = 'no'
    
    #create a folder to save the alignment value files into
    folder_name = f'Gauss_s{pix_per_nm}_{molecule_name}_{pdb_id}_{label}'
    folder_path = f'./{folder_name}'
    os.makedirs(folder_path, exist_ok=True)
    
    #load chosen stack and corresponding applied angles
    image_file = f'Rotation_{pdb_id}_f500_n0_s{pix_per_nm}_z{z_thresh}_imagestack_{molecule_name}_anglerange100.npy'  
    rotation_file = f'Rotation_{pdb_id}_f500_n0_s{pix_per_nm}_z{z_thresh}_anglevalues_{molecule_name}_anglerange100.npy'
    #image_file = f'Rotation_{pdb_id}_f500_n0_s{pix_per_nm}_z{z_thresh}_Imagestack.npy' #streptavidin (earlier version of file name)
    #rotation_file = f'Rotation_{pdb_id}_f500_n0_s{pix_per_nm}_z{z_thresh}_Angles.npy'
    
    image_stack = np.load(image_file)
    applied_angles = np.load(rotation_file)
    applied_angles = applied_angles[1:] #first number in list is 0 for the ref image so remove this because this image is se_ted from the list later
    
    num_frames = image_stack.shape[2]
    images = [image_stack[:, :, i] for i in range(num_frames)]
     
    images_para = [fn.add_para3(image, para_grad)  for image in images]
    
    #apply Gaussian filter
    
    if GAUSS_PIX_OR_NM == 'nm':
        sigma_value_nm = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        sigma_value_pix= [x*pix_per_nm for x in sigma_value_nm]
        
    
    if GAUSS_PIX_OR_NM == 'pix':
        sigma_value_pix = [0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]
        sigma_value_nm = [x/pix_per_nm for x in sigma_value_pix]
    
    sigma_value_nm_r = [round(x, 2) for x in sigma_value_nm]
    sigma_value_pix_r = [round(x, 2) for x in sigma_value_pix]
    
    polar_diff_list = []   
    polar_sdev_list = []
    polar_sem_list = []
    
    ccr_diff_list = []
    ccr_sdev_list = []
    ccr_sem_list = []
    
    example_image_list=[]
    
    #image dimensions for power noise
    x = images[0].shape[0] 
    y = images[0].shape[1] 
    mf100 = 3.4               #multiplication factor for image of line length 100
    lldiff = x-100            #line length difference
    mf = mf100 + lldiff*0.03  #multiplication factor
    
    powernoise_stack = []
    for j in range(len(images)): 
        refsizeimg = np.zeros((x, y))      
        for i in range(0,x):
            pnoise = fn.powernoise(1,y)
            refsizeimg[i, :] = pnoise * mf
        powernoise_stack.append(refsizeimg)
    
    for s in range(len(sigma_value_pix)):
        if powernoise == 'no':
            images_n = [image + np.random.normal(0, noise, images[0].shape) for image in images_para]
       
        if powernoise =='yes':
            images_n = [image + powernoiseimg for image, powernoiseimg in zip(images_para, powernoise_stack)]
                               
        images_n_gauss= [gaussian_filter(image_n, sigma_value_pix[s]) for image_n in images_n]    
                
        stacked_images = np.stack(images_n_gauss, axis=2)
        file_name_npy = f'Rotation_{pdb_id}_f500_s{pix_per_nm}_g{sigma_value_pix[s]}pix_{sigma_value_nm[s]}nm_noise{noise}_stack_{molecule_name}{label}.npy'
        file_path_npy = os.path.join(folder_path, file_name_npy)
        np.save(file_path_npy, stacked_images)
         
        #the first frame is the reference image so separate this
        if noisy_reference == 'yes':
            ref_image = images_n_gauss[0]
        else:        
            ref_image = gaussian_filter(images[0], sigma_value_pix[s])
        rotation_images_n = images_n_gauss[1:]
         
        #Check appearance of noise
        AFM = np.load('lutAFM.npy')
        AFM = ListedColormap(AFM)
        if EXAMPLE_FRAMES_SHOWN == 'yes':
            fig1, ax1 = plt.subplots(2,3)
            plt.suptitle(f'First few frames. Sampling = {pix_per_nm}pix/nm, Sigma = {sigma_value_pix[s]}')
            
            ax1[0,0].imshow(ref_image, cmap = AFM)
            ax1[0,1].imshow(rotation_images_n[0], cmap = AFM)
            ax1[0,2].imshow(rotation_images_n[1], cmap = AFM)
            ax1[1,0].imshow(rotation_images_n[2], cmap = AFM)
            ax1[1,1].imshow(rotation_images_n[3], cmap = AFM)
            ax1[1,2].imshow(rotation_images_n[4], cmap = AFM)
            
            plt.show()
       
        
        #calculate drifts
        
        #1) polar method ........................................................................
        
        polar_images =[]
        MA_polar = []  #Measured Angle
        
        for i in range(len(rotation_images_n)):
            sd = rotation_images_n[i].shape    
            ref_polar=sk.transform.warp_polar(ref_image, radius=ref_image.shape[0])
            img_polar=sk.transform.warp_polar(rotation_images_n[i], radius=sd[0]) 
                     
            ref_polar_cropped= ref_polar[0:polar_angle_range,:]
            img_polar_cropped= img_polar[0:polar_angle_range,:]
            polar_images.append(img_polar_cropped)
             
            shifts, error, phasediff = phase_cross_correlation(ref_polar_cropped, img_polar_cropped, upsample_factor=100, normalization=None)
            MA_polar.append(-shifts[0])
            if i % 10 == 0:
               
                print(f'\rNumber of rotations calculated and corrected: {i}', end='', flush=True) 
             
        
        polar_diff = [applied_angles[i] - MA_polar[i]for i in range(len(applied_angles))]
           
        #2) rot ccr method ............................................................................
       
        if ccr_method=='yes': 
            MA_ccr = []
            for j in range(len(rotation_images_n)):
                ccr_angle_range_list = [-ccr_angle_range/2, ccr_angle_range/2]        
                [angle_unoptimised, ccr_angle, ccr_plot, rotated_image] = NanoLocz_ccr_rot_align(ref_image, rotation_images_n[j], ccr_angle_range_list,0.1,1) #trying different range, works with [-180,180] if this doesn't work
                if ccr_angle[0] <= ccr_angle_range_list[0] or ccr_angle[0] >= ccr_angle_range_list[1]:
                    ccr_angle[0] = np.nan
                print(ccr_angle)
                ccr_angle = ccr_angle[0]
                MA_ccr.append(ccr_angle)
                
            ccr_diff = [applied_angles[i] - MA_ccr[i]for i in range(len(applied_angles))]
        else:
            ccr_diff = [0 for i in range(len(applied_angles))]
            MA_ccr = [0 for i in range(len(applied_angles))]
            
        if EXAMPLE_FRAMES_SHOWN == 'yes' and ccr_method=='yes':
            plt.figure()
            plt.plot(ccr_plot, linewidth = 1)#(hopefully) plot ccr plot of last rotated image in the stack, for each filter strength
            plt.title(f'Example curve for img with sigma={sigma_value_pix[s]} (for frame{j}). s={pix_per_nm}')
        
        
        #save the angles, measured angles and polar difference in an excel file........................................................................................
        
        file_name1 = f'rotation_{pdb_id}_s{pix_per_nm}_g{sigma_value_pix[s]}pix_{sigma_value_nm[s]}nm_noise{noise}_{molecule_name}{label}.csv' 
        file_path1 = os.path.join(folder_path, file_name1)
        with open((file_path1), 'w', newline='') as csvfile:
             fieldnames = ['Applied_angle', 'Measured_angle_polar', 'Difference_polar', 'Measured_angle_ccr', 'Difference_ccr']
             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
             writer.writeheader()
             for AA, MAP, DP, MAC, DC in zip(applied_angles, MA_polar, polar_diff, MA_ccr, ccr_diff):
                 writer.writerow({'Applied_angle': AA, 'Measured_angle_polar': MAP, 'Difference_polar': DP, 'Measured_angle_ccr': MAC, 'Difference_ccr': DC})
        
        print('\nData saved to', (file_name1))
        print('Number of frames: ', (num_frames))
        
        abs_polar_diff = [abs(i) for i in polar_diff]
        abs_ccr_diff = [abs(i) for i in ccr_diff] #need to use absolute value or +ve and -ve differences cancel each other out when mean is found?
        
        av_polar_diff = statistics.mean(abs_polar_diff)
        av_ccr_diff = statistics.mean(abs_ccr_diff)
        #Errors (standard error in the mean) 
        sem_polar_diff= sem(abs_polar_diff)
        sem_ccr_diff = sem(abs_ccr_diff)
        #standard deviation
        sdev_polar_diff = np.std(polar_diff)
        sdev_ccr_diff = np.std(ccr_diff)
           
        polar_diff_list.append(av_polar_diff)
        polar_sdev_list.append(sdev_polar_diff)
        polar_sem_list.append(sem_polar_diff) 
        
        ccr_diff_list.append(av_ccr_diff)
        ccr_sdev_list.append(sdev_ccr_diff)
        ccr_sem_list.append(sem_ccr_diff) 
        
        example_image_list.append(ref_image)    
          
      
    
    
    
    
    folder_path_allcomparison = f'Rotation_all_sampling_comparison_{molecule_name}_{pdb_id}'
    if not os.path.exists(folder_path_allcomparison):
            os.makedirs(folder_path_allcomparison, exist_ok=True)
    
    
    file_name2 = f'Rotation_{pdb_id}_s{pix_per_nm}_noise{noise}_all_filter_strengths_{molecule_name}{label}.csv'
    file_path2 = os.path.join(folder_path, file_name2) 
    file_path_allcomparison = os.path.join(folder_path_allcomparison, file_name2) 
    
    with open((file_path2), 'w', newline='') as csvfile:
         fieldnames = ['Gauss_nm', 'Gauss_pix', 'Polar_diff', 'Polar_sdev', 'Polar_SEM', 'ccr_diff', 'ccr_sdev', 'ccr_SEM']
         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
     
         writer.writeheader()
         for nm, pix, pol, pol_sdev, pol_sem, ccr, ccr_sdev, ccr_sem in zip(sigma_value_nm, sigma_value_pix, polar_diff_list, polar_sdev_list, polar_sem_list, ccr_diff_list, ccr_sdev_list, ccr_sem_list):
             writer.writerow({'Gauss_nm': nm, 'Gauss_pix': pix, 'Polar_diff': pol, 'Polar_sdev': pol_sdev, 'Polar_SEM': pol_sem, 'ccr_diff': ccr, 'ccr_sdev': ccr_sdev, 'ccr_SEM': ccr_sem})
     
    print('\nData saved to', (file_name2))
    
    with open((file_path_allcomparison), 'w', newline='') as csvfile:
         fieldnames = ['Gauss_nm', 'Gauss_pix', 'Polar_diff', 'Polar_sdev', 'Polar_SEM', 'ccr_diff', 'ccr_sdev', 'ccr_SEM']
         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
     
         writer.writeheader()
         for nm, pix, pol, pol_sdev, pol_sem, ccr, ccr_sdev, ccr_sem in zip(sigma_value_nm, sigma_value_pix, polar_diff_list, polar_sdev_list, polar_sem_list, ccr_diff_list, ccr_sdev_list, ccr_sem_list):
             writer.writerow({'Gauss_nm': nm, 'Gauss_pix': pix, 'Polar_diff': pol, 'Polar_sdev': pol_sdev, 'Polar_SEM': pol_sem, 'ccr_diff': ccr, 'ccr_sdev': ccr_sdev, 'ccr_SEM': ccr_sem})
     
    
    #figure
    AFM = np.load('lutAFM.npy')
    AFM = ListedColormap(AFM)
      
    fig, ax = plt.subplots(3,7)
    plt.suptitle(f'Sampling = {pix_per_nm}pix/nm, Standard deviation of noise = {noise} nm ')

ax[0,0].imshow(example_image_list[0], cmap = AFM)
ax[0,0].title.set_text(f'g = {sigma_value_pix[0]} pix')
ax[0,1].imshow(example_image_list[1], cmap = AFM)
ax[0,1].title.set_text(f'g = {sigma_value_pix[1]} pix')
ax[0,2].imshow(example_image_list[2], cmap = AFM)
ax[0,2].title.set_text(f'g = {sigma_value_pix[2]} pix')
ax[0,3].imshow(example_image_list[3], cmap = AFM)
ax[0,3].title.set_text(f'g = {sigma_value_pix[3]} pix')
ax[0,4].imshow(example_image_list[4], cmap = AFM)
ax[0,4].title.set_text(f'g = {sigma_value_pix[4]} pix')
ax[0,5].imshow(example_image_list[5], cmap = AFM)
ax[0,5].title.set_text(f'g = {sigma_value_pix[5]} pix')
ax[0,6].imshow(example_image_list[6], cmap = AFM)
ax[0,6].title.set_text(f'g = {sigma_value_pix[6]} pix')
ax[1,0].imshow(example_image_list[7], cmap = AFM)
ax[1,0].title.set_text(f'g = {sigma_value_pix[7]} pix')
ax[1,1].imshow(example_image_list[8], cmap = AFM)
ax[1,1].title.set_text(f'g = {sigma_value_pix[8]} pix')
ax[1,2].imshow(example_image_list[9], cmap = AFM)
ax[1,2].title.set_text(f'g = {sigma_value_pix[9]} pix')
ax[1,3].imshow(example_image_list[10], cmap = AFM)
ax[1,3].title.set_text(f'g = {sigma_value_pix[10]} pix')
ax[1,4].imshow(example_image_list[11], cmap = AFM)
ax[1,4].title.set_text(f'g = {sigma_value_pix[11]} pix')
ax[1,5].imshow(example_image_list[12], cmap = AFM)
ax[1,5].title.set_text(f'g = {sigma_value_pix[12]} pix')
ax[1,6].imshow(example_image_list[13], cmap = AFM)
ax[1,6].title.set_text(f'g = {sigma_value_pix[13]} pix')
ax[2,0].imshow(example_image_list[14], cmap = AFM)
ax[2,0].title.set_text(f'g = {sigma_value_pix[14]} pix')
ax[2,1].imshow(example_image_list[15], cmap = AFM)
ax[2,1].title.set_text(f'g = {sigma_value_pix[15]} pix')
ax[2,2].imshow(example_image_list[16], cmap = AFM)
ax[2,2].title.set_text(f'g = {sigma_value_pix[16]} pix')
ax[2,3].imshow(example_image_list[17], cmap = AFM)
ax[2,3].title.set_text(f'g = {sigma_value_pix[17]} pix')
ax[2,4].imshow(example_image_list[18], cmap = AFM)
ax[2,4].title.set_text(f'g = {sigma_value_pix[18]} pix')
ax[2,5].imshow(example_image_list[19], cmap = AFM)
ax[2,5].title.set_text(f'g = {sigma_value_pix[19]} pix')
ax[2,6].imshow(example_image_list[20], cmap = AFM)
ax[2,6].title.set_text(f'g = {sigma_value_pix[20]} pix')

plt.tight_layout()
plt.show() #NOTE: SAVE THE FIGURE MANUALLY UNTIL I SORT OUT THE LAYOUT PROBLEM

#plt.savefig(f'Example_frames_{timestamp}.png')
