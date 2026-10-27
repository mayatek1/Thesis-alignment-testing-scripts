# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 19:38:10 2024

@author: pymte
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



run = '1' #if repeating measurements put in run number or label !!
EXAMPLE_FRAMES_SHOWN = 'no' #want to see example images? input 'yes' or not yes
pix_per_nm = 1.25 #REMEMBER TO INPUT THIS. s (sampling) value in the image file name

#load chosen stack 
image_file = 'DRIFT2nm_6J6J_f500_n0.0_s1.25_z0.5_stack_fluctuations.npy'
drift_file = 'DRIFT2nm_6J6J_f500_n0.0_s1.25_z0.5_drifts_fluctuations.npy'
noise = 0.1 #noise in nm

image_stack = np.load(image_file)
drift_array = np.load(drift_file)

num_frames = image_stack.shape[2]
images = [image_stack[:, :, i] for i in range(num_frames)]
 
images_n = [image + np.random.normal(0, noise, images[0].shape) for image in images]

#apply Gaussian filter

#sigma_value_nm = [0, 0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200, 0.225, 0.250, 0.275, 0.300] 

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

for s in range(len(sigma_value_pix)):
    images_n_gauss = []
    for im in range(len(images_n)):
        images_n_gauss1 = gaussian_filter(images_n[im], sigma_value_pix[s]) 
        images_n_gauss.append(images_n_gauss1)    
       
    stacked_images = np.stack(images_n_gauss, axis=2)
    np.save(f'6J6J_f500_s{pix_per_nm}_g{sigma_value_pix[s]}pix_{sigma_value_nm[s]}nm_noise{noise}_stack_{run}.npy', stacked_images)  
       
    #the first frame is the reference image so separate this
    ref_image_n = images_n_gauss[0]
    drift_images_n = images_n_gauss[1:]
        
    #Check appearance of noise
    AFM = np.load('lutAFM.npy')
    AFM = ListedColormap(AFM)
    if EXAMPLE_FRAMES_SHOWN == 'yes':
        fig1, ax1 = plt.subplots(2,3)
        plt.suptitle('First few frames, check noise')
    
        ax1[0,0].imshow(ref_image_n, cmap = AFM)
        ax1[0,1].imshow(drift_images_n[0], cmap = AFM)
        ax1[0,2].imshow(drift_images_n[1], cmap = AFM)
        ax1[1,0].imshow(drift_images_n[2], cmap = AFM)
        ax1[1,1].imshow(drift_images_n[3], cmap = AFM)
        ax1[1,2].imshow(drift_images_n[4], cmap = AFM)
    
        plt.show()
      
    #calculate drifts
    
    #1) phase cross correlation method ........................................................................
    measured_drift = []
    correction_images= []
     
    for i in range(len(drift_images_n)):
         phaseccr_shift, error, phasediff = phase_cross_correlation(ref_image_n, drift_images_n[i], upsample_factor=100, normalization=None)
         phaseccr_shift_matrix = np.array([[1,0,phaseccr_shift[1]],[0,1,phaseccr_shift[0]]]) 
     
         phaseccr_correction = cv2.warpAffine(drift_images_n[i], phaseccr_shift_matrix, (ref_image_n.shape[1],ref_image_n.shape[0]))
         correction_images.append(phaseccr_correction)
         
    #create an image which is the average of all of the aligned particles to use as the new reference 
    sum_image = np.zeros((correction_images[0].shape[0], correction_images[0].shape[1]), dtype=np.uint8)
    for image in correction_images:
        sum_image = np.add(sum_image, image)
    sum_image= sum_image/500
        
    AFM = np.load('lutAFM.npy')
    AFM = ListedColormap(AFM)
    
    if EXAMPLE_FRAMES_SHOWN == 'yes':
        plt.figure()
        plt.imshow(sum_image, cmap=AFM)
        plt.title(f'phaseccr averaged reference image, sigma = {s}')
    
  
    for i in range(len(drift_images_n)): 
        phaseccr_shift2, error, phasediff = phase_cross_correlation(sum_image, drift_images_n[i], upsample_factor=100, normalization=None)
        phaseccr_shift_matrix = np.array([[1,0,phaseccr_shift2[1]],[0,1,phaseccr_shift2[0]]])                 
    
        phaseccr_shift_result = [-x for x in phaseccr_shift2]
         
        measured_drift.append(list(phaseccr_shift_result))
        if i % 10 == 0:
            
             print(f'\rNumber of shifts calculated and corrected: {i}', end='', flush=True)
             
             
                 
     
    # 2) Cross corrlation method...........................................................................
     
    measured_drift_ccr = []
    correction_images2=[]
     
    for i in range(len(drift_images_n)):
         re =100; w=2 #re should be the size of c. which seems to be 99 not 100? c.shape[0] = 99
         c = fn.normxcorr2(drift_images_n[i], ref_image_n) #function from dylan. generates cross correlation image. 
         indices = np.where(c == c.max())
         
         corr_shift_1 = [indices[0][0]-(c.shape[0]/2), indices[1][0]-(c.shape[1]/2)]
         
         #now zoom in to find more precise maximum         
         clip_c = c[indices[0][0]-w:indices[0][0]+w +1,indices[1][0]-w:indices[1][0]+w +1]           
         c_zoom = cv2.resize(clip_c, None, fx = re, fy = re, interpolation = cv2.INTER_CUBIC)      
         indices2 = np.where(c_zoom == c_zoom.max())
                  
         ccr_shift_2 = [(indices2[0][0]-c_zoom.shape[0]/2)/re + 1/re, (indices2[1][0]-c_zoom.shape[1]/2)/re + 1/re]    
                  
         ccr_shift = [(indices[0][0]+ccr_shift_2[0]-ref_image_n.shape[0]/2), (indices[1][0]+ccr_shift_2[1]-ref_image_n.shape[1]/2)]
         ccr_shift_matrix = np.array([[1,0,ccr_shift[1]],[0,1,ccr_shift[0]]]) 
         
         ccr_correction = cv2.warpAffine(drift_images_n[i], ccr_shift_matrix, (ref_image_n.shape[1],ref_image_n.shape[0]))
         correction_images2.append(ccr_correction)
         
    #create an image which is the average of all of the aligned particles to use as the new reference 
    sum_image2 = np.zeros((correction_images2[0].shape[0], correction_images2[0].shape[1]), dtype=np.uint8)
    for image in correction_images2:
        sum_image2 = np.add(sum_image2, image)
    sum_image2= sum_image2/500
                
    if EXAMPLE_FRAMES_SHOWN == 'yes':
        plt.figure()
        plt.imshow(sum_image2, cmap=AFM)
        plt.title(f'ccr averaged reference image, sigma = {s}')
         
    #repeat alignment using sum image as reference   
    for i in range(len(drift_images_n)):         
        c = fn.normxcorr2(drift_images_n[i], sum_image2) #function from dylan. generates cross correlation image. 
        indices = np.where(c == c.max())
         
        
        corr_shift_1 = [indices[0][0]-(c.shape[0]/2), indices[1][0]-(c.shape[1]/2)]
         
        #now zoom in to find more precise maximum         
        clip_c = c[indices[0][0]-w:indices[0][0]+w +1,indices[1][0]-w:indices[1][0]+w +1]           
        c_zoom = cv2.resize(clip_c, None, fx = re, fy = re, interpolation = cv2.INTER_CUBIC)      
        indices2 = np.where(c_zoom == c_zoom.max())
                  
        cr_shift_2 = [(indices2[0][0]-c_zoom.shape[0]/2)/re + 1/re, (indices2[1][0]-c_zoom.shape[1]/2)/re + 1/re]    
                  
        ccr_shift = [(indices[0][0]+ccr_shift_2[0]-ref_image_n.shape[0]/2), (indices[1][0]+ccr_shift_2[1]-ref_image_n.shape[1]/2)]
        ccr_shift_matrix = np.array([[1,0,ccr_shift[1]],[0,1,ccr_shift[0]]])    
        ccr_shift_result = [-x for x in ccr_shift]
        measured_drift_ccr.append(list(ccr_shift_result))
         
    #print(drift_list)
    #print(measured_drift_ccr)   
     
    #put x and y values into separate lists, and find the difference between applied x,y and measured x,y....................................
     
    applied_x = drift_array[:, 0]
    applied_y = drift_array[:, 1]
    phaseccr_x, phaseccr_y = zip(*measured_drift)
    ccr_x, ccr_y = zip(*measured_drift_ccr)
     
    phaseccr_diff_x= [applied_x[i] - phaseccr_x[i] for i in range(len(applied_x))]
    phaseccr_diff_y =  [applied_y[i] - phaseccr_y[i] for i in range(len(applied_y))]
    ccr_diff_x=[applied_x[i] - ccr_x[i] for i in range(len(applied_x))]
    ccr_diff_y=[applied_y[i] - ccr_y[i] for i in range(len(applied_y))]
     
    phaseccr_MD = [np.sqrt(phaseccr_diff_x[i]**2 + phaseccr_diff_y[i]**2) for i in range(len(phaseccr_diff_x))]  #measured difference: sqrt(delta x ^2 + delta y ^2)
    ccr_MD = [np.sqrt(ccr_diff_x[i]**2 + ccr_diff_y[i]**2) for i in range(len(ccr_diff_x))]
     
    drift_mse = []
    for i in range(len(correction_images)):
         mean_square_error=fn.mse(correction_images[i], ref_image_n)
         mean_square_error=round(mean_square_error, 4)
         drift_mse.append(mean_square_error)
     
     
    #Save the drift, measured drift, difference and mean square error in an excel file...............................................
     
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    file_name = f'drift_6J6J_s{pix_per_nm}_g{sigma_value_pix[s]}pix_{sigma_value_nm[s]}nm_noise{noise}_{run}.csv' 
    with open((file_name), 'w', newline='') as csvfile:
         fieldnames = ['Applied_X', 'Applied_Y', 'phaseccr_X', 'phaseccr_Y', 'phaseccr_diff_X', 'phaseccr_diff_Y', 'phaseccr_MD', 'MSE', 'ccr_X', 'ccr_Y', 'ccr_diff_X', 'ccr_diff_Y', 'ccr_MD']
         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
     
         writer.writeheader()
         for ax, ay, m1x, m1y, d1x, d1y, MD1, mse1, m2x, m2y, d2x, d2y, MD2 in zip(applied_x, applied_y, phaseccr_x, phaseccr_y, phaseccr_diff_x, phaseccr_diff_y, phaseccr_MD, drift_mse, ccr_x, ccr_y, ccr_diff_x, ccr_diff_y, ccr_MD):
             writer.writerow({'Applied_X': ax, 'Applied_Y': ay, 'phaseccr_X': m1x, 'phaseccr_Y': m1y, 'phaseccr_diff_X': d1x, 'phaseccr_diff_Y': d1y, 'phaseccr_MD' : MD1, 'MSE': mse1, 'ccr_X': m2x, 'ccr_Y': m2y, 'ccr_diff_X': d2x, 'ccr_diff_Y': d2y, 'ccr_MD' : MD2 })
     
    print('\nData saved to', (file_name))
    
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
    example_image_list.append(ref_image_n)

 

file_name2 = f'drift_6J6J_s{pix_per_nm}_noise{noise}_all_filter_strengths_{run}.csv' 
with open((file_name2), 'w', newline='') as csvfile:
     fieldnames = ['Gauss_nm', 'Gauss_pix', 'phaseccr_MD', 'phaseccr_sdev', 'phaseccr_SEM', 'ccr_MD', 'ccr_sdev', 'ccr_SEM']
     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
 
     writer.writeheader()
     for nm, pix, pc, pc_std, pc_sem, c, c_std, c_sem in zip(sigma_value_nm, sigma_value_pix, phaseccr_MD_list, phaseccr_sdev_list, phaseccr_sem_list, ccr_MD_list, ccr_sdev_list, ccr_sem_list):
         writer.writerow({'Gauss_nm': nm, 'Gauss_pix': pix, 'phaseccr_MD': pc, 'phaseccr_sdev': pc_std, 'phaseccr_SEM': pc_sem, 'ccr_MD': c, 'ccr_sdev' : c_std, 'ccr_SEM': c_sem})
 
print('\nData saved to', (file_name2))

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



#fig.delaxes(ax[4,1])

plt.tight_layout()
manager = plt.get_current_fig_manager()
manager.window.showMaximized()

plt.show() #NOTE: SAVE THE FIGURE MANUALLY UNTIL I SORT OUT THE LAYOUT PROBLEM

#plt.savefig(f'Example_frames_{timestamp}.png')

