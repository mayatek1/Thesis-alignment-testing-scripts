# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 10:16:28 2024

@author: pymte
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

label='dr_10itr_full'
PIX_OR_NM = 'pix' #EDIT PIX OR NM

csv_files = ['drift_rotate_6J6J_s1.25_noise0.1_all_filter_strengths_test1.csv', 'drift_rotate_6J6J_s2.5_noise0.1_all_filter_strengths_test1.csv',
             'drift_rotate_6J6J_s3.75_noise0.1_all_filter_strengths_test1.csv','drift_rotate_6J6J_s5.0_noise0.1_all_filter_strengths_test1.csv',
             'drift_rotate_6J6J_s6.25_noise0.1_all_filter_strengths_test1.csv', 'drift_rotate_6J6J_s7.5_noise0.1_all_filter_strengths_test1.csv',
             'drift_rotate_6J6J_s8.75_noise0.1_all_filter_strengths_test1.csv','drift_rotate_6J6J_s10_noise0.1_all_filter_strengths_test1.csv']

s = [1.25, 2.5, 3.75, 5.0, 6.25, 7.5, 8.75, 10]
sigma = [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00]

drift_diff_list_pix = []
drift_diff_list_nm = []
angle_diff_list = []

for i in range(len(csv_files)):

    csv_file = csv_files[i]

    data = []

    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        rows = list(csv_reader)
        data = list(map(list, zip(*rows)))
    
    
    #DRIFT DIFF values 
    drift_diff_pix = data[2] #phaseccr_MD column
    drift_diff_pix = list(map(float, drift_diff_pix[1:])) #string to float, and exclude title
    drift_diff_list_pix.append(drift_diff_pix)
    
    drift_diff_nm = [x/s[i] for x in drift_diff_pix] #convert shift difference to nm for comparison across different pixel scaling
    drift_diff_list_nm.append(drift_diff_nm)
    
    #ANGLE DIFF values
    angle_diff = data[5] #ccr_MD column
    angle_diff = list(map(float, angle_diff[1:])) #string to float, and exclude title
    angle_diff_list.append(angle_diff)
    
   


#drift plot


plt.figure()

if PIX_OR_NM == 'pix':
    drift_array = np.array(drift_diff_list_pix)
    
else:
    drift_array = np.array(drift_diff_list_nm)

plt.imshow(drift_array, cmap='PuBu') #CHOOSE THE PIX OR NM ARRAY
cbar = plt.colorbar(shrink = 0.6 ) 
cbar.set_label(f'Measured Shift difference ({PIX_OR_NM})', fontsize=16) 
cbar.ax.tick_params(labelsize=12)


plt.xticks(np.arange(0, len(sigma), 1),labels= sigma, fontsize=12)
plt.yticks(np.arange(0, len(s), 1),labels = s, fontsize=12)

plt.xlabel('Gauss Filter Strength (pix)', fontsize = 16)
plt.ylabel('Sampling (pixels/nm)', fontsize = 16)

for i in range(len(sigma)):
    for j in range(len(s)):
        text = plt.text(i, j, f'{drift_array[j,i]:.2f}', ha='center', va = 'center', color = 'r') 

plt.title('Translation + Rotation Drift Difference', fontsize = 16)
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()

plt.savefig(f'drift_comparison_graph_{PIX_OR_NM}_{label}.png') 
plt.show()

#ccr plot

plt.figure()

angle_array = np.array(angle_diff_list)


plt.imshow(angle_array, cmap='PuBu') 
cbar = plt.colorbar(shrink = 0.6 ) 
cbar.set_label('Measured Angle difference (degrees)', fontsize=16) #EDIT TO PIX OR NM
cbar.ax.tick_params(labelsize=12)


plt.xticks(np.arange(0, len(sigma), 1),labels= sigma, fontsize = 12)
plt.yticks(np.arange(0, len(s), 1),labels = s, fontsize = 12)

plt.xlabel('Gauss Filter Strength (pix)', fontsize = 16)
plt.ylabel('Sampling (pixels/nm)', fontsize = 16)

for i in range(len(sigma)):
    for j in range(len(s)):
        text = plt.text(i, j, f'{angle_array[j,i]:.2f}', fontsize = 10, ha='center', va = 'center', color = 'r') #EDIT PIX OR NM


plt.title('Translation + Rotation Angle Difference', fontsize = 16)
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()

plt.savefig(f'angle_comparison_graph_{label}.png') 
plt.show()