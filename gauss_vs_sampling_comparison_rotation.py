# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 10:16:28 2024

@author: pymte
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
cmap = plt.get_cmap('magma_r')
cmap_34 = colors.LinearSegmentedColormap.from_list('magma_34', cmap(np.linspace(0, 0.8, 256)))


#label = 'PowerNoise_polar_rpt'

molecule_name = 'OmpG' #OmpG Streptavidin TRPV3_180
pdb_id = '2IWV' #2IWV 6J6J 6uw4

folder_path = f'Rotation_all_sampling_comparison_{molecule_name}_{pdb_id}'
os.makedirs(folder_path, exist_ok=True)

s = [1.25, 2.5, 3.75, 5.0, 6.25, 7.5, 8.75, 10]
#csv_files = [f'Rotation_{pdb_id}_s{i}_noise0.1_all_filter_strengths_{molecule_name}Powernoise_polar_rpt.csv'for i in s]
#csv_files = [f'Rotation_{pdb_id}_s{i}_noise0.1_all_filter_strengths_{molecule_name}Powernoise_anglerange100_searchangles180.csv'for i in s]
#csv_files = [f'Rotation_6J6J_s{i}_noise0.1_all_filter_strengths_Streptavidinpowernoise_test_nnr.csv'for i in s]
#csv_files = [f'Rotation_{pdb_id}_s{i}_noise0.1_all_filter_strengths_{molecule_name}Powernoise_ccr100_polar100.csv'for i in s] #_anglerange20
csv_files = [f'Rotation_{pdb_id}_s{i}_noise0.1_all_filter_strengths_{molecule_name}Powernoise.csv'for i in s]
print(csv_files)


sigma = [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00]

polar_diff_list = []
ccr_diff_list = []

for csv_file in csv_files:

    file_path = os.path.join(folder_path, csv_file)

    data = []

    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        rows = list(csv_reader)
        data = list(map(list, zip(*rows)))
    
    
    #phaseccr MD values 
    polar_diff = data[2] #polar diff column
    polar_diff = list(map(float, polar_diff[1:])) #string to float, and exclude title
    polar_diff_list.append(polar_diff)
    
    #ccr MD values
    ccr_diff= data[5] #ccr_MD column
    ccr_diff= list(map(float, ccr_diff[1:])) #string to float, and exclude title 
    ccr_diff_list.append(ccr_diff)



#polar plot

plt.figure(figsize=(15, 10))

polar_array = np.array(polar_diff_list)
    
plt.imshow(polar_array, cmap=cmap_34, vmin=0.2, vmax=4) 
cbar = plt.colorbar(shrink = 0.6 ) 
cbar.set_label('Measured Angle difference (degrees)', fontsize=14) 
cbar.ax.tick_params(labelsize=12)

plt.xticks(np.arange(0, len(sigma), 1),labels= sigma)
plt.yticks(np.arange(0, len(csv_files), 1),labels = s)

plt.xlabel('Gauss Filter Strength (pix)', fontsize = 14)
plt.ylabel('Sampling (pixels/nm)', fontsize = 14)

for i in range(len(sigma)):
    for j in range(len(s)):        
        min_col = np.nanargmin(polar_array[j, :])
        
        for i in range(len(sigma)):
            bbox = None
            if i == min_col:
                bbox = dict(
                    facecolor='none',
                    edgecolor='red',
                    linewidth=2,
                    boxstyle='square,pad=0.3')
                
            plt.text(i, j,
            f'{polar_array[j,i]:.2f}',
            ha='center',
            va='center',
            color='r',
            bbox=bbox)

plt.title('Polar Method', fontsize = 16)
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()

#plt.savefig(os.path.join(folder_path, f'ROTATION_polar_comparison_graph_{label}.png'))
plt.show()

#rotccr plot

plt.figure(figsize=(15, 10))

ccr_array = np.array(ccr_diff_list)
    
plt.imshow(ccr_array, cmap=cmap_34, vmin=0.2, vmax=4) 
cbar = plt.colorbar(shrink = 0.6 ) 
cbar.set_label('Measured Angle difference (degrees)', fontsize=14) 
cbar.ax.tick_params(labelsize=12)


plt.xticks(np.arange(0, len(sigma), 1),labels= sigma)
plt.yticks(np.arange(0, len(csv_files), 1),labels = s)

plt.xlabel('Gauss Filter Strength (pix)', fontsize = 14)
plt.ylabel('Sampling (pixels/nm)', fontsize = 14)

for i in range(len(sigma)):
    for j in range(len(s)):
        min_col = np.nanargmin(ccr_array[j, :])
        
        for i in range(len(sigma)):
            bbox = None
            if i == min_col:
                bbox = dict(
                    facecolor='none',
                    edgecolor='red',
                    linewidth=2,
                    boxstyle='square,pad=0.3')
                
            plt.text(i, j,
            f'{ccr_array[j,i]:.2f}',
            ha='center',
            va='center',
            color='r',
            bbox=bbox)



plt.title('Cross correlation Method', fontsize = 16)
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()

#plt.savefig(os.path.join(folder_path, f'ROTATION_ccr_comparison_graph_{label}.png'))
plt.show()

mean_polar= np.mean(polar_array)
sem_polar = np.std(polar_array, ddof=1) / np.sqrt(polar_array.size)

mean_ccr = np.nanmean(ccr_array)
sem_ccr = np.nanstd(ccr_array, ddof=1) / np.sqrt(ccr_array.size)

print(f'mean polar {mean_polar} +/- {sem_polar}')
print(f'mean ccr {mean_ccr} +/- {sem_ccr}')
