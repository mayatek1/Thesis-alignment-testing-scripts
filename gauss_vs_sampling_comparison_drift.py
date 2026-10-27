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

save_tables = 'no'
save_scatter = 'no'
errors = 'SDEV' #input 'SDEV' or 'SEM'
boxes = False

label='Previmalign_THESIS'
PIX_OR_NM = 'pix' #EDIT 'pix' OR 'nm'

title_label = 'OmpG' #TRPV3 OmpG Streptavidin
molecule_name = 'OmpG' #TRPV3_180 OmpG Streptavidin
pdb_id = '2IWV' #6uw4 2IWV 6j6j 

folder_path = f'Drift_all_sampling_comparison_{molecule_name}_{pdb_id}'
os.makedirs(folder_path, exist_ok=True)

#s = [1.25, 2.5, 3.75, 5.0, 6.25, 7.5, 8.75, 10]
s=[1.25, 2.5, 3.75, 5.0, 6.25, 7.5, 8.75, 10]


#csv_files = [f'drift_{pdb_id}_s{i}_noise0.1_all_filter_strengths_Powernoise_prev_image_align.csv'for i in s]
csv_files = [f'drift_{pdb_id}_s{i}_noise0.1_all_filter_strengths_prev_image_align.csv'for i in s]
#csv_files = [f'drift_6J6J_s{i}_noise0.1_all_filter_strengths_avref.csv'for i in s]
#drift_6J6J_s3.75_noise0.1_all_filter_strengths_avref
print(csv_files)

sigma = [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00]

phaseccr_MD_list_pix = []
phaseccr_MD_list_nm = []
phaseccr_SEM_list_pix = []
phaseccr_SEM_list_nm = []
ccr_MD_list_pix = []
ccr_MD_list_nm = []
ccr_SEM_list_pix = []
ccr_SEM_list_nm = []
phaseccr_SDEV_list_pix = []
phaseccr_SDEV_list_nm = []
ccr_SDEV_list_pix = []
ccr_SDEV_list_nm = []

for i in range(len(csv_files)):

    file_path = os.path.join(folder_path, csv_files[i])

    data = []

    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        rows = list(csv_reader)
        data = list(map(list, zip(*rows)))
    
    
    #phaseccr MD values 
    phaseccr_MD_pix = data[2] #phaseccr_MD column
    phaseccr_MD_pix = list(map(float, phaseccr_MD_pix[1:])) #string to float, and exclude title
    phaseccr_MD_list_pix.append(phaseccr_MD_pix)
        
    phaseccr_MD_nm = [x/s[i] for x in phaseccr_MD_pix] #convert shift difference to nm for comparison across different pixel scaling
    phaseccr_MD_list_nm.append(phaseccr_MD_nm)
    
    #phaseccr SDEV values
    phaseccr_SDEV_pix = data[3] #phaseccr_SEM column
    phaseccr_SDEV_pix = list(map(float, phaseccr_SDEV_pix[1:])) #string to float, and exclude title
    phaseccr_SDEV_list_pix.append(phaseccr_SDEV_pix)
    
    phaseccr_SDEV_nm = [x/s[i] for x in phaseccr_SDEV_pix] 
    phaseccr_SDEV_list_nm.append(phaseccr_SDEV_nm)
    
    #phaseccr SEM values
    phaseccr_SEM_pix = data[4] #phaseccr_SEM column
    phaseccr_SEM_pix = list(map(float, phaseccr_SEM_pix[1:])) #string to float, and exclude title
    phaseccr_SEM_list_pix.append(phaseccr_SEM_pix)
    
    phaseccr_SEM_nm = [x/s[i] for x in phaseccr_SEM_pix] 
    phaseccr_SEM_list_nm.append(phaseccr_SEM_nm)
    
    #ccr MD values
    ccr_MD_pix = data[5] #ccr_MD column
    ccr_MD_pix = list(map(float, ccr_MD_pix[1:])) #string to float, and exclude title
    ccr_MD_list_pix.append(ccr_MD_pix)
    
    ccr_MD_nm = [x/s[i] for x in ccr_MD_pix] #convert shift difference to nm for comparison across different pixel scaling
    ccr_MD_list_nm.append(ccr_MD_nm)
    
    #ccr SDEV values
    ccr_SDEV_pix = data[3] #ccr_SEM column
    ccr_SDEV_pix = list(map(float, ccr_SDEV_pix[1:])) #string to float, and exclude title
    ccr_SDEV_list_pix.append(ccr_SDEV_pix)
    
    ccr_SDEV_nm = [x/s[i] for x in ccr_SDEV_pix] 
    ccr_SDEV_list_nm.append(ccr_SDEV_nm)
    
    #ccr SEM values
    ccr_SEM_pix = data[4] #ccr_SEM column
    ccr_SEM_pix = list(map(float, ccr_SEM_pix[1:])) #string to float, and exclude title
    ccr_SEM_list_pix.append(ccr_SEM_pix)
    
    ccr_SEM_nm = [x/s[i] for x in ccr_SEM_pix] 
    ccr_SEM_list_nm.append(ccr_SEM_nm)


#phaseccr plot


plt.figure(figsize=(10, 6))

if PIX_OR_NM == 'pix':
    phaseccr_array = np.array(phaseccr_MD_list_pix)
    
else:
    phaseccr_array = np.array(phaseccr_MD_list_nm)

plt.imshow(phaseccr_array, cmap=cmap_34, vmin=0.1, vmax=0.7) #CHOOSE THE PIX OR NM ARRAY
cbar = plt.colorbar(shrink = 0.6 ) 
cbar.set_label(f'Measured Shift difference ({PIX_OR_NM})', fontsize=10) 
cbar.ax.tick_params(labelsize=5)


plt.xticks(np.arange(0, len(sigma), 1),labels= sigma, fontsize=8)
plt.yticks(np.arange(0, len(s), 1),labels = s, fontsize=8)

plt.xlabel('Gauss Filter Strength (pix)', fontsize = 10)
plt.ylabel('Sampling (pixels/nm)', fontsize = 10)

for i in range(len(sigma)):
    for j in range(len(s)):
        #text = plt.text(i, j, f'{phaseccr_array[j,i]:.2f}', fontsize = 6, ha='center', va = 'center', color = 'r') 
        min_col = np.argmin(phaseccr_array[j, :])
        
        for i in range(len(sigma)):
            if boxes:
                bbox = None
                if i == min_col:
                    bbox = dict(
                        facecolor='none',
                        edgecolor='red',
                        linewidth=2,
                        boxstyle='square,pad=0.3')
            else:
                bbox = None    
                 
            plt.text(i, j,
            f'{phaseccr_array[j,i]:.2f}',
            fontsize=7,
            ha='center',
            va='center',
            color='r',
            bbox=bbox)

plt.title(f'Phaseccr Method, {title_label}', fontsize = 12)
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()

if save_tables == 'yes':
    plt.savefig(os.path.join(folder_path,f'phaseccr_comparison_graph_{PIX_OR_NM}_{pdb_id}_{label}.png'), dpi=1000) 
plt.show()

#ccr plot

plt.figure(figsize=(10, 6))

if PIX_OR_NM == 'pix':
    ccr_array = np.array(ccr_MD_list_pix)
    
else:
    ccr_array = np.array(ccr_MD_list_nm)

plt.imshow(ccr_array, cmap=cmap_34, vmin=0.1, vmax=0.7) 
cbar = plt.colorbar(shrink = 0.6 ) 
cbar.set_label(f'Measured Shift difference({PIX_OR_NM})', fontsize=10) #EDIT TO PIX OR NM
cbar.ax.tick_params(labelsize=5)


plt.xticks(np.arange(0, len(sigma), 1),labels= sigma, fontsize = 8)
plt.yticks(np.arange(0, len(s), 1),labels = s, fontsize = 8)

plt.xlabel('Gauss Filter Strength (pix)', fontsize = 10)
plt.ylabel('Sampling (pixels/nm)', fontsize = 10)

for i in range(len(sigma)):
    for j in range(len(s)):
        #text = plt.text(i, j, f'{ccr_array[j,i]:.2f}', fontsize = 6, ha='center', va = 'center', color = 'r') 
        min_col = np.argmin(ccr_array[j, :])
        
        for i in range(len(sigma)):
            if boxes:
                bbox = None
                if i == min_col:
                    bbox = dict(
                        facecolor='none',
                        edgecolor='red',
                        linewidth=2,
                        boxstyle='square,pad=0.3')
            else:
                bbox = None
                
            plt.text(i, j,
            f'{ccr_array[j,i]:.2f}',
            fontsize=7,
            ha='center',
            va='center',
            color='r',
            bbox=bbox)
            
plt.title(f'Cross correlation Method, {title_label}', fontsize = 12)
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()

if save_tables == 'yes':
    plt.savefig(os.path.join(folder_path, f'ccr_comparison_graph_{PIX_OR_NM}_{pdb_id}_{label}.png'), dpi=1000) #EDIT PIX OR NM
plt.show()

#phaseccr line graph
fig, ax = plt.subplots()

for i in range(len(s)):
    if PIX_OR_NM == 'pix':
        ax.scatter(sigma, phaseccr_MD_list_pix[i], label=f'{s[i]} pix/nm')    
        if errors == 'SDEV':
            ax.errorbar(sigma, phaseccr_MD_list_pix[i], yerr=phaseccr_SDEV_list_pix[i], fmt="o")
        if errors == 'SEM':
            ax.errorbar(sigma, phaseccr_MD_list_pix[i], yerr=phaseccr_SEM_list_pix[i], fmt="o")
    else:
        ax.scatter(sigma, phaseccr_MD_list_nm[i], label=f'{s[i]} pix/nm')    
        if errors == 'SDEV':
            ax.errorbar(sigma, phaseccr_MD_list_nm[i], yerr=phaseccr_SDEV_list_nm[i], fmt="o")
        if errors == 'SEM':
            ax.errorbar(sigma, phaseccr_MD_list_nm[i], yerr=phaseccr_SEM_list_nm[i], fmt="o")

ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
ax.set_xlabel('Sigma')
ax.set_ylabel(f'Measured difference ({PIX_OR_NM})')
ax.set_title(f'Phaseccr {title_label}. Error bars = {errors}')
if save_scatter == 'yes':
    plt.savefig(os.path.join(folder_path, f'phaseccr_scatter_graph_{PIX_OR_NM}_{pdb_id}_{label}.png'), dpi=1000)
plt.show()

#ccr line graph
fig, ax = plt.subplots()

for i in range(len(s)):
    if PIX_OR_NM == 'pix':
        ax.scatter(sigma, ccr_MD_list_pix[i], label=f'{s[i]} pix/nm')    
        if errors == 'SDEV':
            ax.errorbar(sigma, ccr_MD_list_pix[i], yerr=ccr_SDEV_list_pix[i], fmt="o")
        if errors == 'SEM':
            ax.errorbar(sigma, ccr_MD_list_pix[i], yerr=ccr_SEM_list_pix[i], fmt="o")
    else:
        ax.scatter(sigma, ccr_MD_list_nm[i], label=f'{s[i]} pix/nm')    
        if errors == 'SDEV':
            ax.errorbar(sigma, ccr_MD_list_nm[i], yerr=ccr_SDEV_list_nm[i], fmt="o")
        if errors == 'SEM':
            ax.errorbar(sigma, ccr_MD_list_nm[i], yerr=ccr_SEM_list_nm[i], fmt="o")
    
ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
ax.set_xlabel('Sigma')
ax.set_ylabel(f'Measured difference ({PIX_OR_NM})')
ax.set_title(f'Ccr {title_label}. Error bars = {errors}')
if save_scatter == 'yes':
    plt.savefig(os.path.join(folder_path, f'ccr_scatter_graph_{PIX_OR_NM}_{pdb_id}_{label}_{errors}.png'), dpi=1000)
plt.show()

mean_phaseccr = np.mean(phaseccr_array)
sem_phaseccr = np.std(phaseccr_array, ddof=1) / np.sqrt(phaseccr_array.size)

mean_ccr = np.mean(ccr_array)
sem_ccr = np.std(ccr_array, ddof=1) / np.sqrt(ccr_array.size)

print(f'mean phaseccr {mean_phaseccr} +/- {sem_phaseccr}')
print(f'mean ccr {mean_ccr} +/- {sem_ccr}')
