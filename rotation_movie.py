# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 16:55:50 2023

@author: pymte
"""

import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cmath
import random 
import os
import imageio
from time import time

plt.close('all')


pdb_id = '6uw4'
pdb_file = f'{pdb_id}.pdb'
molecule_name = 'TRPV3'
angle_range = 20
label = f'{molecule_name}_180_anglerange{angle_range}'

#Settings........................................................................
r =1               #tip radius (nm)
angle = 6          #cone angle (o)
pix_per_nm =10 #sampling (pix/nm)
noise = 0        #noise (rms Å)

#rotation about different axis:
num_frames = 500
img_list= []
angle_list = []
for k in range(num_frames):
    theta_x = 180.0   
    theta_y = 0.0
    theta_z = ()
    if k == 0:
        theta_z = 0
    else:
        theta_z = round(random.uniform(-angle_range/2, angle_range/2),1)
    
    z_thresh =0.8  #set the fraction (0-1) of coordinates to exclude (eg membrane embedded fraction)
    #can't get it to work when z_thresh = 0.0. says theres a problem with the end_pos bit. fine now
    
    
    #Calculations....................................................................
    # Load the PDB file using mdtraj
    protein = md.load(pdb_file)
    
    # Get the coordinates of all atoms in the PDB file
    coords = protein.xyz
    coords = coords[0,:,:] #given initially as a 1xNx3 matrix. only want Nx3
    
    
    
    
    # rotations
    if theta_x != 0:
        radian_angle_x = np.deg2rad(theta_x)  # Convert degrees to radians
        coords_rx = coords.copy()  # Create a copy of 'coords' to store the rotated values
    
        # Perform the rotation for Y and Z coordinates
        coords_rx[:, 1] = coords[:, 1] * np.cos(radian_angle_x) - coords[:, 2] * np.sin(radian_angle_x)
        coords_rx[:, 2] = coords[:, 1] * np.sin(radian_angle_x) + coords[:, 2] * np.cos(radian_angle_x)
    
        # Overwrite the original Y and Z coordinates with the rotated values
        coords[:, 1:3] = coords_rx[:, 1:3]
    
       
    if theta_y != 0:
        radian_angle_y = np.deg2rad(theta_y)
        coords_ry = coords.copy()  # Create a copy of 'coords' to store the rotated values
    
        # Perform the Y-axis rotation
        coords_ry[:, 0] = coords[:, 0] * np.cos(radian_angle_y) + coords[:, 2] * np.sin(radian_angle_y)
        coords_ry[:, 2] = coords[:, 2] * np.cos(radian_angle_y) - coords[:, 0] * np.sin(radian_angle_y)
    
        # Overwrite the original X and Z coordinates with the rotated values
        coords[:, 0] = coords_ry[:, 0]
        coords[:, 2] = coords_ry[:, 2]
    
    # Perform rotation around the Z-axis (theta_z)
    if theta_z != 0:
        radian_angle_z = np.deg2rad(theta_z)
      
        v = np.vstack((coords[:, 0], coords[:, 1]))
        R = np.array([[np.cos(radian_angle_z), np.sin(radian_angle_z)],
                  [-np.sin(radian_angle_z), np.cos(radian_angle_z)]])
        so = np.dot(R, v)
        
    
        # Update the X and Y coordinates
        coords[:, 0] = so[0, :]
        coords[:, 1] = so[1, :]  
    
    if z_thresh > 0:
        z_thresh1 = z_thresh * (np.max(coords[:, 2]) - np.min(coords[:, 2])) + np.min(coords[:, 2])
        mask = coords[:, 2] > z_thresh1
    
    # Use numpy.where to get the indices where the condition is met
        indices = np.where(mask)
    
    # Extract the corresponding elements from coords
        coords = coords[indices]
    
    # Shift the z-coordinates
    coords[:, 2] = coords[:, 2] - np.min(coords[:, 2])
    
    # Calculate pixel scaling for tip radius
    rs = r * pix_per_nm
    
    # Set image size
    fspace = (np.max(coords[:, 2]) - r) * np.tan(angle * np.pi / 180) * pix_per_nm + 1
    
    
    end_pos = [int(np.floor(min(coords[:, 0] * pix_per_nm) - rs - fspace)),
               int(np.ceil(max(coords[:, 0] * pix_per_nm) + rs + fspace)),
               int(np.floor(min(coords[:, 1] * pix_per_nm) - rs - fspace)),
               int(np.ceil(max(coords[:, 1] * pix_per_nm) + rs + fspace))]
    
    centre_x = pix_per_nm*(min(coords[:,0])+(max(coords[:,0])-min(coords[:,0]))/2)
    centre_y = pix_per_nm*(min(coords[:,1])+(max(coords[:,1])-min(coords[:,1]))/2)
    
    # Create an image matrix
    #img = np.zeros((end_pos[1] - end_pos[0] + 20, end_pos[3] - end_pos[2] + 20))
    imsize_nm = 20
    imsize_pix = round(imsize_nm*pix_per_nm)
    img = np.zeros([imsize_pix,imsize_pix])
    
    # Initialize coords_s
    coords_s = np.zeros((coords.shape[0], 3))
    
    
    # Pixel scaling of coordinates
    coords_s[:, 2] = coords[:, 2]
    coords_s[:, 0] = coords[:, 0] * pix_per_nm
    coords_s[:, 1] = coords[:, 1] * pix_per_nm
    
    
    dx = np.arange(-rs - fspace, rs + fspace) #+1) 
    dy = np.arange(-rs - fspace, rs + fspace) #+1)
    dx, dy = np.meshgrid(dx, dy, indexing='ij')
    
    offs_x = np.round(coords_s[:, 0]) - coords_s[:, 0]
    offs_y = np.round(coords_s[:, 1]) - coords_s[:, 1]
     
    for i in range(len(coords)):
        dxl = dx.ravel('F') - offs_x[i]
        dyl = dy.ravel('F') - offs_y[i]
        
        dh = np.zeros(len(dxl))
        h = np.zeros(len(dxl))
    
        for j in range(len(dxl)):
            
            dh[j] = np.real(cmath.sqrt(-((dxl[j] / pix_per_nm) ** 2) - ((dyl[j] / pix_per_nm) ** 2) + r ** 2)) - r #tip interaction
            
            if dh[j] > -r:
                h[j] = coords_s[i, 2] + dh[j]
            else:
                di = np.real(cmath.sqrt(((dxl[j] / pix_per_nm) ** 2) + ((dyl[j] / pix_per_nm) ** 2))) - r
                h[j] = coords_s[i, 2] - r - (di / np.tan(angle * np.pi / 180))
               
            #pos_x = round(coords_s[i, 0] + dxl[j] - end_pos[0] + 10)
            #pos_y = round(coords_s[i, 1] + dyl[j] - end_pos[2] + 10)
            pos_x = round(coords_s[i, 0] + dxl[j] - centre_x + imsize_pix/2) # 25 = image size / 2
            pos_y = round(coords_s[i, 1] + dyl[j] - centre_y + imsize_pix/2)
            
            if img[pos_x, pos_y] < h[j]:
                img[pos_x, pos_y] = h[j]
    
    
    
    img_n = img+ np.random.randn(*img.shape)*noise
    img_list.append(img_n)
    angle_list.append(theta_z)
    
    if k == 0:
        frame1time = time()
    if k == 1:
        frame2time = time()
        total_time_estimate = (((frame2time - frame1time)*num_frames)/60)
        print('Total time estimate = ',round(total_time_estimate, 0),' mins (',round(total_time_estimate/60, 2), 'hours)')
    if k >= 1:    
        if k % 1 == 0:
            #print('Number of frames:', k)
            print(f'\rNumber of frames: {k}', end='', flush=True)

#Create gif 
AFM = np.load('lutAFM.npy')
AFM = ListedColormap(AFM)  
    
output_directory = "temp_images"
os.makedirs(output_directory, exist_ok=True)

# Save slices along the third dimension as separate image files in the temporary directory
image_paths = []
for i, matrix in enumerate(img_list):
    # Apply custom colormap
    matrix_normalised = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
    image_colormap = AFM(matrix_normalised)
    
    # Convert to uint8 for saving as image
    image_colormap_uint8 = (image_colormap[:, :, :3]*255).astype(np.uint8)
    
    image_path = os.path.join(output_directory, f"frame_{i}.png")
    
    # Save the image
    imageio.imwrite(image_path, image_colormap_uint8)
    image_paths.append(image_path)

# Set the output GIF file path
output_gif_path = f"Rotation_{pdb_id}_f{num_frames}_n{noise}_s{pix_per_nm}_z{z_thresh}_gif_{label}.gif"

# Read saved images and save as a GIF using imageio
images = [imageio.imread(image_path) for image_path in image_paths]
imageio.mimsave(output_gif_path, images, duration=0.1, subrectangles=False)  

# Clean up: Remove temporary image files
for image_path in image_paths:
    os.remove(image_path)
os.rmdir(output_directory)

print(f"GIF saved at: {output_gif_path}")



#stack 2D image matrices into a 3D array. save the images and drifts 

stacked_images = np.stack(img_list, axis=2)
np.save(f'Rotation_{pdb_id}_f{num_frames}_n{noise}_s{pix_per_nm}_z{z_thresh}_imagestack_{label}.npy', stacked_images)


angle_array = np.array(angle_list)
np.save(f'Rotation_{pdb_id}_f{num_frames}_n{noise}_s{pix_per_nm}_z{z_thresh}_anglevalues_{label}.npy', angle_array)
