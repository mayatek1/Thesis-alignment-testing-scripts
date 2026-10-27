# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:55:14 2024

@author: pymte
"""

import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cmath
from time import time
import random
import os
import imageio



plt.close('all')


start1 = time()
pdb_file = '6j6j.pdb'

#Settings........................................................................
r =1            #tip radius (nm)
angle = 6         #cone angle (o)
pix_per_nm = 10  #sampling (pix/nm)
noise = 0.0          #noise (rms Å)

#rotation about different axis:
theta_x = 90.0    
theta_y = 0.0
theta_z = 0.0

z_thresh =0.5  #set the fraction (0-1) of coordinates to exclude (eg membrane embedded fraction)
#can't get it to work when z_thresh = 0.0. says theres a problem with the end_pos bit. fine now

imsize_increase_nm = 10
imsize_increase_pix = int(imsize_increase_nm*pix_per_nm)

drift_nm=2
drift_pix = drift_nm*pix_per_nm

#Calculations....................................................................
# Load the PDB file using mdtraj
protein = md.load(pdb_file)

# Get the coordinates of all atoms in the PDB file
coords = protein.xyz
coords = coords[0,:,:] #given initially as a 1xNx3 matrix. only want Nx3

end1= time()
time_taken1 = end1 - start1
print('time to collect atomic coords:', time_taken1)


# rotations
if theta_x > 0:
    radian_angle_x = np.deg2rad(theta_x)  # Convert degrees to radians
    coords_rx = coords.copy()  # Create a copy of 'coords' to store the rotated values

    # Perform the rotation for Y and Z coordinates
    coords_rx[:, 1] = coords[:, 1] * np.cos(radian_angle_x) - coords[:, 2] * np.sin(radian_angle_x)
    coords_rx[:, 2] = coords[:, 1] * np.sin(radian_angle_x) + coords[:, 2] * np.cos(radian_angle_x)

    # Overwrite the original Y and Z coordinates with the rotated values
    coords[:, 1:3] = coords_rx[:, 1:3]

   
if theta_y > 0:
    radian_angle_y = np.deg2rad(theta_y)
    coords_ry = coords.copy()  # Create a copy of 'coords' to store the rotated values

    # Perform the Y-axis rotation
    coords_ry[:, 0] = coords[:, 0] * np.cos(radian_angle_y) + coords[:, 2] * np.sin(radian_angle_y)
    coords_ry[:, 2] = coords[:, 2] * np.cos(radian_angle_y) - coords[:, 0] * np.sin(radian_angle_y)

    # Overwrite the original X and Z coordinates with the rotated values
    coords[:, 0] = coords_ry[:, 0]
    coords[:, 2] = coords_ry[:, 2]

# Perform rotation around the Z-axis (theta_z)
if theta_z > 0:
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

# Create an image matrix
img = np.zeros((end_pos[1] - end_pos[0] + imsize_increase_pix, end_pos[3] - end_pos[2] + imsize_increase_pix))

# Initialize coords_s
coords_s = np.zeros((coords.shape[0], 3))


# Pixel scaling of coordinates
coords_s[:, 2] = coords[:, 2] + np.random.normal(0, 0.1, coords.shape[0])
coords_s[:, 0] = coords[:, 0] * pix_per_nm + np.random.normal(0, 0.1, coords.shape[0])*pix_per_nm
coords_s[:, 1] = coords[:, 1] * pix_per_nm + np.random.normal(0, 0.1, coords.shape[0])*pix_per_nm
start2= time()
dx = np.arange(-rs - fspace, rs + fspace) #+1) 
dy = np.arange(-rs - fspace, rs + fspace) #+1)
dx, dy = np.meshgrid(dx, dy, indexing='ij')

offs_x = np.round(coords_s[:, 0]) - coords_s[:, 0]
offs_y = np.round(coords_s[:, 1]) - coords_s[:, 1]

#generate initial image......................................................................................................
 
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
           
        pos_x = round(coords_s[i, 0] + dxl[j] - end_pos[0] +imsize_increase_pix/2)
        pos_y = round(coords_s[i, 1] + dyl[j] - end_pos[2] +imsize_increase_pix/2)
         
        if img[pos_x, pos_y] < h[j]:
            img[pos_x, pos_y] = h[j]
            
img_n = img+ np.random.randn(*img.shape)*noise
             
#generate N images shifted by a random amount..............................................................................................
        
num_frames=500
drift_list = []
drift_img_list = []

 
for k in range(num_frames):
    drift_img = np.zeros((end_pos[1] - end_pos[0] + imsize_increase_pix, end_pos[3] - end_pos[2] + imsize_increase_pix))  # Initialize drift_img for each frame
    drift=[round(random.uniform(-drift_pix, drift_pix), 1), round(random.uniform(-drift_pix, drift_pix), 1)]    
    drift_list.append(drift)
    
    coords_s = np.zeros((coords.shape[0], 3))


    # Pixel scaling of coordinates
    coords_s[:, 2] = coords[:, 2] + np.random.normal(0, 0.01, coords.shape[0])
    coords_s[:, 0] = coords[:, 0] * pix_per_nm + np.random.normal(0, 0.1, coords.shape[0])*pix_per_nm
    coords_s[:, 1] = coords[:, 1] * pix_per_nm + np.random.normal(0, 0.1, coords.shape[0])*pix_per_nm
    start2= time()
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
               
            pos_xd = round(coords_s[i, 0] + dxl[j] - end_pos[0] +imsize_increase_pix/2 +drift[0])
            pos_yd = round(coords_s[i, 1] + dyl[j] - end_pos[2] +imsize_increase_pix/2 +drift[1])
             
            if drift_img[pos_xd, pos_yd] < h[j]:
                drift_img[pos_xd, pos_yd] = h[j]
                drift_img_n = drift_img+ np.random.randn(*img.shape)*noise
    
    drift_img_list.append(drift_img_n)
    if k % 1 == 0:
        #print('Number of frames:', k)
        print(f'\rNumber of frames: {k}', end='', flush=True)
                
#print('Applied drift:', drift_list)

#display the ref image and first few frames to check it's working as expected
AFM = np.load('lutAFM.npy')
AFM = ListedColormap(AFM)  

fig, ax = plt.subplots(2,3)
ax[0,0].imshow(img_n, cmap = AFM)
ax[0,0].set_title('Ref Image')

ax[0,1].imshow(drift_img_list[0], cmap = AFM)
ax[0,1].set_title(f'Drift= {drift_list[0]}')

ax[0,2].imshow(drift_img_list[1], cmap = AFM)
ax[0,2].set_title(f'Drift= {drift_list[1]}')

ax[1,0].imshow(drift_img_list[2], cmap = AFM)
ax[1,0].set_title(f'Drift= {drift_list[2]}')

ax[1,1].imshow(drift_img_list[3], cmap = AFM)
ax[1,1].set_title(f'Drift= {drift_list[3]}')

ax[1,2].imshow(drift_img_list[4], cmap = AFM)
ax[1,2].set_title(f'Drift= {drift_list[4]}')



plt.show()

#put ref and drift images onto one list

drift_img_list.insert(0, img_n)

#make GIF

output_directory = "temp_images"
os.makedirs(output_directory, exist_ok=True)

# Save slices along the third dimension as separate image files in the temporary directory
image_paths = []
for i, matrix in enumerate(drift_img_list):
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
output_gif_path = f"DRIFT{drift_nm}nm_6J6J_f{num_frames}_n{noise}_s{pix_per_nm}_z{z_thresh}_fluctuations.gif"

# Read saved images and save as a GIF using imageio
images = [imageio.imread(image_path) for image_path in image_paths]
imageio.mimsave(output_gif_path, images, duration=0.1, subrectangles=False)  

# Clean up: Remove temporary image files
for image_path in image_paths:
    os.remove(image_path)
os.rmdir(output_directory)

print(f"GIF saved at: {output_gif_path}")



#stack 2D image matrices into a 3D array. save the images and drifts 

stacked_images = np.stack(drift_img_list, axis=2)
np.save(f'DRIFT{drift_nm}nm_6J6J_f{num_frames}_n{noise}_s{pix_per_nm}_z{z_thresh}_stack_fluctuations.npy', stacked_images)


drift_array = np.array(drift_list)
np.save(f'DRIFT{drift_nm}nm_6J6J_f{num_frames}_n{noise}_s{pix_per_nm}_z{z_thresh}_drifts_fluctuations.npy', drift_array)
