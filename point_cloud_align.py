# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:20:40 2024

@author: pymte
"""

import numpy as np
import functions as fn

import pyvista as pv
from pycpd import RigidRegistration

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation   
from scipy.ndimage import gaussian_filter


pix_per_nm = 6.25
image_file = f'Rotation_6J6J_f500_n0_s{pix_per_nm}_z0.5_Imagestack.npy'
angle_file = f'Rotation_6J6J_f500_n0_s{pix_per_nm}_z0.5_Angles.npy'

#set params
para_angle = 60
para_grad = 1/(np.tan(np.deg2rad(para_angle))*pix_per_nm)
z_thresh = 0.85
noise = 0.1
sigma_value_pix = 2.5

image_stack = np.load(image_file)
angles_array=np.load(angle_file)

num_frames = image_stack.shape[2]
images = [image_stack[:, :, i] for i in range(num_frames)]
images_para = [fn.add_para2(image, para_grad)  for image in images]
images_para_n = [image + np.random.normal(0, noise, images_para[0].shape) for image in images_para]
images_para_ng = [gaussian_filter(image, sigma_value_pix) for image in images_para_n]

#the first frame is the reference image so separate this
  
ref_image = images_para_n[0] 
img_no = 30 #choose an image to test
rot_image = images_para_n[img_no] 

ref = fn.convert_to_nx3_array(ref_image) 
im = fn.convert_to_nx3_array(rot_image) 

#threshold above mica surface
if z_thresh > 0:
    z_thresh_ref = z_thresh * (np.max(ref[:, 2]) - np.min(ref[:, 2])) + np.min(ref[:, 2])
    mask = ref[:, 2] > z_thresh_ref
    indices = np.where(mask)
    ref = ref[indices]
ref[:, 2] = ref[:, 2] - np.min(ref[:, 2])

if z_thresh > 0:
    z_thresh_im = z_thresh * (np.max(im[:, 2]) - np.min(im[:, 2])) + np.min(im[:, 2])
    mask = im[:, 2] > z_thresh_im
    indices = np.where(mask)
    im = im[indices]
im[:, 2] = im[:, 2] - np.min(im[:, 2])


fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ref[:, 0], ref[:, 1], ref[:, 2], color='green', label='Ref')
ax.scatter(im[:, 0], im[:, 1], im[:, 2], color='blue', label='Rotated')
plt.legend()
plt.show()

reg = RigidRegistration(X=ref, Y=im)
TY, (s_reg, R_reg, t_reg) = reg.register()

print('Applied angle:', angles_array[img_no])
print('Rotation matrix of the registration:') 
print(R_reg) 
#need to convert from rotation matrix to angles so can compare with applied angle

### first transform the matrix to euler angles
r =  Rotation.from_matrix(R_reg)
angles = r.as_euler("zyx",degrees=True)

print('zyx angles from registration:')
print(angles)



'''
#point cloud
pcA = pv.PolyData(A)
pcA.plot(eye_dome_lighting=True)
pcB = pv.PolyData(B)
pcB.plot(eye_dome_lighting=True) 

#A = np.packbits(A.astype(np.uint8))
#B = np.packbits(B.astype(np.uint8))

#A = A.view(dtype=[('coords', '<f4', (3,))]).reshape((-1))
#B = B.view(dtype=[('coords', '<f4', (3,))]).reshape((-1))



print(A.shape)
print(A.dtype.descr)

colors = {'A': 'green', 'B': 'blue'}

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')

x_A = A[:, 0]
y_A = A[:, 1]
z_A = A[:, 2]

# Assuming A and B are different point clouds with different colors
ax.scatter(x_A, y_A, z_A, color=colors['A'], label='A')

# Assuming B is another point cloud
x_B = B[:, 0]
y_B = B[:, 1]
z_B = B[:, 2]

ax.scatter(x_B, y_B, z_B, color=colors['B'], label='B')

plt.legend()
plt.show()
'''