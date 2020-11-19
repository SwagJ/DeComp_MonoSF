import open3d as o3d
from matplotlib import pyplot as plt
import numpy as np
import cv2
import math
from PIL import Image


def pixel_coord_np(width, height):
    x = np.linspace(0, width - 1, width).astype(np.int)
    y = np.linspace(0, height - 1, height).astype(np.int)
    [x, y] = np.meshgrid(x, y)
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))



depth_raw = cv2.imread('depth.png', cv2.IMREAD_ANYDEPTH)
color_raw = cv2.imread('color.png')
depth = depth_raw / 1000

print(depth_raw.shape,color_raw.shape)
print(np.min(depth_raw),np.max(depth_raw))


#image to coord list + RGB

height = depth_raw.shape[0]
width = depth_raw.shape[1]

image_coord = pixel_coord_np(width,height)

#image_coord_homo = np.concatenate((image_coord,np.ones((image_coord.shape[0],1))),axis=1)
#depth_raw = np.asarray(depth_raw)
#print(depth_raw.shape)
print(np.max(depth_raw),np.min(depth_raw))
intrinsic_raw = np.loadtxt("intrinsics.txt")
pose = np.loadtxt("pose.txt")

RT = pose[:3,:3]
C = pose[:3,3]
R = RT.T
t = -np.matmul(R,C)
extrinsic = np.eye(4)
extrinsic[:3,:3] = R
extrinsic[:3,3] = t

trans = np.matmul(intrinsic_raw,extrinsic[:3,:])
inv_intrinsic = np.linalg.inv(intrinsic_raw)

cam_coords = (inv_intrinsic[:3, :3] @ image_coord * depth.flatten()).T
print(cam_coords.shape)
print(np.min(cam_coords[:,0]),np.max(cam_coords[:,0]))
print(np.min(cam_coords[:,1]),np.max(cam_coords[:,1]))
print(np.min(cam_coords[:,2]),np.max(cam_coords[:,2]))

#back proj check
tmp_coords = (intrinsic_raw @ cam_coords.T / (depth + 1e-6).flatten()).T

print(tmp_coords.shape)
print(np.min(tmp_coords[:,0]),np.max(tmp_coords[:,0]))
print(np.min(tmp_coords[:,1]),np.max(tmp_coords[:,1]))
print(np.min(tmp_coords[:,2]),np.max(tmp_coords[:,2]))


#intrinsic = o3d.camera.PinholeCameraIntrinsic(width=depth_raw.shape[1],
							#		height=depth_raw.shape[0],
							#		fx=intrinsic_raw[0,0],fy=intrinsic_raw[1,1],
							#		cx=intrinsic_raw[0,2],cy=intrinsic_raw[1,2])

#depth_raw = depth_raw / 1000
#depth_raw = o3d.geometry.Image(depth_raw.astype(np.float32))

#rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
#    color_raw, depth_raw,depth_scale = 1, depth_trunc = 20,convert_rgb_to_intensity=False)
#plt.subplot(1, 2, 1)
#plt.imshow(rgbd_image.color)
#plt.subplot(1, 2, 2)
#plt.imshow(rgbd_image.depth)
#plt.show()

#pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic, extrinsic,
						#							project_valid_depth_only = True)
#o3d.visualization.draw_geometries([pc])

#vis = o3d.visualization.Visualizer()
#vis.create_window()
#vis.add_geometry(pc)
#ctr = vis.get_view_control()

#ctr.set_up(-R[1,:])
#ctr.set_front(-R[2,:])
#ctr.set_lookat(R[0,:])
#vis.run()
#vis.destroy_window()

# image restore
# print(np.asarray(pc.points).shape)
# points = np.asarray(pc.points)
# colors = np.asarray(pc.colors)
# print(np.min(points[:,0]),np.min(points[:,1]),np.min(points[:,2]))
# print(np.max(points[:,0]),np.max(points[:,1]),np.max(points[:,2]))
# points = np.concatenate((points,np.ones((points.shape[0],1))),axis=1)
# print(points.shape)
# # recover image
# trans_mat = np.matmul(intrinsic_raw,extrinsic[:3,:])
# print(trans_mat)


# test trans_mat
# camera_coord = np.matmul(trans_mat,points.T).T
# print(camera_coord[0:10,:])
# print(np.max(camera_coord[:,2]),np.min(camera_coord[:,2]))
# image_coord = camera_coord / camera_coord[:,2][:,None]
# print(np.min(image_coord[:,0]),np.max(image_coord[:,0]))
# print(np.min(image_coord[:,1]),np.max(image_coord[:,1]))

#rotation
theta_x = -math.pi/12
theta_y = -math.pi / 12
theta_z = -math.pi / 12
Ry = np.array([[1,0,0],
				[0,math.cos(theta_x),-math.sin(theta_x)],
				[0,math.sin(theta_x),math.cos(theta_x)]])
Rx = np.array([[math.cos(theta_y),0,math.sin(theta_y)],
				[0,1,0],
				[-math.sin(theta_y),0,math.cos(theta_y)]]
				)
Rz = np.array([[math.cos(theta_z),-math.sin(theta_z),0],
				[math.sin(theta_z),math.cos(theta_z),0],
				[0,0,1]])

total_R = np.matmul(np.matmul(Rz,Ry),Rx)

#translation
T = [0,0,-0.5]
# rotation on cam_coords
print(cam_coords.shape)
new_cam_coords = np.matmul(cam_coords,total_R)

# translation on cam_coords
# valid_area = new_cam_coords[:,2] >= -T[-1]
# new_cam_coords[valid_area,-1] = new_cam_coords[valid_area,-1] + T[-1]

print(new_cam_coords.shape)
print(np.min(new_cam_coords[:,0]),np.max(new_cam_coords[:,0]))
print(np.min(new_cam_coords[:,1]),np.max(new_cam_coords[:,1]))
print(np.min(new_cam_coords[:,2]),np.max(new_cam_coords[:,-1]))

new_image_coords = (intrinsic_raw @ new_cam_coords.T / (new_cam_coords[:,2]+ 1e-6)).T
print(new_image_coords.shape)
print(np.min(new_image_coords[:,0]),np.max(new_image_coords[:,0]))
print(np.min(new_image_coords[:,1]),np.max(new_image_coords[:,1]))

new_img_x = new_image_coords[:,0]
new_img_y = new_image_coords[:,1]

new_img_x = new_img_x.reshape(height,width).astype(np.float32)
new_img_y = new_img_y.reshape(height,width).astype(np.float32)

trans = cv2.remap(color_raw,new_img_x,new_img_y,cv2.INTER_LINEAR)
cv2.imwrite('-15xyz.png',trans)
#cv2.imshow('transformed.png',trans)

