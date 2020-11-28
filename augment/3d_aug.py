#import open3d as o3d
from matplotlib import pyplot as plt
import numpy as np
import cv2
import math
from PIL import Image
import argparse

def str2bool(v):
  return v.lower() in ('true', '1')

def pixel_coord_np(width, height):
    x = np.linspace(0, width - 1, width).astype(np.int)
    y = np.linspace(0, height - 1, height).astype(np.int)
    [x, y] = np.meshgrid(x, y)
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))

def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.
    return depth

def get_r_matrix(rx=0,ry=0,rz=0):
	theta_x = 2*math.pi * rx / 360
	theta_y = 2*math.pi * ry / 360
	theta_z = 2*math.pi * rz / 360

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

	return total_R

def get_shear_x(sy=0,sz=0):
	Shx = np.array([[1,0,0,0],
				[sy,1,0,0],
				[sz,0,1,0],
				[0,0,0,1]])

	return Shx

def get_shear_y(sx=0,sz=0):
	Shy = np.array([[1,sx,0,0],
				[0,1,0,0],
				[0,sz,1,0],
				[0,0,0,1]])
	return Shy

def get_shear_z(sx=0,sy=0):
	Shz = np.array([[1,0,sx,0],
				[0,1,sy,0],
				[0,0,1,0],
				[0,0,0,1]])
	return Shz

def read_raw_calib_file(filepath):
    # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data

def perform_shearing(cam_coords, axis, s1,s2):
	if axis=='x':
		sh = get_shear_x(s1,s2)
	elif axis=='y':
		sh = get_shear_y(s1,s2)
	else:
		sh = get_shear_z(s1,s2)

	#expand input dim
	ones = np.ones((cam_coords.shape[0],1))
	print("ones shape:",ones.shape)
	cam_coords = np.concatenate((cam_coords,ones),axis=1)
	print("concatenated shape:",cam_coords.shape)

	shed_cam_coords = np.matmul(sh,cam_coords.T).T

	return shed_cam_coords[:,:3],sh


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-rotate', action='store_true')
	parser.add_argument('-trans', action='store_true')
	parser.add_argument('-shear', action='store_true')
	parser.add_argument('-recenter', action='store_true')
	
	parser.add_argument('-rx', type=int, default=0)
	parser.add_argument('-ry', type=int, default=0)
	parser.add_argument('-rz', type=int, default=0)

	parser.add_argument('-tx', type=float, default=0)
	parser.add_argument('-ty', type=float, default=0)
	parser.add_argument('-tz', type=float, default=0)

	parser.add_argument('-axis', type=str, default='x')
	parser.add_argument('-s1', type=float, default=0)
	parser.add_argument('-s2', type=float, default=0)


	parser.add_argument('-save_dir', type=str, default='./aug_kitti_example')

	args = parser.parse_args()

	#load image
	depth = depth_read('completed_depth_kitti.png')
	color_raw = cv2.imread('rgb.png')

	# for kitti
	intrinsics_raw = read_raw_calib_file('calib_cam_to_cam.txt')
	intrinsics_raw = np.reshape(intrinsics_raw['P_rect_02'],(3,4))
	intrinsic = intrinsics_raw[:3,:3]
	inv_intrinsic = np.linalg.inv(intrinsic)

	#image to coord list + RGB

	height = depth.shape[0]
	width = depth.shape[1]

	#reproject to cam_coord
	image_coord = pixel_coord_np(width,height)
	cam_coords = (inv_intrinsic[:3, :3] @ image_coord * depth.flatten()).T

	center = np.array([height/2, width/2,1])
	center_depth = depth[int(height/2),int(width/2)]
	center_in_cam_coord = (inv_intrinsic[:3, :3] @ center * center_depth).T
	print(center_in_cam_coord.shape)


	filename = ''

	if args.rotate == True:
		print("Using rotation: rx ry rz",args.rx,args.ry,args.rz)
		R = get_r_matrix(args.rx,args.ry,args.rz)
		cam_coords = np.matmul(cam_coords,R)
		center_in_cam_coord = np.matmul(center_in_cam_coord,R)
		filename += ('x'+str(args.rx)+'_y'+str(args.ry)+'_z'+str(args.rz))
	else:
		filename += 'NoR_'

	if args.shear == True:
		print("Using Shearing along " + args.axis + "axis: s1 s2" + str(args.s1) + str(args.s2))
		cam_coords,sh = perform_shearing(cam_coords,args.axis,args.s1,args.s2)
		center_in_cam_coord = np.append(center_in_cam_coord,1)
		center_in_cam_coord = np.matmul(sh,center_in_cam_coord.T).T[:3]
		filename += (args.axis +'_s1_'+str(args.s1)+'_s2_'+str(args.s2))
	else:
		filename += 'NoS_'

	if args.trans == True:
		print("Using translation: ",[args.tx,args.ty,args.tz])
		T = [args.tx,args.ty,args.tz]
		valid_area = cam_coords[:,2] >= -T[-1]
		cam_coords[valid_area,-1] = cam_coords[valid_area,-1] + T[-1]
		if center_in_cam_coord[-1] >= -T[-1]:
			center_in_cam_coord = center_in_cam_coord + T

		filename += '_t'+str(args.tx)+'_'+str(args.ty)+'_'+str(args.tz)
	else:
		filename += 'NoT_'

	#back proj
	print(center_in_cam_coord.shape)
	new_image_coords = (intrinsic @ cam_coords.T / (cam_coords[:,2]+ 1e-6)).T
	new_image_center = (intrinsic @ center_in_cam_coord.T / (center_in_cam_coord[2]+ 1e-6)).T

	new_img_x = new_image_coords[:,0]
	new_img_y = new_image_coords[:,1]

	# bring warpped image back to frame
	offset_x = int(height/2) - new_image_center[0]
	offset_y = int(width/2) - new_image_center[1]


	if args.recenter == True:
		new_img_x = new_img_x + offset_x
		new_img_y = new_img_y + offset_y
		filename += '_C'
	else:
		filename += '_NC'

	new_img_x = (new_img_x).reshape(height,width).astype(np.float32)
	new_img_y = (new_img_y).reshape(height,width).astype(np.float32)


	trans = cv2.remap(color_raw,new_img_x,new_img_y,cv2.INTER_LINEAR)
	final_path = args.save_dir + '/' + filename + '.png'
	cv2.imwrite(final_path,trans)


if __name__ == "__main__":
	main()






