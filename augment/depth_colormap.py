import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float) / 256.
    #depth[depth_png == 0] = -1.
    return depth

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

def disp2depth_kitti_K(disp, k_value): 

    mask = (disp > 0).astype(float)
    depth = k_value * 0.54 / (disp + (1.0 - mask))

    return depth

#read intrinsic
intrinsics_raw = read_raw_calib_file('calib_cam_to_cam.txt')
intrinsics_raw = np.reshape(intrinsics_raw['P_rect_02'],(3,4))
intrinsic = intrinsics_raw[:3,:3]


GT_depth = depth_read('original.png')
depth_sparse = depth_read('depth_kitti.png')
generated_depth = depth_read('generated.png')

print("Max & min of sparse:",np.max(GT_depth),np.min(GT_depth))
print("Max & min of dense:",np.max(generated_depth),np.min(generated_depth))
print("Max & min of dense:",np.max(depth_sparse),np.min(depth_sparse))

GT_depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(GT_depth, alpha=255/GT_depth.max()), cv2.COLORMAP_JET)
generated_depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(generated_depth, alpha=255/generated_depth.max()), cv2.COLORMAP_JET)
depth_sparse_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_sparse, alpha=255/depth_sparse.max()), cv2.COLORMAP_JET)
cv2.imwrite('GT_depth_colored.png',GT_depth_colored)
cv2.imwrite('generated_depth_colored.png',generated_depth_colored)
cv2.imwrite('depth_sparse_colored.png',depth_sparse_colored)


# GT_normed_depth = (GT_depth / GT_depth.max() * 255).astype(np.uint8)
# plt.imsave('gt_normed_depth.png', GT_normed_depth, cmap='plasma',vmin=0,vmax=255)
# depth_sparse_normed = (depth_sparse / depth_sparse.max() * 255).astype(np.uint8)
# plt.imsave('depth_sparse_normed.png', depth_sparse_normed, cmap='plasma',vmin=0,vmax=255)
# generated_depth_normed = (generated_depth / generated_depth.max() * 255).astype(np.uint8)
# plt.imsave('generated_depth_normed.png', generated_depth_normed, cmap='plasma',vmin=0,vmax=255)

#perform inverse transformation
inversed_GT = disp2depth_kitti_K(GT_depth, intrinsic[0, 0])
print("Max & min of dense:",np.max(inversed_GT),np.min(inversed_GT))
inversed_GT_colored = cv2.applyColorMap(cv2.convertScaleAbs(inversed_GT, alpha=255/inversed_GT.max()), cv2.COLORMAP_JET)
cv2.imwrite('inversed_GT_colored.png',inversed_GT_colored)

