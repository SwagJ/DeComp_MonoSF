import cv2
import numpy as np


depth = cv2.imread('depth.png',cv2.IMREAD_ANYDEPTH) / 1000
#cv2.imshow('depth_og',depth)
#cv2.waitKey(0)
print(np.sum(depth > 30))

im_color=cv2.applyColorMap(cv2.convertScaleAbs(depth,alpha=255/depth.max()),cv2.COLORMAP_JET)
#cv2.imshow('depth_color',im_color)
cv2.imwrite('depth_color.png',im_color)
cv2.waitKey(0)

