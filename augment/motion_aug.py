#import open3d as o3d
from matplotlib import pyplot as plt
import numpy as np
import cv2
import math
from PIL import Image

import scipy.misc as sp
instance_semantic_gt = sp.imread('instance.png')
instance = instance_semantic_gt // 256
og = cv2.imread('og.png')


print(instance.shape,og.shape)
print(np.unique(instance))





