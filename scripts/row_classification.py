import rasterio
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
import argparse
import os
import sys

script_dir = os.path.dirname( __file__ )
modules_dir = os.path.join( script_dir, '../modules' )
modules_dir = os.path.normpath(modules_dir)
sys.path.append(modules_dir)
import aerial_topomapping as at

parser = argparse.ArgumentParser(description='Transforms a pointcloud into a non traversable occupancy map')
parser.add_argument("--input_image", help="Input TIF Image",required=True)
parser.add_argument("--cluster_ratio_threshold", help="Output image resolution [Default: 30]",default=30)
parser.add_argument("--label_number", type=int, help="Label defining cluster belonging to rows [Default: 2]",default=2)
parser.add_argument("--plot", help="Whether to show the output or not [Default: True]",default=True,type=bool)
args = parser.parse_args()

image_raw = rasterio.open(args.input_image)
image = image_raw.read(1) # read the first band

###############################################################
# obtain the clusters classified as vine rows
###############################################################
# INPUT: tif image, ratio to detect the row cluster, label assigned to rows
# OUPUT: numpy array with the labeled image
labels = at.row_detection(image,args.cluster_ratio_threshold,args.label_number) 
np.save(args.input_image[:-4]+'_labels',labels)

if args.plot:
	#plotting
	fig, axes = plt.subplots(nrows=1, ncols=1)
	axes.imshow(labels,cmap="gist_ncar_r")
	#matplotlib.image.imsave(args.input_image[:-4]+'_labels.png', labels,cmap="gist_ncar_r")
	plt.show()
