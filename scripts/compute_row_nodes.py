import rasterio
import matplotlib.image
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import argparse
import sys
from os.path import exists
from json_merge_patch import merge

script_dir = os.path.dirname( __file__ )
modules_dir = os.path.join( script_dir, '../modules' )
modules_dir = os.path.normpath(modules_dir)
sys.path.append(modules_dir)
import aerial_topomapping as at


parser = argparse.ArgumentParser(description='Compute the nodes in each side of the rows')
parser.add_argument("--input_image", type=str,help="Input TIF Image",required=True)
parser.add_argument("--labels", type=str,help="Labels array with per pixel classification",required=True)
parser.add_argument("--image_resolution", type=float,help="Image resolution in m/pix",required=True)
parser.add_argument("--row_separation", type=float, help="Average distance between rows [m]",required=True)
parser.add_argument("--merge_row_lines_distance_threshold", type=float, help="Merge row lines distance threshold [Default: 5 m]",default=5)
parser.add_argument("--merge_row_lines_angle_threshold", type=float, help="Average distance between rows [Default: 1 degree]",default=1)
parser.add_argument("--distance_precorridor_nodes", type=float, help="Distance between the node at the beginning of the corridor and the one outside [Default: 2 m]",default=2)
parser.add_argument("--merge_corridor_distance_threshold", type=float, help="Distance threshold to merge corridor nodes  [Default: 1.5 m]",default=1.5)
parser.add_argument("--row_label_number", type=int, help="Label defining cluster belonging to rows [Default: 2]",default=2)
parser.add_argument("--plot", type=bool, help="Whether to show the output or not [Default: True]",default=True)
args = parser.parse_args()

image_raw = rasterio.open(args.input_image)
labels = np.load(args.labels)
rows_only_image = labels == args.row_label_number

###############################################################
# Compute the vine row lines
###############################################################
row_lines = at.compute_row_lines(rows_only_image)
row_lines = at.merge_row_lines(row_lines,args.image_resolution, args.merge_row_lines_distance_threshold , args.merge_row_lines_angle_threshold)
#np.save(image_filename[:-4]+'_row_lines',row_lines)

###############################################################
# Compute the corridor topological nodes
###############################################################
corridor_toponodes_pix,mask_image = at.compute_corridor_nodes(row_lines, args.image_resolution, args.row_separation, 0,args.distance_precorridor_nodes,rows_only_image.shape)
corridor_toponodes_pix = at.merge_corridor_nodes(corridor_toponodes_pix, args.image_resolution, args.merge_corridor_distance_threshold)
#np.save(image_filename[:-4]+'_corridor_toponodes_pix',corridor_toponodes_pix)
np.save(args.input_image[:-4]+'_mask',mask_image) # mask with all the points that we don't waht to consider as free space anymore (like the aread inside the corridors)

###############################################################
# convert pix coordinates to lon/lat coordinates
###############################################################
corridor_nodes_lonlat_json = at.reproject_corridor_nodes_coordinates_to_lonlat(corridor_toponodes_pix,str(image_raw.crs),image_raw.transform)
if exists(args.input_image[:-4]+'_nodes_lonlat.json'):
	print("Nodes files already exist, merging/updating with the new corridor nodes")
	with open(args.input_image[:-4]+'_nodes_lonlat.json', 'r') as f:
		corridor_toponodes_lonlat_base = json.load(f)
	corridor_nodes_lonlat_json_merge = merge(corridor_toponodes_lonlat_base,corridor_nodes_lonlat_json)
	with open(args.input_image[:-4]+'_nodes_lonlat.json','w') as f:
		json.dump(corridor_nodes_lonlat_json_merge,f,indent=4)
else:
	print("Nodes files don't exist, create new with the new corridor nodes")
	with open(args.input_image[:-4]+'_nodes_lonlat.json','w') as f:
		json.dump(corridor_nodes_lonlat_json,f,indent=4)

###############################################################
# visualisation
###############################################################
if args.plot:
	#plotting
	fig, axes = plt.subplots(nrows=1, ncols=1)
	axes.imshow(rows_only_image==0,cmap="gray")
	for c in corridor_toponodes_pix:
	 	axes.scatter(c[0],c[1],75,'r')
	 	axes.scatter(c[2],c[3],75,'r')
	 	axes.scatter(c[4],c[5],75,'r')
	 	axes.scatter(c[6],c[7],75,'r')
	 	axes.plot([c[0],c[2],c[4],c[6]],[c[1],c[3],c[5],c[7]],'r')

	for r in row_lines:
		axes.scatter(r[0],r[1],75,'g')
		axes.scatter(r[2],r[3],75,'g')
		axes.plot([r[0],r[2]],[r[1],r[3]],'g')

	## matplotlib.image.imsave(args.input_image[:-4]+'_labels.png', labels,cmap="gist_ncar_r")

plt.show()