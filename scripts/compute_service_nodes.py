import numpy as np
import matplotlib.pyplot as plt
import rasterio
import json
import os
import sys
import argparse
from os.path import exists
from json_merge_patch import merge

script_dir = os.path.dirname( __file__ )
modules_dir = os.path.join( script_dir, '../modules' )
modules_dir = os.path.normpath(modules_dir)
sys.path.append(modules_dir)
import aerial_topomapping as at


parser = argparse.ArgumentParser(description='Compute the service nodes in the free space')
parser.add_argument("--input_image", type=str,help="Input TIF Image",required=True)
parser.add_argument("--labels", type=str,help="Labels array with per pixel classification",required=True)
parser.add_argument("--mask", type=str,help="Mask array to define extra pixels considered non-free",default= [])
parser.add_argument("--type", type=int,help="[0] skeleton (deafault), [1] Grid",default=0)
parser.add_argument("--plot", type=bool, help="Whether to show the output or not [Default: True]",default=True)
args = parser.parse_args()

image_raw = rasterio.open(args.input_image)
labels = np.load(args.labels)

###############################################################
# Apply mask in case it is provided
###############################################################
if args.mask != []:
	mask = np.load(args.mask,allow_pickle=True)
	image = (labels + mask) > 0 
else:
	image = labels > 0

###############################################################
# Compute service nodes depending on the type
###############################################################
if args.type == 0: # skelenotization process
	service_toponodes_pix, graph = at.compute_service_nodes_skeleton(image)
elif args.type == 1:
	nodes_grid = at.create_topogrid(image, 50)
	service_toponodes_pix = []

###############################################################
# convert pix coordinates to lon/lat coordinates
###############################################################
service_nodes_lonlat_json = at.reproject_service_nodes_coordinates_to_lonlat(service_toponodes_pix,str(image_raw.crs),image_raw.transform)
if exists(args.input_image[:-4]+'_nodes_lonlat.json'):
	print("Nodes files already exist, merging/updating with the new corridor nodes")
	with open(args.input_image[:-4]+'_nodes_lonlat.json', 'r') as f:
		service_nodes_lonlat_json_base = json.load(f)
	service_nodes_lonlat_json_merge = merge(service_nodes_lonlat_json_base,service_nodes_lonlat_json)
	with open(args.input_image[:-4]+'_nodes_lonlat.json','w') as f:
		json.dump(service_nodes_lonlat_json_merge,f,indent=4)
else:
	print("Nodes files don't exist, create new with the new corridor nodes")
	with open(args.input_image[:-4]+'_nodes_lonlat.json','w') as f:
		json.dump(service_nodes_lonlat_json,f,indent=4)


###############################################################
# visualisation
###############################################################
if args.plot:

	fig, axes = plt.subplots(nrows=1, ncols=1)
	axes.imshow(image, cmap='gray')

	if args.type == 0: # skelenotization process
		# #draw edges by pts
		for (s,e) in graph.edges():
			if len(graph)> 1:
				ps = graph[s][e]['pts']
				axes.plot(ps[:,1], ps[:,0], 'green')

		axes.scatter(service_toponodes_pix[:,1], service_toponodes_pix[:,0],55, 'y',)

	elif args.type == 1:
		for n in nodes_grid:
		 	axes.plot(n[1], n[0], 'y.')




	plt.show()
