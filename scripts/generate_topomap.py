import os
import sys
import json
import yaml
import argparse

script_dir = os.path.dirname( __file__ )
modules_dir = os.path.join( script_dir, '../modules' )
modules_dir = os.path.normpath(modules_dir)
sys.path.append(modules_dir)
import aerial_topomapping as at


parser = argparse.ArgumentParser(description='Transform the file with the nodes in map coordinates to a topomap that can be used by the topological_navigation package')
parser.add_argument("--nodes_map_coord_filename", type=str,help="Toponodes map coordinates filename",required=True)
parser.add_argument("--output_tmap_name", type=str,help="Name of the outpul file",required=True)
args = parser.parse_args()


#read the toponodes file
with open(args.nodes_map_coord_filename, 'r') as f:
	nodes_map_coord = json.load(f)

#load node and edges templates
with open("./template_yaml_structure/template_toponode.yaml", 'r') as f:
	template_toponode = yaml.safe_load(f)

with open("./template_yaml_structure/template_topoedge.yaml", 'r') as f:
	template_topoedge = yaml.safe_load(f)

topomap = at.generate_topological_map(nodes_map_coord,args.output_tmap_name,template_toponode,template_topoedge)

path = ""
for p in args.nodes_map_coord_filename.split("/")[:-1]:
	path = path + p + "/"
with open(path+args.output_tmap_name+".tmap2",'w') as f:
	yaml.dump(topomap, f)
