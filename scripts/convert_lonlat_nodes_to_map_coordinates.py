import json
import rospy
from geographic_msgs.msg import GeoPoint
from robot_localization.srv import FromLL
import argparse

def convert_nodes_from_lonlat_to_map_coordinates_with_ros_navsat(toponodes_lonlat):
	
	print ("-- Converting toponodes locations from Lon/Lat to Map coordinates with ROS Navsat Node --")
	print ("Waiting for the 'fromLL' rosservice to be available...")
	rospy.wait_for_service("fromLL")
	print("Rosservice ready. Computing the transforms for all the nodes...")
	
	toponodes_map_coord = {}
	toponodes_map_coord['corridor_nodes'] = []
	toponodes_map_coord['service_nodes'] = []

	try:
		fromLL_service = rospy.ServiceProxy('fromLL', FromLL)
		for c in toponodes_lonlat['corridor_nodes']:
			temp_corridor = []
			for p in range(0,8,2):
				geopoint_msg = GeoPoint()
				geopoint_msg.longitude = c[p]
				geopoint_msg.latitude = c[p+1]
				resp = fromLL_service(geopoint_msg)
				x_map = resp.map_point.x
				y_map = resp.map_point.y
				temp_corridor.append(x_map)
				temp_corridor.append(y_map)
			toponodes_map_coord['corridor_nodes'].append(temp_corridor)

		for n in toponodes_lonlat['service_nodes']:
			geopoint_msg = GeoPoint()
			geopoint_msg.longitude = n[0]
			geopoint_msg.latitude = n[1]
			resp = fromLL_service(geopoint_msg)
			x_map = resp.map_point.x
			y_map = resp.map_point.y
			toponodes_map_coord['service_nodes'].append([x_map,y_map])

	except rospy.ServiceException as e:
		print("Service call failed: %s"%e)		
	print("-- Done --")
	return toponodes_map_coord

parser = argparse.ArgumentParser(description='Transform toponodes from lonlat to map coordinates with Ros Navsat node')
parser.add_argument("--nodes_lonlat_filename", type=str,help="Toponodes lonlat filename",required=True)
args = parser.parse_args()

#read the toponodes file
with open(args.nodes_lonlat_filename, 'r') as f:
	nodes_lonlat = json.load(f)

# transform the geo located nodes to the ros map coordinates 
nodes_map_coord = convert_nodes_from_lonlat_to_map_coordinates_with_ros_navsat(nodes_lonlat)

# save into a new file
with open(args.nodes_lonlat_filename[:-11]+'map.json','w') as f:
	json.dump(nodes_map_coord, f,indent=4)
