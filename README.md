# aerial_topomapping

Generate topometric maps from UAV imagery to be used with the topological navigation framework (https://github.com/LCAS/topological_navigation).
The pipeline proposed in this repo uses a pointcloud as the starting input. The pointcloud can be obtained from multiple aerial images through photogrammetry using the opensource project called Open Drone Map. For more info, please check https://opendronemap.org/ as this step not included in this wiki.

## Repo structure
- **scripts**: set of python callable scripts to run the multiple steps in the proposed pipeline. All the scripts can be called with the `--help` argument to show all the arguments available and their description. 
- **modules**: this folder includes the python modules with the functions needed for executing the different steps in the pipeline.
- **conda_env**: contains the preconfigured conda environment to install all dependencies needed to run the modules
- **data**: contains the example data and where the ouputs from all the steps should be stored. **Please do not commit more data as the pointclouds and output are quite heavy!**
- **roslaunch**: contains a ROS launch file as an example to run the navsat_transform node needed in one the steps.
- **etc**: images for the tutorial

## Dependencies:
- Python 3.7
```
sudo apt-get install python3.7
```
- ROS and robot_localization package.  
Although this is not a ROS package, there is a step in the pipeline that requires ROS (any) with the robot_localisation package installed (http://wiki.ros.org/robot_localization)
```
sudo apt-get install ros-<yourrosversion>-robot-localisation
```
- Conda environment  
The rest of the dependencies are installed using a preconfigured Conda environment (https://docs.conda.io/en/latest/).  
Installing and setting up the Conda environment:  
  - Download Miniconda for python 3.7 from: https://docs.conda.io/en/latest/miniconda.html#linux-installers
  - Install Miniconda  
  ```
  bash <path_to_you_download>/Miniconda3-XXXX-Linux-x86_64.sh
  ```
  - Open new terminal, you should be now in the default `base` environment. This is indicated in the terminal by `(base)` in front of your user.
  - Update conda:  
  ```
  conda update conda
  ```
  - Create the new aerial topomapping environment with all the dependencies:
  ```
  conda env create -f <path_to_this_repo>/aerial_topomapping/conda_env/atenv.yaml
  ```
  - Activating the new environment:
  ```
  conda activate atenv
  ```

  Important Note: by default every time you open a new terminal now you will be in the conda base enviroment. In order to avoid this behaviour you can do:  
  ```
  conda config --set auto_activate_base false
  ```

## Running the pipeline with the example data:
**1.** All the scripts (except step 7) must be run within the `atenv` evironment previously create.  To activate the conda `atenv` environment in a new terminal run:  
```
conda activate atenv
```
**2.** Go the the scripts folder:  
```
cd <path_to_this_repo>/aerial_topomapping/scripts
```
**3.** Create a binary occupancy map from the input pointcloud:  
```
python pointcloud_to_occupancymap.py --input_las_pointcloud ../data/KG_small/KG_small.las --resolution 0.1
```
![input image](https://github.com/LCAS/aerial_topomapping/blob/main/etc/input_pointcloud.png?raw=true)  
**4.** Classify all the clusters in the occupancy map that belong to rows:  
```
python row_classification.py --input_image ../data/KG_small/KG_small.tif
```
![output_classification](https://github.com/LCAS/aerial_topomapping/blob/main/etc/output_classification.png?raw=true)  
**5.** Compute the nodes that will be placed in the corridor using the row lines:  
```
python compute_row_nodes.py --input_image ../data/KG_small/KG_small.tif --labels ../data/KG_small/KG_small_labels.npy --image_resolution 0.1 --row_separation 2.7
```
![corridor_nodes](https://github.com/LCAS/aerial_topomapping/blob/main/etc/corridor_nodes.png?raw=true)  
**6.** Compute the nodes in the rest of the free space:    
```
python compute_service_nodes.py --input_image ../data/KG_small/KG_small.tif --labels ../data/KG_small/KG_small_labels.npy --mask ../data/KG_small/KG_small_mask.npy
```
![service_nodes](https://github.com/LCAS/aerial_topomapping/blob/main/etc/service_nodes_maskon.png?raw=true)  
**7.** (This script must be run **outside!** of the conda environment to avoid possible incompatibilities with the ROS environment that you have installed) Convert the nodes from longitude latitude coordinates to map coordinates. This step is performed using the ROS navsat_transform node which is part of the robot_localisation package.  
**7.1.** In one terminal run the ROS node and param configuration. The datum file defines the latitude and longitude that will be considered as the origin of the map coordinates.  
```
roslaunch <path_to_this_repo>/roslaunch/navsat.launch
```
**7.2.** In another run the script:  
```
python convert_lonlat_nodes_to_map_coordinates.py --nodes_lonlat_filename ../data/KG_small/KG_small_nodes_lonlat.json
```
**8.** Generate the topological map so it can be used with toponav:  
```
python generate_topomap.py --nodes_map_coord_filename ../data/KG_small/KG_small_nodes_map.json --output_tmap_name KG_small_topomap
```
**9.** At this point you should have a topological map (.tmap2 extension) that can be used and visualised using the topological navigation repo.
