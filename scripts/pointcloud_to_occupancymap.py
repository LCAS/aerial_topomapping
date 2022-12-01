import pdal
import argparse

pipeline_json = """
[
    {
        "type":"readers.las",
        "filename":"%s"
    },
    {
        "type":"filters.outlier",
        "method":"statistical"
    },
    {
        "type":"filters.smrf"
    },
    {
        "type":"filters.range",
        "limits":"Classification[1:1]"
    },
    {
        "type":"writers.las",
        "filename":"%s"
    },
    {
        "type": "writers.gdal",
        "gdaldriver":"GTiff",
        "output_type":"max",
        "resolution":"%s",
        "nodata":"0",
        "filename":"%s"
    }
]"""

parser = argparse.ArgumentParser(description='Transforms a pointcloud into a non traversable occupancy map')
parser.add_argument("--input_las_pointcloud", type=str,help="Input Pointcloud",required=True)
parser.add_argument("--resolution", type=float, help="Output image resolution [m/pix]",required=True)
args = parser.parse_args()

pipeline_json = pipeline_json % (args.input_las_pointcloud, args.input_las_pointcloud[:-4] + "_non_ground.las",args.resolution,args.input_las_pointcloud[:-4] + ".tif" )
pipeline = pdal.Pipeline(pipeline_json)
count = pipeline.execute()
