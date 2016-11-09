#Usage: ./pipeLine.sh <location_of_data_to_use> <location_of_build_dir>
#CSE Scale = 98.0
#DUC Scale = 73.5

#!/bin/bash

#Make all the needed folders
mkdir -p $1/placementOptions/V1
mkdir -p $1/placementOptions/V2
mkdir -p $1/panoramas/images
mkdir -p $1/panoramas/data
mkdir -p $1/cloudNormals
mkdir -p $1/binaryFiles
mkdir -p $1/densityMaps/R3
mkdir -p $1/densityMaps/R0
mkdir -p $1/densityMaps/rotations
mkdir -p $1/densityMaps/R1
mkdir -p $1/densityMaps/R2
mkdir -p $1/densityMaps/zeros
mkdir -p $1/voxelGridcs/R3
mkdir -p $1/voxelGrids/R0
mkdir -p $1/voxelGrids/R1
mkdir -p $1/voxelGrids/R2
mkdir -p $1/voxelGrids/metaData
mkdir -p $1/doors/pointcloud
mkdir -p $1/doors/floorplan

#run the 4 programs

# echo "Running preprocessor"
# make --no-print-directory -j4 -C $2/preprocessor || exit 1
# preprocessor=$2/preprocessor/preprocessor
# $preprocessor -dataPath=$1 -redo || exit 1

# echo "Running scanDensity"
# make --no-print-directory -j4 -C $2/scanDensity || exit 1
# scanDensity=$2/scanDensity/scanDensity
# $scanDensity -dataPath=$1 -redo || exit 1

echo "Running placeScan"
make --no-print-directory -j4 -C $2/placeScan || exit 1
placeScan=$2/placeScan/placeScan
$placeScan -dataPath=$1 -redo -V1 || exit 1

# echo "Running joiner"
# make --no-print-directory -j4 -C $2/joiner || exit 1
# joiner=$2/joiner/joiner
# $joiner -dataPath=$1 -redo || exit 1
