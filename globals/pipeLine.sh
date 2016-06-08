#Usage: ./pipeLine.sh <location_of_data_to_use> <location_of_progs>
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
mkdir -p $1/voxelGrids/R3
mkdir -p $1/voxelGrids/R0
mkdir -p $1/voxelGrids/R1
mkdir -p $1/voxelGrids/R2
mkdir -p $1/voxelGrids/metaData

#run the 3 programs
# cd $2/preprocessor
# echo "Running preprocessor"
# make || exit 1
# ./preprocessor -dataPath=$1
# echo "Running scanDensity"
cd $2/scanDensity
make || exit 1
./scanDensity -dataPath=$1 -redo -2D -pe
# echo "Running placeScan"
# cd $2/placeScan
# make || exit 1
# ./placeScan -dataPath=$1 -V1 -redo -threads=4
