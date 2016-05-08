#Usage: ./pipeLine.sh <location_of_data_to_use>
#CSE Scale = 98.0
#DUC Scale = 73.5

#!/bin/bash

#Make all the needed folders
cd $1
mkdir -p ./placementOptions/V1
mkdir -p ./placementOptions/V2
mkdir -p ./panoramas/images
mkdir -p ./panoramas/data
mkdir -p ./Archives
mkdir -p ./cloudNormals
mkdir -p ./descriptors
mkdir -p ./binaryFiles
mkdir -p ./densityMaps/R3
mkdir -p ./densityMaps/R0
mkdir -p ./densityMaps/rotations
mkdir -p ./densityMaps/R1
mkdir -p ./densityMaps/R2
mkdir -p ./densityMaps/zeros
mkdir -p ./voxelGrids/R3
mkdir -p ./voxelGrids/R0
mkdir -p ./voxelGrids/R1
mkdir -p ./voxelGrids/R2
mkdir -p ./voxelGrids/metaData
mkdir -p ./PTXFiles
mkdir -p ./SIFT
#run the 3 programs
cd ~/Projects/c++/preprocessor
make
./preprocessor -dataPath=$1 -redo -ptx
cd ../scanDensity
make
./scanDensity -dataPath=$1 -redo
cd ../placeScan
make
./placeScan -dataPath=$1 -redo