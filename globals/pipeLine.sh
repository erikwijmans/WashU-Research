#Usage: ./pipeLine.sh <location_of_data_to_use>
#CSE Scale = 98.0
#DUC Scale = 73.5

#!/bin/bash

#Make all the needed folders
mkdir -p $1
cd $1
mkdir -p placementOptions/V1
mkdir -p placementOptions/V2
mkdir -p panoramas/images
mkdir -p panoramas/data
mkdir -p cloudNormals
mkdir -p binaryFiles
mkdir -p densityMaps/R3
mkdir -p densityMaps/R0
mkdir -p densityMaps/rotations
mkdir -p densityMaps/R1
mkdir -p densityMaps/R2
mkdir -p densityMaps/zeros
mkdir -p voxelGrids/R3
mkdir -p voxelGrids/R0
mkdir -p voxelGrids/R1
mkdir -p voxelGrids/R2
mkdir -p voxelGrids/metaData

#run the 3 programs
cd ~/Projects/c++/preprocessor
echo "Running preprocessor"
make || exit 1
./preprocessor -dataPath=$1 -ptx
# echo "Running scanDensity"
# cd ~/Projects/c++/scanDensity
# make || exit 1
# ./scanDensity -dataPath=$1 -redo
# echo "Running placeScan"
# cd ~/Projects/c++/placeScan
# make || exit 1
# ./placeScan -dataPath=$1 -V1 -redo
