#Usage: ./pipeLine.sh <location_of_data_to_use>/ <floorPlan>

#!/bin/bash
cd ~/Projects/3DscanData/DUC/Floor1/
find -type d -links 2 -exec mkdir -p "$1{}" \;
cd ~/Projects/c++/scanDensity
make
./csvToBinary -inFolder $1PTXFiles/ -outFolder $1binaryFiles/
cd ../cloudNormals
make
./cloudNormals -inFolder $1binaryFiles/ -outFolder $1cloudNormals/
cd ../getRotations
make
./getRotations -inFolder $1cloudNormals/ -outFolder $1densityMaps/rotations/
cd ../scanDensity
make
./scanDensity -redo -inFolder $1binaryFiles/ -outFolder $1densityMaps/ -zerosFolder $1densityMaps/zeros/ -rotFolder $1densityMaps/rotations/ -voxelFolder $1voxelGrids/
cd ../placeScan
make
./placeScan -redo -dmFolder $1densityMaps/ -preDone $1placementOptions/V1/ -preDone $1placementOptions/V2/ -voxelFolder $1voxelGrids/ -zerosFolder $1densityMaps/zeros/ -floorPlan $2 -nopreviewOut