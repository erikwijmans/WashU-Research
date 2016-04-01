#Usage: ./pipeLine.sh <location_of_data_to_use>/
#CSE Scale = 98.0
#DUC Scale = 73.5

#!/bin/bash


cd ~/Projects/3DscanData/DUC/Floor1/
find -type d -links 2 -exec mkdir -p "$1{}" \;
cd ~/Projects/c++/preprocessor
make
./preprocessor -dataPath=$1
cd ../getRotations
make
./getRotations -dataPath=$1
cd ../scanDensity
make
./scanDensity -dataPath=$1 -redo
cd ../placeScan
make
./placeScan -dataPath=$1 -redo -V1