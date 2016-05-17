#!/bin/bash
# usage: ./driver.sh <root dir>
echo "Starting pipline with dataPath=$1DUC/Floor1"
./pipeLine.sh $1/DUC/Floor1
echo "Starting pipline with dataPath=$1DUC/Floor2"
./pipeLine.sh $1/DUC/Floor2
echo "Starting pipline with dataPath=$1CSE/Floor3"
./pipeLine.sh $1/CSE/Floor3
echo "Starting pipline with dataPath=$1CSE/Floor4"
./pipeLine.sh $1/CSE/Floor4
echo "Starting pipline with dataPath=$1CSE/Floor5"
./pipeLine.sh $1/CSE/Floor5