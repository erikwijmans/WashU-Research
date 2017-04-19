#!/usr/bin/env zsh

function copyImgs {
  mkdir -p $1
  cd $1
  cp -rfv /home/erik/Projects/ReleaseData/$1/imgs .
  cp -rfv /home/erik/Projects/ReleaseData/$1/lowres .
  cd ..
}

copyImgs DUC1
copyImgs DUC2
copyImgs CSE3
copyImgs CSE4
copyImgs CSE5