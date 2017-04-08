#!/usr/bin/env zsh

function copyImgs {
  mkdir -p $1
  cd $1
  cp -rv ../../ReleaseData/$1/imgs .
  cp -rv ../../ReleaseData/$1/lowres .
  cd ..
}

copyImgs DUC1
copyImgs DUC2
copyImgs CSE3
copyImgs CSE4
copyImgs CSE5