# Exploiting 2D Floorplan for Building-scale Panorama RGBD Alignment

This is the git repo with the code for our [CVPR17 Paper](http://cvpr17.wijmans.xyz/CVPR2017-0111.pdf).  Visit the [project site](http://cvpr17.wijmans.xyz) for more information and the data.

## Usage
1. Get the dependencies:

    * [OpenCV](http://opencv.org/releases.html)
    * [Point Cloud Library](http://pointclouds.org/downloads/)
    * [OpenGM](http://hciweb2.iwr.uni-heidelberg.de/opengm/index.php?l0=library)
    * [Boost](http://www.boost.org)
    * [fmtlib](http://fmtlib.net/latest/index.html)
    * [gflags](https://gflags.github.io/gflags/)
    * [glog](https://github.com/google/glog/releases)

    As of right now, I am using some c++17 features, which can be compiled by [clang++-4.0](http://releases.llvm.org) or g++-7.0 (which should be available soon)

2. Download the latest release from the releases tab: [https://github.com/erikwijmans/WashU-Research/releases](https://github.com/erikwijmans/WashU-Research/releases)
3. Building
    ```
    git submodule update --init --recursive
    mkdir build
    cd build
    cmake ..
    make
    ```
4. Running
    `pipLine.sh` shows a suggest way to run the 4 programs in order to replicate our results.  It will also create the folder structure expected:

    ```
    ├── binaryFiles
    ├── cloudNormals
    ├── densityMaps
    │   ├── R0
    │   ├── R1
    │   ├── R2
    │   ├── R3
    │   ├── rotations
    │   └── zeros
    ├── doors
    │   ├── floorplan
    │   └── pointcloud
    ├── panoramas
    │   ├── data
    │   └── images
    ├── placementOptions
    │   ├── V1
    │   └── V2
    ├── PTXFiles
    └── voxelGrids
        ├── metaData
        ├── R0
        ├── R1
        ├── R2
        └── R3
    ```

    Before running `pipeLine.sh`, this is what the programs expect to be present:

    ```
    ├── PTXFiles
    │   ├── PTX_1
    │   ├── ....
    ├── scale.txt
    ├── doorSymbol.png
    └── floorPlan.png
    ```
    Where `scale.txt` simply contains the number of pixels on the floor per unit distance of the scans.  The `doorSymbol.png` is simply a doorSymbol from the floorplan.  `floorPlan.png` is the ground truth floorplan. `PTXFiles` is a folder containing with all the scans in the PTX format.  `pipeLine.sh` should then be run as such:

    `./pipeLine.sh /path/to/PTXFiles/.. /path/to/build`