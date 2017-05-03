<!DOCTYPE html>
<html lang="en-us">
  <head>
    <meta charset="UTF-8">
    <title>Panorama RGBD Alignment</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="icon" href="Brookings_Seal(RGB).png">
    <link rel="stylesheet" type="text/css" href="../stylesheets/normalize.css" media="screen">
    <link href='https://fonts.googleapis.com/css?family=Open+Sans:400,700' rel='stylesheet' type='text/css'>
    <link rel="stylesheet" type="text/css" href="../stylesheets/stylesheet.css" media="screen">
    <link rel="stylesheet" type="text/css" href="../stylesheets/github-light.css" media="screen">
    <link rel="stylesheet" type="text/css" href="../stylesheets/prism.css">
    <script src="https://code.jquery.com/jquery-3.1.1.min.js" integrity="sha256-hVVnYaiADRTO2PzUGmuLJr8BLUSjGIZsDYGmIJLv2b8=" crossorigin="anonymous"></script>
  </head>
  <body>
    <section class="page-header">
      <h1 class="project-name">Exploiting 2D Floorplan for Building-scale Panorama RGBD Alignment - Dataset</h1>
      <h2 class="project-tagline"><a class="site-link" href="http://wijmans.xyz">Erik Wijmans</a> and <a class="site-link" href="http://www.cse.wustl.edu/~furukawa/"> Yasutaka Furukawa</a></h2>
      <center>
      <a href="https://github.com/erikwijmans/WashU-Research/tree/v0.1" class="btn">Code</a>
      <a href="/CVPR2017-0111-supp.pdf" class="btn">Paper</a>
      <a href="/" class="btn">Project</a>
      </center>
    </section>
<section class="main-content">

# Point clouds

1. Downloading

    | Building Name (Building Code) |Floor # (Floor Code) | Download Link   | Number of Scans | Area (ft^2)  |% Correct  |
    |:---:|:---:|:---:|:---:|:---:|:---:|
    |Center (DUC)|Floor 1 (DUC1)| [LRZ (15GB)](https://wustl.box.com/v/wustl3d-duc1-lrz) <br>   [ZIP (24GB)](https://wustl.box.com/v/wustl3d-duc1-zip)   | 50 | ~100,000 |  98%   |
    | Center (DUC) |Floor 2 (DUC2)| [LRZ (23GB)](https://wustl.box.com/v/wustl3d-duc2-lrz) <br>  [ZIP (39GB)](https://wustl.box.com/v/wustl3d-duc2-zip)   | 80 | ~100,000 |  100% |
    | Hall (CSE) |Floor 3 (CSE3) | [LRZ (2.4GB)](https://wustl.box.com/v/wustl3d-cse3-lrz) <br>  [ZIP (3.9GB)](https://wustl.box.com/v/wustl3d-cse3-zip)   | 7 | ~8,000  | 85%   |
    | Hall (CSE) |Floor 4 (CSE4) | [LRZ (24GB)](https://wustl.box.com/v/wustl3d-cse4-lrz) <br> [ZIP (40GB)](https://wustl.box.com/v/wustl3d-cse4-zip)  | 75 | ~19,000 |  93%  |
    | Hall (CSE) |Floor 5 (CSE5) | [LRZ (22GB)](https://wustl.box.com/v/wustl3d-cse5-lrz) <br> [ZIP (36GB)](https://wustl.box.com/v/wustl3d-cse5-zip)  | 65 | ~20,000 |  66%  |


2. Decompressing

    The point-clouds are available as LRZ or ZIP compressed PTX files.
    We provide lrzipped to accommodate those with slow Internet connections and/or data-caps.  Here is a quick guide to getting lrzip and decompressing the files.  However, we will not provide any additional support in terms of using/obtaining lrzip and encourage you to simply use the zipped versions if possible.

    1. Install lrzip

        * Debian/Ubuntu

            `sudo apt-get install lrzip`

        * Fedora

            `sudo yum install lrzip`

        * OS X

            `brew install lrzip`

        * Windows

            [Cygwin](https://www.cygwin.com/) provides a version of lrzip

    2. Decompress

        `lrunzip *.lrz`

3. Usage

    Files are named as such:  `<Building Code>_scan_<Scan_ID>.ptx`

    Information on the PTX format can be found [here](http://w3.leica-geosystems.com/kb/?guid=5532D590-114C-43CD-A55F-FE79E5937CB2).
    We only utilize the rows and columns field and transformation matrices are in separate files however.

    The points in the PTX file constitute as column-major panorama with the number of rows and columns specified in the PTX header.

# Transformation Matrices

1. Download

    |Floor Code | Link |
    |:---:|:---:|
    | DUC1 | [DUC1 Alignment](https://wustl.box.com/v/wustl3d-duc1-alignment) |
    | DUC2 | [DUC2 Alignment](https://wustl.box.com/v/wustl3d-duc2-alignment) |
    | CSE3 | [CSE3 Alignment](https://wustl.box.com/v/wustl3d-cse3-alignment) |
    | CSE4 | [CSE4 Alignment](https://wustl.box.com/v/wustl3d-cse4-alignment) |
    | CSE5 | [CSE5 Alignment](https://wustl.box.com/v/wustl3d-cse5-alignment) |



2. Decompress

    `unzip alignment.zip`

3. What you get

    Transformation matrices are provided in two places, in all_transformations.txt formated as such:
    ```
      Scan_ID

      Before general icp:
      Before GICP 4x4 Transformation Matrix

      After general icp:
      After GICP 4x4 Transformation Matrix
    ```
    The `Before GICP 4x4 Transformation Matrix` is the result of our algorithm.  We have then used [GICP](http://www.roboticsproceedings.org/rss05/p21.pdf) to improve the fine alignment.  Note that the "Before" and "After" GICP matrices may be the same in the case that GICP failed

    `Scan_ID` corresponds to the PTX filename.

    Transformation matrices are also in individual files in the transformations folder withe the naming scheme `<Building Code>_trans_<Scan_ID>.txt`

    The `Scan_ID` of known incorrectly placed scans are in `known_incorrect.txt`

4. Usage

    Transformation matrices can be used as such.  Let `p` be a point from a PTX file, `p'` be the coordinate of that point in the aligned coordinate system, and `T` be the 4x4 transformation matrix for that scan.

    Using homogeneous coordinates

    ```
    p' equal-up-to-scale T * p
    ```

    Using [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)

    ```cpp
    p' = (T * p.homogeneous()).eval().hnormalized()
    ```

## Floorplan and Scale

The scale for DUC1 and DUC2 has included in their respective `alignment.zip`'s as `scale.txt`.  The floorplans can be found [here](https://duc.wustl.edu/floor-plan/).


## License

```
This Wustl Indoor RGBD dataset is made available under the Open Database License: http://opendatacommons.org/licenses/odbl/1.0/. Any rights in individual contents of the database are licensed under the Database Contents License: http://opendatacommons.org/licenses/dbcl/1.0/

If you use our data, please cite our paper:

@inproceedings{wijmans17rgbd,
  author = {Erik Wijmans and
            Yasutaka Furukawa},
  title = {Exploiting 2D Floorplan for Building-scale Panorama RGBD Alignment},
  booktitle = {Computer Vision and Pattern Recognition, {CVPR}},
  year = {2017},
  url = {http://cvpr17.wijmans.xyz/CVPR2017-0111.pdf}
}
```

<script src="../javascripts/prism.min.js"></script>