# Point clouds

1. Downloading

    | Building Name (Building Code) |Floor # (Floor Code) | Download Link   | Number of Scans | Area (ft^2)  |% Correct  |
    |:---:|:---:|:---:|:---:|:---:|:---:|
    |Center (DUC)|Floor 1 (DUC1)| [ZIP](https://www.dropbox.com/s/z52vuaf0tszv87q/DUC1.zip?dl=0) | 50 | ~100,000 |  98%   |
    | Center (DUC) |Floor 2 (DUC2)| [ZIP](https://www.dropbox.com/s/s7mh2zwgia427wx/DUC2.zip?dl=0)   | 80 | ~100,000 |  100% |
    | Hall (CSE) |Floor 3 (CSE3) |   [ZIP](https://www.dropbox.com/s/pdiy0ej77iomwgp/CSE3.zip?dl=0)   | 7 | ~8,000  | 85%   |
    | Hall (CSE) |Floor 4 (CSE4) |  [ZIP](https://www.dropbox.com/s/y57nahwszp9440u/CSE4.zip?dl=0)  | 75 | ~19,000 |  93%  |
    | Hall (CSE) |Floor 5 (CSE5) |  [ZIP](https://www.dropbox.com/s/sjqz2uluxtz7ego/CSE5.zip?dl=0)  | 65 | ~20,000 |  66%  |



2. Usage

    Files are named as such:  `<Building Code>_scan_<Scan_ID>.ptx`

    Information on the PTX format can be found [here](http://w3.leica-geosystems.com/kb/?guid=5532D590-114C-43CD-A55F-FE79E5937CB2).
    We only utilize the rows and columns field and transformation matrices are in separate files however.

    The points in the PTX file constitute as column-major panorama with the number of rows and columns specified in the PTX header.

# Transformation Matrices

1. Download

    |Floor Code | Download Link |
    |:---:|:---:|
    | DUC1 | [DUC1 Alignment (Coming Soon)](TODO) |
    | DUC2 | [DUC2 Alignment (Coming Soon)](TODD) |
    | CSE3 | [CSE3 Alignment (Coming Soon)](TODO) |
    | CSE4 | [CSE4 Alignment (Coming Soon)](TODO) |
    | CSE5 | [CSE5 Alignment (Coming Soon)](TODO) |



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
