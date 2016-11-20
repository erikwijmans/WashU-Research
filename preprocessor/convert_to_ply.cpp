#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/StdVector>

#include <boost/progress.hpp>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <scan_gflags.h>
#include <scan_typedefs.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

typedef pcl::PointXYZRGB PointType;

DEFINE_string(ply_folder, "PLY", "folder to put ply files");

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  prependDataPath();
  FLAGS_ply_folder = FLAGS_dataPath + "/" + FLAGS_ply_folder;
  std::cout << FLAGS_ply_folder << std::endl;
  check_dir(FLAGS_ply_folder);

  std::vector<std::string> binary_file_names;
  parseFolder(FLAGS_binaryFolder, binary_file_names);

  boost::progress_display prog(binary_file_names.size());
  for (auto &name : binary_file_names) {
    const std::string number = name.substr(name.find(".") - 3, 3);
    const std::string buildName = name.substr(name.rfind("/") + 1, 3);
    std::vector<scan::PointXYZRGBA> my_cloud;

    std::ifstream in(FLAGS_binaryFolder + "/" + name,
                     std::ios::in | std::ios::binary);

    int columns, rows;
    in.read(reinterpret_cast<char *>(&columns), sizeof(columns));
    in.read(reinterpret_cast<char *>(&rows), sizeof(rows));

    my_cloud.resize(columns * rows);
    for (auto &p : my_cloud)
      p.loadFromFile(in);

    in.close();

    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);

    for (auto &p : my_cloud) {
      auto &point = p.point;
      auto &rgb = p.rgb;
      PointType tmp;
      tmp.x = point[0];
      tmp.y = point[1];
      tmp.z = point[2];
      tmp.r = static_cast<uint8_t>(rgb[0]);
      tmp.g = static_cast<uint8_t>(rgb[1]);
      tmp.b = static_cast<uint8_t>(rgb[2]);
      cloud->push_back(tmp);
    }

    const std::string outname =
        FLAGS_ply_folder + "/" + buildName + "_pointcloud_" + number + ".ply";
    pcl::io::savePLYFileBinary(outname, *cloud);
    ++prog;
  }

  return 0;
}