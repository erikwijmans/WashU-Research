#include <Eigen/Geometry>
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/StdVector>

#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/visualization/keyboard_event.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <gflags/gflags.h>

using namespace fmt::literals;

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

typedef pcl::PointXYZRGB PointType;
typedef pcl::PointNormal NormalType;

pcl::PointCloud<PointType>::Ptr createPCLPointCloud(const fs::path &d_file,
                                                    const fs::path &t_file);

pcl::visualization::PCLVisualizer::Ptr
rgbVis(pcl::PointCloud<PointType>::ConstPtr cloud) {
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
      new pcl::visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor(0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<PointType> rgb(cloud);
  viewer->addPointCloud<PointType>(cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();
  viewer->registerKeyboardCallback(
      [&, viewer](const pcl::visualization::KeyboardEvent &kb) {
        pcl::visualization::Camera c;
        viewer->getCameraParameters(c);

        Eigen::Map<Eigen::Vector3d> focal(c.focal);
        Eigen::Map<Eigen::Vector3d> pos(c.pos);
        Eigen::Vector3d view = (focal - pos).normalized();
        view[2] = 0;
        view.normalize();
        const double incMag = (kb.isShiftPressed() ? 1.0 : 0.5);

        Eigen::Vector3d incDir = incMag * view;
        Eigen::Vector3d perpInc =
            incMag * Eigen::Vector3d(-view[1], view[0], view[2]);

        if (kb.getKeySym() == "Up") {
          if (!kb.isCtrlPressed())
            focal += incDir;
          pos += incDir;
        }
        if (kb.getKeySym() == "Down") {
          if (!kb.isCtrlPressed())
            focal -= incDir;
          pos -= incDir;
        }
        if (kb.getKeySym() == "Left") {
          if (!kb.isCtrlPressed())
            focal += perpInc;
          pos += perpInc;
        }
        if (kb.getKeySym() == "Right") {
          if (!kb.isCtrlPressed())
            focal -= perpInc;
          pos -= perpInc;
        }
        viewer->setCameraParameters(c);
      });
  viewer->setCameraPosition(1, 0, 0, -1, 0, 0, 0, 0, 1);
  return (viewer);
}

constexpr double targetNumPoints = 100e6;
constexpr double startScale = 0.005;

int main(int argc, char **argv) {
  fs::path transform_folder("transformations"), data_folder("data");

  const fs::path cloudName("aligned_pointCloud.ply");
  pcl::PointCloud<PointType>::Ptr output_cloud(new pcl::PointCloud<PointType>);

  if (!fs::exists(cloudName)) {

    std::vector<fs::path> data_files_names, transform_files;
    for (auto &file : fs::directory_iterator(transform_folder)) {
      transform_files.push_back(file);
    }
    std::sort(transform_files.begin(), transform_files.end(),
              [](fs::path &a, fs::path &b) {
                auto astr = a.filename().string(), bstr = b.filename().string();
                return astr < bstr;
              });

    for (auto &file : fs::directory_iterator(data_folder)) {
      data_files_names.push_back(file);
    }
    std::sort(data_files_names.begin(), data_files_names.end(),
              [](fs::path &a, fs::path &b) {
                auto astr = a.filename().string(), bstr = b.filename().string();
                return astr < bstr;
              });

    double subSampleSize = startScale;

    for (int k = 0; k < transform_files.size(); ++k) {
      auto &t_file = transform_files[k];
      auto &d_file = data_files_names[k];
      fmt::print("Enter: {}\n", d_file);

      auto current_cloud = createPCLPointCloud(d_file, t_file);

      output_cloud->insert(output_cloud->end(), current_cloud->begin(),
                           current_cloud->end());

      current_cloud = nullptr;

      pcl::UniformSampling<PointType> uniform_sampling;
      uniform_sampling.setInputCloud(output_cloud);
      output_cloud =
          pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
      uniform_sampling.setRadiusSearch(subSampleSize);
      uniform_sampling.filter(*output_cloud);

      if (output_cloud->size() > targetNumPoints) {
        subSampleSize *= std::sqrt(output_cloud->size() / targetNumPoints);

        output_cloud =
            pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
        uniform_sampling.setRadiusSearch(subSampleSize);
        uniform_sampling.filter(*output_cloud);
      }

#if 0
      pcl::visualization::PCLVisualizer::Ptr viewer = rgbVis(output_cloud);
      while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
      }
#endif

      fmt::print("Leaving: {}\n", output_cloud->size());
    }

    fmt::print("Final sample size: {}\n", subSampleSize);

    pcl::StatisticalOutlierRemoval<PointType> sor;
    sor.setInputCloud(output_cloud);
    sor.setMeanK(50);
    sor.setStddevMulThresh(2.0);
    output_cloud =
        pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    sor.filter(*output_cloud);

    std::cout << "Saving" << std::endl;
    pcl::io::savePLYFileBinary(cloudName.string(), *output_cloud);
  } else
    pcl::io::loadPLYFile(cloudName.string(), *output_cloud);

  if (true) {
    pcl::UniformSampling<PointType> uniform_sampling;
    uniform_sampling.setInputCloud(output_cloud);
    pcl::PointCloud<PointType>::Ptr ss(new pcl::PointCloud<PointType>);
    uniform_sampling.setRadiusSearch(0.05);
    uniform_sampling.filter(*ss);
    pcl::visualization::PCLVisualizer::Ptr viewer = rgbVis(ss);
    while (!viewer->wasStopped()) {
      viewer->spinOnce(100);
      boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
  }
}

pcl::PointCloud<PointType>::Ptr createPCLPointCloud(const fs::path &d_file,
                                                    const fs::path &t_file) {
  pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);

  Eigen::Matrix4d T;

  std::ifstream t_in(t_file.string(), std::ios::in);
  {
    std::string tmp;
    for (int i = 0; i < 3; ++i) {
      t_in >> tmp;
    }
    double t_double;
    for (int i = 0; i < 16; ++i) {
      t_in >> t_double;
    }
    for (int i = 0; i < 3; ++i) {
      t_in >> tmp;
    }
  }

  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 4; ++i) {
      t_in >> T(j, i);
    }
  }
  fmt::print("Transformation matrix: \n"
             "{}\n\n",
             T);

  std::ifstream ptx_in(d_file.string(), std::ios::in);
  int columns, rows;
  ptx_in >> columns >> rows;
  cloud->reserve(rows * columns);
  {
    std::string line;

    for (int i = 0; i < 9; ++i) {
      std::getline(ptx_in, line);
    }
  }

  for (int i = 0; i < rows * columns; ++i) {
    PointType pt;
    ptx_in >> pt.x >> pt.y >> pt.z;

    double d_tmp;
    ptx_in >> d_tmp;

    double rgb_tmp[3];
    for (int k = 0; k < 3; ++k) {
      ptx_in >> rgb_tmp[k];
    }
    pt.r =
        static_cast<unsigned char>(std::min(std::max(rgb_tmp[0], 0.0), 255.0));
    pt.g =
        static_cast<unsigned char>(std::min(std::max(rgb_tmp[1], 0.0), 255.0));
    pt.b =
        static_cast<unsigned char>(std::min(std::max(rgb_tmp[2], 0.0), 255.0));

    pt.getVector3fMap() = (T * pt.getVector3fMap().cast<double>().homogeneous())
                              .eval()
                              .hnormalized()
                              .cast<float>();

    cloud->push_back(pt);
  }

  /* pcl::visualization::PCLVisualizer::Ptr viewer = rgbVis(cloud);
  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
    boost::this_thread::sleep(boost::posix_time::microseconds(100000));
  } */

  return cloud;
}
