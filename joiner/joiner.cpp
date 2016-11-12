#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/StdVector>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/keyboard_event.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <scan_gflags.h>
#include <scan_typedefs.hpp>

typedef pcl::PointXYZRGB PointType;

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

void createPCLPointCloud(const std::vector<scan::PointXYZRGBA> &points,
                         pcl::PointCloud<PointType>::Ptr &cloud,
                         const Eigen::Matrix3d &rotMat,
                         const Eigen::Vector3d &trans);

constexpr double targetNumPoints = 30e6;

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  prependDataPath();

  std::vector<std::string> binaryFileNames;
  parseFolder(FLAGS_binaryFolder, binaryFileNames);
  const std::string buildName = binaryFileNames[0].substr(0, 3);
  const std::string cloudName = FLAGS_outputV2 + buildName + "_pointCloud.ply";
  std::cout << cloudName << std::endl;

  pcl::PointCloud<PointType>::Ptr output_cloud(new pcl::PointCloud<PointType>);

  if (FLAGS_redo || !fexists(cloudName)) {
    const std::string fileName = FLAGS_outputV2 + "final_0.dat";
    if (!fexists(fileName)) {
      std::cout << "Could not find: " << fileName << std::endl;
      exit(1);
    }
    std::ifstream in(FLAGS_outputV2 + "final_0.dat",
                     std::ios::in | std::ios::binary);

    int num;
    in.read(reinterpret_cast<char *>(&num), sizeof(num));
    std::cout << num << std::endl;

    std::vector<Eigen::Matrix3d> rotMats(num);
    std::vector<Eigen::Vector3d> translations(num);

    for (int i = 0; i < num; ++i) {
      in.read(reinterpret_cast<char *>(rotMats[i].data()),
              sizeof(Eigen::Matrix3d));
      in.read(reinterpret_cast<char *>(translations[i].data()),
              sizeof(Eigen::Vector3d));
    }
    in.close();
    assert(num <= binaryFileNames.size());
    double subSampleSize = 0.0085;

    if (FLAGS_numScans != -1)
      num = FLAGS_numScans;

    for (int k = FLAGS_startIndex;
         k < std::min((int)rotMats.size(), FLAGS_startIndex + num); ++k) {
      std::cout << "Enter: " << binaryFileNames[k] << std::endl;
      if (rotMats[k] == Eigen::Matrix3d::Zero())
        continue;
      in.open(FLAGS_binaryFolder + binaryFileNames[k],
              std::ios::in | std::ios::binary);
      int rows, cols;
      in.read(reinterpret_cast<char *>(&rows), sizeof(rows));
      in.read(reinterpret_cast<char *>(&cols), sizeof(cols));
      std::vector<scan::PointXYZRGBA> points(rows * cols);
      for (auto &p : points)
        p.loadFromFile(in);
      in.close();

      createPCLPointCloud(points, output_cloud, rotMats[k].inverse(),
                          translations[k]);

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
      std::cout << "Leaving: " << output_cloud->size() << std::endl;
    }
    std::cout << "Saving" << std::endl;
    pcl::io::savePLYFileBinary(cloudName, *output_cloud);
  } else
    pcl::io::loadPLYFile(cloudName, *output_cloud);

  if (FLAGS_visulization) {
    pcl::visualization::PCLVisualizer::Ptr viewer = rgbVis(output_cloud);
    while (!viewer->wasStopped()) {
      viewer->spinOnce(100);
      boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
  }
}

void boundingBox(const std::vector<scan::PointXYZRGBA> &points,
                 Eigen::Vector3f &pointMin, Eigen::Vector3f &pointMax) {
  Eigen::Vector3f average = Eigen::Vector3f::Zero();
  Eigen::Vector3f sigma = Eigen::Vector3f::Zero();

  for (auto &point : points)
    average += point.point;
  average /= points.size();

  for (auto &point : points)
    for (int i = 0; i < 3; ++i)
      sigma[i] += (point.point[i] - average[i]) * (point.point[i] - average[i]);

  sigma /= points.size() - 1;
  for (int i = 0; i < 3; ++i)
    sigma[i] = sqrt(sigma[i]);

  Eigen::Vector3f range(13, 13, 8);
  Eigen::Vector3f delta;
  for (int i = 0; i < delta.size(); ++i)
    delta[i] = 1.1 * range[i] * sigma[i];

  pointMin = average - delta / 2.0;
  pointMax = average + delta / 2.0;
}

void createPCLPointCloud(const std::vector<scan::PointXYZRGBA> &points,
                         pcl::PointCloud<PointType>::Ptr &cloud,
                         const Eigen::Matrix3d &rotMat,
                         const Eigen::Vector3d &trans) {
  Eigen::Vector3f pointMin, pointMax;
  boundingBox(points, pointMin, pointMax);
  int count = 0;
  for (auto &p : points) {
    bool in = true;
    for (int i = 0; i < 3; ++i)
      if (p.point[i] < pointMin[i] || p.point[i] > pointMax[i])
        in = false;

    if (!in)
      continue;

    Eigen::Vector3d point = p.point.cast<double>();
    point[1] *= -1;
    point = rotMat * point;
    point += trans;
    point[1] *= -1;
    auto &rgb = p.rgb;
    PointType tmp;
    tmp.x = point[0];
    tmp.y = point[1];
    tmp.z = point[2];
    tmp.r = cv::saturate_cast<uint8_t>(rgb[0]);
    tmp.g = cv::saturate_cast<uint8_t>(rgb[1]);
    tmp.b = cv::saturate_cast<uint8_t>(rgb[2]);
    cloud->push_back(tmp);
  }
}
