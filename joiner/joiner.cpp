#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/StdVector>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <scan_gflags.h>
#include <scan_typedefs.hpp>

void createPCLPointCloud(const std::vector<scan::PointXYZRGBA> &points,
                         pcl::PointCloud<PointType>::Ptr &cloud);

typedef pcl::PointXYZRGB PointType;

constexpr double targetNumPoints = 20e6;

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  prependDataPath();

  std::vector<std::string> binaryFileNames;
  parseFolder(FLAGS_binaryFolder, binaryFileNames);

  std::ifstream in(FLAGS_outputV2 + "final.dat",
                   std::ios::in | std::ios::binary);
  int num;
  in.read(reinterpret_cast<char *>(&num), sizeof(num));
  assert(num == binaryFileNames.size());
  std::vector<Eigen::Matrix3d> rotMats(num);
  std::vector<Eigen::Vector3d> translations(num);

  for (int i = 0; i < num; ++i) {
    in.read(reinterpret_cast<char *>(rotMats[i].data()),
            sizeof(Eigen::Matrix3d));
    in.read(reinterpret_cast<char *>(translations[i].data()),
            sizeof(Eigen::Vector3d));
  }
  in.close();
  double subSampleSize = 0.0085;
  pcl::PointCloud<PointType>::Ptr output_cloud(new pcl::PointCloud<PointType>);
  pcl::UniformSampling<PointType> uniform_sampling;
  for (int k = 0; k < num; ++k) {
    in.open(FLAGS_binaryFolder + binaryFileNames[k],
            std::ios::in | std::ios::binary);
    int rows, cols;
    in.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char *>(&cols), sizeof(cols));
    std::vector<scan::PointXYZRGBA> points(rows * cols);
    for (auto &p : points) {
      p.loadFromFile(in);
      p.point[1] *= -1;
      p.point = rotMats[k].inverse() * p.point;
      p.point += translations[k];
      p.point[1] *= -1;
    }

    pcl::PointCloud<PointType>::Ptr current_cloud(
        new pcl::PointCloud<PointType>);
    createPCLPointCloud(points, current_cloud);

    current_cloud->insert(current_cloud->end(), output_cloud->begin(),
                          output_cloud->end());

    output_cloud->clear();
    pcl::PointCloud<int> sampled_indices;
    uniform_sampling.setInputCloud(current_cloud);
    uniform_sampling.setRadiusSearch(subSampleSize);
    uniform_sampling.compute(sampled_indices);

    pcl::copyPointCloud(*current_cloud, sampled_indices.points, *output_cloud);

    if (output_cloud->size() > targetNumPoints) {
      subSampleSize *= output_cloud->size() / targetNumPoints;

      output_cloud->clear();
      pcl::PointCloud<int> sampled_indices;
      uniform_sampling.setInputCloud(current_cloud);
      uniform_sampling.setRadiusSearch(subSampleSize);
      uniform_sampling.compute(sampled_indices);

      pcl::copyPointCloud(*current_cloud, sampled_indices.points,
                          *output_cloud);
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

  Eigen::Vector3f range(10, 10, 6);
  Eigen::Vector3f delta;
  for (int i = 0; i < delta.size(); ++i)
    delta[i] = 1.1 * range[i] * sigma[i];

  pointMin = average - delta / 2.0;
  pointMax = average + delta / 2.0;
}

void createPCLPointCloud(const std::vector<scan::PointXYZRGBA> &points,
                         pcl::PointCloud<PointType>::Ptr &cloud) {
  Eigen::Vector3f pointMin, pointMax;
  boundingBox(points, pointMin, pointMax);

  for (auto &p : points) {
    bool in = true;
    for (int i = 0; i < 3; ++i)
      if (p.point[i] < pointMin[i] || p.point[i] > pointMax[i])
        in = false;

    if (!in)
      continue;
    if (p.intensity < 0.2)
      continue;

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
}
