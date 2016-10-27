/**
  The preprocessor is responsible for ingesting raw PTX files
  and converting them to binary, extracting surface normals,
  creating panoramas, and getting the 4 possible rotations to
  align the scan's dominates directions to the Manhattan world
  assumption (ie walls should be aligned with the X or Y axis)
*/

#include "k4pcs.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/StdVector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <boost/progress.hpp>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/registration/ia_kfpcs.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <dirent.h>

constexpr double voxel_size = 1;

struct result {
  Eigen::Matrix4f mat;
  int i, j;
  float fitness;
};

result k4pcs(const pcl::PointCloud<PointType>::Ptr &source,
             const pcl::PointCloud<NormalType>::Ptr &source_normals,
             const pcl::PointCloud<PointType>::Ptr &target,
             const pcl::PointCloud<NormalType>::Ptr &target_normals);

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  prependDataPath();

  std::vector<std::string> csvFileNames;

  DIR *dir;
  struct dirent *ent;
  if ((dir = opendir(FLAGS_PTXFolder.data())) != NULL) {
    while ((ent = readdir(dir)) != NULL) {
      std::string fileName = ent->d_name;
      if (fileName != ".." && fileName != ".") {
        csvFileNames.push_back(fileName);
      }
    }
    closedir(dir);
  } else {
    perror("");
    return EXIT_FAILURE;
  }

  sort(csvFileNames.begin(), csvFileNames.end(),
       [](const std::string &a, const std::string &b) {
         int numA = std::stoi(a.substr(a.find(".") - 3, 3));
         int numB = std::stoi(b.substr(b.find(".") - 3, 3));
         return numA < numB;
       });
  if (FLAGS_startNumber != -1)
    FLAGS_startIndex = numberToIndex(csvFileNames, FLAGS_startNumber);

  if (FLAGS_numScans == -1)
    FLAGS_numScans = csvFileNames.size() - FLAGS_startIndex;
  if (FLAGS_threads)
    omp_set_num_threads(FLAGS_threads);

  boost::progress_display *show_progress = nullptr;
  if (FLAGS_quietMode)
    show_progress =
        new boost::progress_display(FLAGS_numScans * (FLAGS_numScans - 1) / 2);

  std::vector<result> results;

  for (int i = FLAGS_startIndex; i < FLAGS_numScans + FLAGS_startIndex; ++i) {
    const std::string number =
        csvFileNames[i].substr(csvFileNames[i].find(".") - 3, 3);
    const std::string buildName =
        csvFileNames[i].substr(csvFileNames[i].rfind("/") + 1, 3);
    const std::string csvFileName = FLAGS_PTXFolder + csvFileNames[i];

    const std::string binaryFileName =
        FLAGS_binaryFolder + buildName + "_binary_" + number + ".dat";
    const std::string normalsName =
        FLAGS_normalsFolder + buildName + "_normals_" + number + ".dat";

    std::vector<scan::PointXYZRGBA> pointCloud;
    convertToBinary(csvFileName, binaryFileName, pointCloud);

    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
    createPCLPointCloud(pointCloud, cloud);
    pointCloud.clear();

    pcl::PointCloud<NormalType>::Ptr cloud_normals(
        new pcl::PointCloud<NormalType>);
    pcl::PointCloud<PointType>::Ptr normals_points(
        new pcl::PointCloud<PointType>);
    getNormals(cloud, cloud_normals, normals_points, normalsName);
    cloud->clear();

    for (int j = i + 1; j < FLAGS_numScans + FLAGS_startIndex; ++j) {
      const std::string number_target =
          csvFileNames[j].substr(csvFileNames[j].find(".") - 3, 3);
      const std::string buildName_target =
          csvFileNames[j].substr(csvFileNames[j].rfind("/") + 1, 3);
      const std::string csvFileName_target = FLAGS_PTXFolder + csvFileNames[j];

      const std::string binaryFileName_target = FLAGS_binaryFolder +
                                                buildName_target + "_binary_" +
                                                number_target + ".dat";
      const std::string normalsName_target = FLAGS_normalsFolder +
                                             buildName_target + "_normals_" +
                                             number_target + ".dat";

      std::vector<scan::PointXYZRGBA> pointCloud_target;
      convertToBinary(csvFileName, binaryFileName_target, pointCloud_target);

      pcl::PointCloud<PointType>::Ptr cloud_target(
          new pcl::PointCloud<PointType>);
      createPCLPointCloud(pointCloud_target, cloud_target);
      pointCloud_target.clear();

      pcl::PointCloud<NormalType>::Ptr cloud_normals_target(
          new pcl::PointCloud<NormalType>);
      pcl::PointCloud<PointType>::Ptr normals_points_target(
          new pcl::PointCloud<PointType>);
      getNormals(cloud_target, cloud_normals_target, normals_points_target,
                 normalsName_target);
      cloud_target->clear();

      results.emplace_back(k4pcs(normals_points, cloud_normals,
                                 normals_points_target, cloud_normals_target));
      results.back().i = i;
      results.back().j = j;

      if (show_progress)
        ++(*show_progress);
    }
  }

  std::ofstream out("out.dat", std::ios::out | std::ios::binary);
  int size = results.size();
  out.write(reinterpret_cast<const char *>(&size), sizeof(size));
  for (auto &r : results) {
    out.write(reinterpret_cast<const char *>(&r.i), sizeof(int));
    out.write(reinterpret_cast<const char *>(&r.j), sizeof(int));
    out.write(reinterpret_cast<const char *>(&r.fitness), sizeof(float));
    out.write(reinterpret_cast<const char *>(r.mat.data()), 16 * sizeof(float));
  }
  out.close();
  if (show_progress)
    delete show_progress;

  std::cout << "Leaving" << std::endl;
  return 0;
}

void convertToBinary(const std::string &fileNameIn, const std::string &outName,
                     std::vector<scan::PointXYZRGBA> &pointCloud) {

  if (!FLAGS_quietMode)
    std::cout << outName << std::endl;

  std::ifstream in(outName, std::ios::in | std::ios::binary);

  if (!in.is_open() || FLAGS_redo) {
    in.close();
    std::ifstream scanFile(fileNameIn, std::ios::in);
    int columns, rows;
    scanFile >> columns >> rows;
    std::ofstream out(outName, std::ios::out | std::ios::binary);
    out.write(reinterpret_cast<const char *>(&columns), sizeof(columns));
    out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));

    std::string line;
    if (!FLAGS_quietMode)
      std::cout << rows << "   " << columns << std::endl;
    pointCloud.reserve(columns * rows);
    for (int i = 0; i < 9; ++i) {
      getline(scanFile, line);
    }

    for (int k = 0; k < columns * rows; ++k) {
      scan::PointXYZRGBA tmp;
      scanFile >> tmp.point[0] >> tmp.point[1] >> tmp.point[2];

      scanFile >> tmp.intensity;

      double itmp[3];
      for (int j = 0; j < 3; ++j) {
        scanFile >> itmp[j];
        tmp.rgb[j] = cv::saturate_cast<uchar>(itmp[j]);
      }

      tmp.writeToFile(out);
      pointCloud.push_back(tmp);
    }

    out.close();

  } else {
    int columns, rows;
    in.read(reinterpret_cast<char *>(&columns), sizeof(columns));
    in.read(reinterpret_cast<char *>(&rows), sizeof(rows));

    pointCloud.resize(columns * rows);
    for (auto &p : pointCloud)
      p.loadFromFile(in);

    in.close();
  }
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
    if (p.intensity < 0.01)
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

  std::cout << cloud->size() << std::endl;
}

bool reloadNormals(pcl::PointCloud<NormalType>::Ptr &cloud_normals,
                   pcl::PointCloud<PointType>::Ptr &normals_points,
                   const std::string &outName) {
  std::ifstream in(outName, std::ios::in | std::ios::binary);
  if (!in.is_open())
    return false;

  size_t size;
  in.read(reinterpret_cast<char *>(&size), sizeof(size_t));

  cloud_normals->resize(size);
  normals_points->resize(size);

  for (int i = 0; i < size; ++i) {
    in.read(reinterpret_cast<char *>(&cloud_normals->at(i)),
            sizeof(NormalType));
    in.read(reinterpret_cast<char *>(&normals_points->at(i)),
            sizeof(PointType));
  }

  return true;
}

void getNormals(const pcl::PointCloud<PointType>::Ptr &cloud,
                pcl::PointCloud<NormalType>::Ptr &cloud_normals,
                pcl::PointCloud<PointType>::Ptr &normals_points,
                const std::string &outName) {

  /*   if (!FLAGS_redo && reloadNormals(cloud_normals, normals_points, outName))
       return;*/

  pcl::PointCloud<PointType>::Ptr filtered_cloud(
      new pcl::PointCloud<PointType>);

  pcl::UniformSampling<PointType> uniform_sampling;
  uniform_sampling.setInputCloud(cloud);
  // 85 mm
  uniform_sampling.setRadiusSearch(voxel_size);
  uniform_sampling.filter(*filtered_cloud);

  /*pcl::visualization::PCLVisualizer::Ptr viewer = rgbVis(filtered_cloud);
  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
    boost::this_thread::sleep(boost::posix_time::microseconds(100000));
  }*/

  pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
  pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
  norm_est.setInputCloud(filtered_cloud);
  norm_est.setSearchSurface(cloud);
  norm_est.setSearchMethod(tree);
  norm_est.setRadiusSearch(0.03f);
  norm_est.compute(*cloud_normals);

  std::vector<int> indices;
  pcl::removeNaNNormalsFromPointCloud(*cloud_normals, *cloud_normals, indices);
  pcl::copyPointCloud(*filtered_cloud, indices, *normals_points);

  std::cout << normals_points->size() << "  " << cloud_normals->size()
            << std::endl;
}

void saveNormals(const pcl::PointCloud<NormalType>::Ptr &cloud_normals,
                 pcl::PointCloud<PointType>::Ptr &normals_points,
                 const std::string &outName) {
  assert(cloud_normals->size() == normals_points->size());
  std::ofstream out(outName, std::ios::out | std::ios::binary);

  size_t size = cloud_normals->points.size();
  out.write(reinterpret_cast<const char *>(&size), sizeof(size_t));

  for (int i = 0; i < size; ++i) {
    out.write(reinterpret_cast<const char *>(&cloud_normals->at(i)),
              sizeof(NormalType));
    out.write(reinterpret_cast<const char *>(&normals_points->at(i)),
              sizeof(PointType));
  }

  out.close();
}

void boundingBox(const std::vector<scan::PointXYZRGBA> &points,
                 Eigen::Vector3f &pointMin, Eigen::Vector3f &pointMax) {
  Eigen::Vector3f average = Eigen::Vector3f::Zero();
  Eigen::Vector3f sigma = Eigen::Vector3f::Zero();
  for (auto &point : points)
    average += point.point;

  average /= points.size();

  for (auto &point : points) {
    Eigen::Vector3f tmp = point.point - average;
    sigma += tmp.cwiseProduct(tmp);
  }

  sigma /= points.size() - 1;
  for (int i = 0; i < 3; ++i)
    sigma[i] = sqrt(sigma[i]);

  Eigen::Vector3f range(6, 6, 6);
  Eigen::Vector3f delta = 1.1 * sigma.cwiseProduct(range);

  pointMin = average - delta / 2.0;
  pointMax = average + delta / 2.0;
}

result k4pcs(const pcl::PointCloud<PointType>::Ptr &source,
             const pcl::PointCloud<NormalType>::Ptr &source_normals,
             const pcl::PointCloud<PointType>::Ptr &target,
             const pcl::PointCloud<NormalType>::Ptr &target_normals) {

  std::cout << "Entering k4pcs" << std::endl;
  std::cout << source->size() << "  " << source_normals->size() << std::endl;
  std::cout << target->size() << "  " << target_normals->size() << std::endl;

  pcl::search::KdTree<PointType>::Ptr source_tree(
      new pcl::search::KdTree<PointType>),
      target_tree(new pcl::search::KdTree<PointType>);

  pcl::registration::KFPCSInitialAlignment<PointType, PointType, NormalType,
                                           float>
      aligner;

  aligner.setDelta(voxel_size, false);

  aligner.setInputSource(source);
  aligner.setInputTarget(target);

  aligner.setSearchMethodSource(source_tree);
  aligner.setSearchMethodTarget(target_tree);

  aligner.setSourceNormals(source_normals);
  aligner.setTargetNormals(target_normals);

  aligner.setNumberOfThreads(8);

  pcl::PointCloud<PointType>::Ptr output(new pcl::PointCloud<PointType>);
  aligner.align(*output);

  std::cout << aligner.getFinalTransformation() << std::endl;

  result res;

  res.mat = aligner.getFinalTransformation();
  res.fitness = aligner.getFitnessScore();
  return res;
}