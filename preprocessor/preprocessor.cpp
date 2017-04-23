/**
  The preprocessor is responsible for ingesting raw PTX files
  and converting them to binary, extracting surface normals,
  creating panoramas, and getting the 4 possible rotations to
  align the scan's dominates directions to the Manhattan world
  assumption (ie walls should be aligned with the X or Y axis)
*/
#include "preprocessor.h"
#include "HashVoxel.hpp"
#include "getRotations.h"

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

#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/uniform_sampling.h>

static int PTXrows, PTXcols;

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  prependDataPath();

  auto csvFileNames = utils::parse_folder(FLAGS_PTXFolder);

  sort(csvFileNames.begin(), csvFileNames.end(),
       [](const fs::path &_a, const fs::path &_b) {
         auto a = _a.string(), b = _b.string();
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

  utils::progress_display show_progress(FLAGS_numScans * 5);

  for (int i = FLAGS_startIndex; i < FLAGS_numScans + FLAGS_startIndex; ++i) {
    const auto &csvFileName = csvFileNames[i];
    auto[building_name, scan_id] = parse_name(csvFileName);

    const fs::path binaryFileName =
        fs::path(FLAGS_binaryFolder) /
        "{}_binary_{}.dat"_format(building_name, scan_id);
    const fs::path normalsName =
        fs::path(FLAGS_normalsFolder) /
        "{}_normals_{}.dat"_format(building_name, scan_id);
    const fs::path dataName = fs::path(FLAGS_panoFolder) / "data" /
                              "{}_data_{}.dat"_format(building_name, scan_id);
    const fs::path rotName =
        fs::path(FLAGS_rotFolder) /
        "{}_rotations_{}.dat"_format(building_name, scan_id);
    const fs::path panoName =
        fs::path(FLAGS_panoFolder) / "images" /
        "{}_panorama_{}.png"_format(building_name, scan_id);
    const fs::path doorName = fs::path(FLAGS_doorsFolder) / "pointcloud" /
                              "{}_doors_{}.dat"_format(building_name, scan_id);
    if (FLAGS_redo ||
        !(fs::exists(binaryFileName) && fs::exists(normalsName) &&
          fs::exists(dataName) && fs::exists(rotName) && fs::exists(panoName) &&
          fs::exists(doorName))) {
      std::vector<scan::PointXYZRGBA> pointCloud;
      convertToBinary(csvFileName, binaryFileName, pointCloud);

      ++show_progress;

      pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
      createPCLPointCloud(pointCloud, cloud);

      ++show_progress;

      pcl::PointCloud<NormalType>::Ptr cloud_normals(
          new pcl::PointCloud<NormalType>);
      pcl::PointCloud<PointType>::Ptr normals_points(
          new pcl::PointCloud<PointType>);
      getNormals(cloud, cloud_normals, normals_points, normalsName);

      ++show_progress;

      Eigen::Vector3d M1, M2, M3;
      getRotations(cloud_normals, rotName, M1, M2, M3);

      findDoors(normals_points, M1, M2, M3, doorName);

      ++show_progress;

      createPanorama(pointCloud, cloud_normals, normals_points, panoName,
                     dataName);

      ++show_progress;
    } else
      show_progress += 5;
  }

  return 0;
}

static double ransacZ(const std::vector<double> &Z) {
  return utils::ransac(
      Z, []() -> double { return 0.0; },
      [](double z, double zest) { return std::abs(z - zest) < 0.03; });
}

static Eigen::VectorXd getZPlanes(const std::vector<double> &z) {
  std::vector<double> zCoords(z.begin(), z.end());
  Eigen::VectorXd domZs = Eigen::VectorXd::Zero(20);
  int count = 0;
  do {
    double z0 = ransacZ(zCoords);

    zCoords.erase(
        std::remove_if(zCoords,
                       [&z0](auto &z) { return std::abs(z - z0) < 0.05; }),
        zCoords.end());

    domZs[count++] = z0;
  } while (domZs.minCoeff() >= -1.5 || count < 2);
  return domZs;
}

void convertToBinary(const fs::path &fileNameIn, const fs::path &outName,
                     std::vector<scan::PointXYZRGBA> &pointCloud) {

  LOG(INFO) << "Working on: " << outName << std::endl;

  if (!fs::exists(outName) || FLAGS_redo) {
    std::ifstream scanFile(fileNameIn.string(), std::ios::in);
    CHECK(scanFile.is_open()) << "Could not open " << fileNameIn << std::endl;
    int columns, rows;
    scanFile >> columns >> rows;
    std::ofstream out(outName.string(), std::ios::out | std::ios::binary);
    out.write(reinterpret_cast<const char *>(&columns), sizeof(columns));
    out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    PTXcols = columns;
    PTXrows = rows;
    std::string line;

    LOG(INFO) << rows << "   " << columns << std::endl;
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
    std::ifstream in(outName.string(), std::ios::in | std::ios::binary);
    int columns, rows;
    in.read(reinterpret_cast<char *>(&columns), sizeof(columns));
    in.read(reinterpret_cast<char *>(&rows), sizeof(rows));

    PTXcols = columns;
    PTXrows = rows;

    pointCloud.resize(columns * rows);
    for (auto &p : pointCloud)
      p.loadFromFile(in);

    in.close();
  }
}

static Eigen::Vector3d cartesianToPolar(const Eigen::Vector3d &coords) {
  double r = coords.norm();
  double theta = atan2(coords[1], coords[0]);
  double phi = acos(coords[2] / r);
  return Eigen::Vector3d(r, theta, phi);
}

static Eigen::Vector3d pointCloudToPanorama(const Eigen::Vector3f &worldCoord,
                                            const cv::Size &resolution) {
  auto polar = cartesianToPolar(worldCoord.cast<double>());
  int x = (polar[1] / PI + 1.0) * (resolution.width - 1) / 2.0;
  int y = polar[2] * (resolution.height - 1) / maxPhi;
  return Eigen::Vector3d(x, y, polar[0]);
}

template <class T1, class T2> void fillGaps(T1 &mat, T2 &mask) {
  assert(mat.rows() == mask.rows());
  assert(mat.cols() == mask.cols());
  int count = 0;
  constexpr int limit = 15;
  decltype(mat.data()) current = nullptr;

  for (int j = 0, loop = 1; j < mat.rows() && loop; ++j) {
    for (int i = 0; i < mat.cols() && loop; ++i) {
      if (mask(j, i)) {
        current = &mat(j, i);
        loop = 0;
      }
    }
  }

  count = 0;
  for (int j = 0; j < mat.rows(); ++j) {
    for (int i = 0; i < mat.cols(); ++i) {
      if (mask(j, i)) {
        current = &mat(j, i);
        count = 0;
      } else if (count++ < limit) {
        mat(j, i) = *current;
      }
    }
  }

  for (int j = mat.rows() - 1, loop = 1; j >= 0 && loop; --j) {
    for (int i = mat.cols() - 1; i >= 0 && loop; --i) {
      if (mask(j, i)) {
        current = &mat(j, i);
        loop = 0;
      }
    }
  }

  count = 0;
  for (int j = mat.rows() - 1; j >= 0; --j) {
    for (int i = mat.cols() - 1; i >= 0; --i) {
      if (mask(j, i)) {
        current = &mat(j, i);
        count = 0;
      } else if (count++ < limit) {
        mat(j, i) = *current;
      }
    }
  }

  for (int i = 0, loop = 1; i < mat.cols() && loop; ++i) {
    for (int j = 0; j < mat.rows() && loop; ++j) {
      if (mask(j, i)) {
        current = &mat(j, i);
        loop = 0;
      }
    }
  }

  count = 0;
  for (int i = 0; i < mat.cols(); ++i) {
    for (int j = 0; j < mat.rows(); ++j) {
      if (mask(j, i)) {
        current = &mat(j, i);
        count = 0;
      } else if (count++ < limit) {
        mat(j, i) = *current;
      }
    }
  }

  for (int i = mat.cols() - 1, loop = 1; i >= 0 && loop; --i) {
    for (int j = mat.rows() - 1; j >= 0 && loop; --j) {
      if (mask(j, i)) {
        current = &mat(j, i);
        loop = 0;
      }
    }
  }

  count = 0;
  for (int i = mat.cols() - 1; i >= 0; --i) {
    for (int j = mat.rows() - 1; j >= 0; --j) {
      if (mask(j, i)) {
        current = &mat(j, i);
        count = 0;
      } else if (count++ < limit) {
        mat(j, i) = *current;
      }
    }
  }
}

void createPanorama(const std::vector<scan::PointXYZRGBA> &pointCloud,
                    pcl::PointCloud<NormalType>::Ptr &cloud_normals,
                    pcl::PointCloud<PointType>::Ptr &normals_points,
                    const fs::path &panoName, const fs::path &dataName) {
  if (!FLAGS_redo && fs::exists(panoName) && fs::exists(dataName))
    return;

  cv::Mat trackingPanorama(PTXrows, PTXcols, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat_<cv::Vec3b> _trackingPanorama = trackingPanorama;

  cv::Mat PTXPanorama(PTXrows, PTXcols, CV_8UC3);
  cv::Mat_<cv::Vec3b> _PTXPanorama = PTXPanorama;
  Eigen::RowMatrixXd rMap = Eigen::RowMatrixXd::Zero(PTXrows, PTXcols);

  Eigen::RowMatrixXb hasNormal = Eigen::RowMatrixXb::Zero(PTXrows, PTXcols);
  Eigen::ArrayXV3f surfaceNormals(PTXrows, PTXcols);

  int row = PTXrows - 1;
  int col = 0.995 * (PTXcols - 1) / 2.0;
  Eigen::RowMatrixXb touched = Eigen::RowMatrixXb::Zero(PTXrows, PTXcols);
  std::vector<double> zCoords;
  for (auto &element : pointCloud) {
    assert(row >= 0 && row < PTXPanorama.rows);
    assert(col >= 0 && col < PTXPanorama.cols);
    _PTXPanorama(row, col)[0] = element.rgb[2];
    _PTXPanorama(row, col)[1] = element.rgb[1];
    _PTXPanorama(row, col)[2] = element.rgb[0];

    zCoords.push_back(element.point[2]);

    if (row == 0) {
      row = PTXrows - 1;
      col = col == 0 ? PTXcols - 1 : col - 1;
    } else
      --row;

    auto panoCoord =
        pointCloudToPanorama(element.point, trackingPanorama.size());
    const double r = panoCoord[2];
    const int trackedRow = panoCoord[1];
    const int trackedCol = panoCoord[0];

    if (trackedRow < 0 || trackedRow >= trackingPanorama.rows)
      continue;
    if (trackedCol < 0 || trackedCol >= trackingPanorama.cols)
      continue;
    if (!(element.rgb[2] || element.rgb[1] || element.rgb[0]))
      continue;

    _trackingPanorama(trackedRow, trackedCol)[0] = element.rgb[2];
    _trackingPanorama(trackedRow, trackedCol)[1] = element.rgb[1];
    _trackingPanorama(trackedRow, trackedCol)[2] = element.rgb[0];

    touched(trackedRow, trackedCol) = 1;
    rMap(trackedRow, trackedCol) = r;
  }

  for (int i = 0; i < cloud_normals->size(); ++i) {
    auto &p = normals_points->at(i);
    auto &n = cloud_normals->at(i);
    Eigen::Vector3f coord(p.x, p.y, p.z);
    auto panoCoord = pointCloudToPanorama(coord, trackingPanorama.size());
    const int trackedRow = panoCoord[1];
    const int trackedCol = panoCoord[0];
    surfaceNormals(trackedRow, trackedCol) =
        Eigen::Vector3f(n.normal_x, n.normal_y, n.normal_z);
    hasNormal(trackedRow, trackedCol) = 1;
  }

  for (int j = 0; j < PTXrows; ++j) {
    for (int i = 0; i < PTXcols; ++i) {
      if (!hasNormal(j, i))
        surfaceNormals(j, i) = Eigen::Vector3f::Zero();
    }
  }
  fillGaps(surfaceNormals, hasNormal);
  fillGaps(rMap, rMap);

  const double scale = pow(2, -6.0 / 2);
  cv::Mat scaledTracking, scaledPTX;
  cv::resize(trackingPanorama, scaledTracking, cv::Size(), scale, scale,
             CV_INTER_AREA);
  cv::resize(PTXPanorama, scaledPTX, cv::Size(), scale, scale, CV_INTER_AREA);

  Eigen::RowMatrixXd scaledRMap(scaledPTX.rows, scaledPTX.cols);
  for (int j = 0; j < scaledRMap.rows(); ++j) {
    for (int i = 0; i < scaledRMap.cols(); ++i) {
      int row = floor(j / scale);
      int col = floor(i / scale);
      double average = 0;
      int count = 0;
      for (int y = 0; y < 1.0 / scale; ++y) {
        for (int x = 0; x < 1.0 / scale; ++x) {
          if (row + y < 0 || row + y >= rMap.rows())
            continue;
          if (col + x < 0 || col + x >= rMap.cols())
            continue;
          if (!rMap(row + y, col + x))
            continue;
          average += rMap(row + y, col + x);
          ++count;
        }
      }
      average /= count;
      scaledRMap(j, i) = count > 0 ? average : 0;
    }
  }

  Eigen::ArrayXV3f scaledSurfNormals(scaledPTX.rows, scaledPTX.cols);
  for (int j = 0; j < scaledSurfNormals.rows(); ++j) {
    for (int i = 0; i < scaledSurfNormals.cols(); ++i) {
      int row = floor(j / scale);
      int col = floor(i / scale);
      Eigen::Vector3d average = Eigen::Vector3d::Zero();
      for (int y = 0; y < 1.0 / scale; ++y) {
        for (int x = 0; x < 1.0 / scale; ++x) {
          if (row + y < 0 || row + y >= surfaceNormals.rows())
            continue;
          if (col + x < 0 || col + x >= surfaceNormals.cols())
            continue;
          average += surfaceNormals(row + y, col + x).cast<double>();
        }
      }
      if (average != Eigen::Vector3d::Zero()) {
        average /= average.norm();
        scaledSurfNormals(j, i) = average.cast<float>();
      }
    }
  }

  std::vector<cv::KeyPoint> keypoints;
  cv::Ptr<cv::Feature2D> SIFT = cv::xfeatures2d::SIFT::create();

  auto domZs = getZPlanes(zCoords);

  place::Panorama pano;
  pano.imgs.resize(1);
  pano.imgs[0] = scaledPTX;
  pano.rMap = scaledRMap.cast<float>();
  pano.surfaceNormals = scaledSurfNormals;
  pano.floorCoord = domZs.minCoeff();

  LOG(INFO) << "Floorcoord: " << domZs.minCoeff() << std::endl;

  SIFT->detect(pano.imgs[0], keypoints);

  std::sort(keypoints, [](const auto &a, const auto &b) {
    return a.response > b.response;
  });

  keypoints.erase(keypoints.begin() + 0.7 * keypoints.size() + 1,
                  keypoints.end());

  cv::KeyPoint::convert(keypoints, pano.keypoints);

  double startSize = pano.keypoints.size();
  pano.keypoints.erase(std::remove_if(pano.keypoints,
                                      [&pano](const auto &kp) {
                                        return !pano.rMap(kp.y, kp.x) ||
                                               pano.surfaceNormals(kp.y,
                                                                   kp.x) ==
                                                   Eigen::Vector3f::Zero();
                                      }),
                       pano.keypoints.end());

  if (FLAGS_save)
    pano.writeToFile(panoName, dataName);
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
}

bool reloadNormals(pcl::PointCloud<NormalType>::Ptr &cloud_normals,
                   pcl::PointCloud<PointType>::Ptr &normals_points,
                   const fs::path &outName) {
  if (!fs::exists(outName))
    return false;

  std::ifstream in(outName.string(), std::ios::in | std::ios::binary);
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
                const fs::path &outName) {

  if (!FLAGS_redo && reloadNormals(cloud_normals, normals_points, outName))
    return;

  LOG(INFO) << "Calculating normals";
  pcl::PointCloud<PointType>::Ptr filtered_cloud(
      new pcl::PointCloud<PointType>);

  pcl::UniformSampling<PointType> uniform_sampling;
  uniform_sampling.setInputCloud(cloud);
  // 85 mm
  uniform_sampling.setRadiusSearch(0.0085f);
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
  norm_est.setRadiusSearch(0.03);
  norm_est.compute(*cloud_normals);

  std::vector<int> indices;
  pcl::removeNaNNormalsFromPointCloud(*cloud_normals, *cloud_normals, indices);
  pcl::copyPointCloud(*filtered_cloud, indices, *normals_points);

  LOG(INFO) << "Found {} normals"_format(cloud_normals->size());

  if (FLAGS_save)
    saveNormals(cloud_normals, normals_points, outName);
}

void saveNormals(const pcl::PointCloud<NormalType>::Ptr &cloud_normals,
                 pcl::PointCloud<PointType>::Ptr &normals_points,
                 const fs::path &outName) {
  assert(cloud_normals->size() == normals_points->size());
  std::ofstream out(outName.string(), std::ios::out | std::ios::binary);

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

  Eigen::Array3f init = Eigen::Array3f::Zero();
  auto[average, sigma] = utils::ave_and_stdev(
      points, init, [](auto &p) -> Eigen::Array3f { return p.point.array(); });

  Eigen::Array3f range(10, 10, 6);
  Eigen::Vector3f delta = (1.1 * sigma * range).matrix();

  pointMin = average.matrix() - delta / 2.0;
  pointMax = average.matrix() + delta / 2.0;
}

void findDoors(pcl::PointCloud<PointType>::Ptr &pointCloud,
               const Eigen::Vector3d &M1, const Eigen::Vector3d &M2,
               const Eigen::Vector3d &M3, const fs::path &outName) {
  if (!FLAGS_redo && fs::exists(outName))
    return;

  constexpr double voxelsPerMeter = 50, gradCutoff = 2.0,
                   wMin = 0.4 * voxelsPerMeter, wMax = 2.5 * voxelsPerMeter;

  voxel::HashVoxel<Eigen::Vector3i, double> grid;
  std::vector<double> zCoords;

  for (auto &point : *pointCloud) {
    const double depth = Eigen::Vector3d(point.x, point.y, point.z).norm();
    auto voxel = grid(point.x * voxelsPerMeter, point.y * voxelsPerMeter,
                      point.z * voxelsPerMeter);
    if (voxel)
      *voxel = (*voxel + depth) / 2.0;
    else
      grid.insert(depth, point.x * voxelsPerMeter, point.y * voxelsPerMeter,
                  point.z * voxelsPerMeter);

    zCoords.emplace_back(point.z);
  }

  auto domZs = getZPlanes(zCoords);

  const double hMax =
      std::max(
          2.1,
          std::min(2.6, 0.9 * std::abs(domZs.maxCoeff() - domZs.minCoeff()))) *
      voxelsPerMeter;

  const double hMin =
      std::min(1.8, 0.5 * std::abs(domZs.maxCoeff() - domZs.minCoeff())) *
      voxelsPerMeter;

  const int z0Index = (domZs.minCoeff() * voxelsPerMeter);
  const Eigen::Vector3d axises[] = {M2, M3, -M2, -M3};

  std::vector<place::Door> doors;
  auto traverse = [](const Eigen::Vector3d &x, double xInc,
                     const Eigen::Vector3d &z,
                     double zInc) { return x * xInc + z * zInc; };

  std::function<double(double)> r = [](const double &v) {
    return std::round(v);
  };

  auto getdepth = [&grid, &r](
      const Eigen::Vector3d &point, const Eigen::Vector3d &axis,
      const Eigen::Vector3d &zAxis, double xStop, double yStop, double zStop) {
    Eigen::Vector3d a2(axis[1], -axis[0], axis[2]);
    if (a2.dot(point) < 0.0)
      a2 *= -1.0;

    for (int i = 0; i <= xStop * voxelsPerMeter; ++i) {
      for (int j = 0; j <= yStop * voxelsPerMeter; ++j) {
        for (int k = 0; k <= zStop * voxelsPerMeter; ++k) {
          Eigen::Vector3i cur =
              (point + i * a2 + j * axis + k * zAxis).unaryExpr(r).cast<int>();
          auto v = grid(cur);
          if (v)
            return *v;

          if (i <= xStop * voxelsPerMeter / 2.0 &&
              j <= yStop * voxelsPerMeter / 2.0 &&
              k <= zStop * voxelsPerMeter / 2.0) {
            cur = (point - i * a2 - j * axis - k * zAxis)
                      .unaryExpr(r)
                      .cast<int>();
            v = grid(cur);
            if (v)
              return *v;
          }
        }
      }
    }

    return 1e10;
  };

#pragma omp declare reduction(                                                 \
    merge : std::vector <                                                      \
    place::Door > : omp_out.insert(omp_out.end(),                              \
                                               omp_in.begin(), omp_in.end()))

#pragma omp parallel for reduction(merge : doors)
  for (int j = grid.min()[1]; j < grid.max()[1]; ++j) {
    for (int i = grid.min()[0]; i < grid.max()[0]; ++i) {
      Eigen::Vector3d current = traverse(M2, i, M3, j);
      current[2] += z0Index;

      for (auto &axis : axises) {
        double h = 0.2 * voxelsPerMeter, w = 0, deltaDepth;
        do {
          ++h;
          Eigen::Vector3d delta = traverse(axis, 0.15 * voxelsPerMeter, M1, h);

          double leftDepth = getdepth(current + delta, axis, M1, 0.8, 0.0, 0.0);

          delta = traverse(axis, -0.15 * voxelsPerMeter, M1, h);

          double rightDepth =
              getdepth(current + delta, axis, M1, 0.8, 0.0, 0.0);

          deltaDepth = std::abs(leftDepth - rightDepth);
        } while (deltaDepth > gradCutoff && h <= hMax + 10);

        if (h < hMin || h > hMax)
          continue;

        double wUpper = 0, wLower = 1e10;
        for (double ch = h / 2; ch <= 18 * h / 20; ch += h / 20) {
          double testW = 5;
          Eigen::Vector3d delta;
          do {
            ++testW;
            delta = traverse(axis, testW, M1, ch);
          } while (getdepth(current + delta, axis, M1, 0.8, 0.0, 0.0) > 1e9 &&
                   testW <= wMax);

          wLower = std::min(testW, wLower);
          wUpper = std::max(testW, wUpper);

          if (testW < wMin || testW + 1 >= wMax)
            break;
        }

        w = (wUpper + wLower) / 2.0;

        if (wLower < wMin || wUpper + 1 >= wMax)
          continue;

        if (std::abs(wUpper - wLower) > 0.1 * voxelsPerMeter)
          continue;

        int onEdge = 0;
        for (double cw = 0; cw < w; ++cw) {
          Eigen::Vector3d delta =
              traverse(axis, cw, M1, h + 0.1 * voxelsPerMeter);

          double leftDepth = getdepth(current + delta, axis, M1, 0.3, 0.1, 0.0);

          delta = traverse(axis, cw, M1, h - 0.1 * voxelsPerMeter);

          double rightDepth =
              getdepth(current + delta, axis, M1, 0.3, 0.1, 0.0);

          deltaDepth = std::abs(leftDepth - rightDepth);

          if (deltaDepth > gradCutoff)
            ++onEdge;
        }

        if (onEdge < 0.8 * w)
          continue;

        double h2 = h / 2;
        do {
          ++h2;
          Eigen::Vector3d delta = traverse(axis, 0.05 * voxelsPerMeter, M1, h2);

          double leftDepth =
              getdepth(current + delta, axis, M1, 0.1, 0.00, 0.0);

          delta = traverse(axis, -0.05 * voxelsPerMeter, M1, h2);

          double rightDepth =
              getdepth(current + delta, axis, M1, 0.1, 0.00, 0.0);

          deltaDepth = std::abs(leftDepth - rightDepth);
        } while (deltaDepth > gradCutoff && h2 <= hMax);

        if (std::abs(h2 - h) > 0.1 * voxelsPerMeter)
          continue;

        doors.emplace_back(current / voxelsPerMeter, axis, M1,
                           h / voxelsPerMeter, w / voxelsPerMeter);
      }
    }
  }

  std::vector<double> heights;
  for (auto &d : doors)
    heights.push_back(d.h);

  double ave, sigma;
  std::tie(ave, sigma) = utils::ave_and_stdev(heights);

  LOG(INFO) << "Culling inconsistent heights: {} {}"_format(ave, sigma);

  doors.erase(
      std::remove_if(
          doors, [&](auto &d) { return std::abs(d.h - ave) / sigma > 1.0; }),
      doors.end());

  if (FLAGS_save) {
    int numDoors = doors.size();
    std::ofstream out(outName.string(), std::ios::out | std::ios::binary);
    out.write(reinterpret_cast<const char *>(&numDoors), sizeof(numDoors));
    for (auto &d : doors)
      d.writeToFile(out);
    out.close();
  }
}
