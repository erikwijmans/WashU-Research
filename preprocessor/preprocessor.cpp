/**
  The preprocessor is responsible for ingesting raw PTX files
  and converting them to binary, extracting surface normals,
  creating panoramas, and getting the 4 possible rotations to
  align the scan's dominates directions to the Manhattan world
  assumption (ie walls should be aligned with the X or Y axis)
*/
#include "preprocessor.h"
#include "getRotations.h"

#include <algorithm>
#include <fstream>
#include <iostream>
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
#include <pcl/visualization/pcl_visualizer.h>

#include <dirent.h>

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
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();
  return (viewer);
}

static int PTXrows, PTXcols;

static inline bool fexists(const std::string &file) {
  std::ifstream in(file, std::ios::in);
  return in.is_open();
}

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
    show_progress = new boost::progress_display(FLAGS_numScans * 5);

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
    const std::string dataName =
        FLAGS_panoFolder + "data/" + buildName + "_data_" + number + ".dat";
    const std::string rotName =
        FLAGS_rotFolder + buildName + "_rotations_" + number + ".dat";
    const std::string panoName = FLAGS_panoFolder + "images/" + buildName +
                                 "_panorama_" + number + ".png";

    if (FLAGS_redo ||
        !(fexists(binaryFileName) && fexists(normalsName) &&
          fexists(dataName) && fexists(rotName) && fexists(panoName))) {
      std::vector<scan::PointXYZRGBA> pointCloud;
      convertToBinary(csvFileName, binaryFileName, pointCloud);
      if (show_progress)
        ++(*show_progress);

      pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
      createPCLPointCloud(pointCloud, cloud);
      if (show_progress)
        ++(*show_progress);

      if (!FLAGS_quietMode)
        std::cout << "Calculating Normals" << std::endl;

      pcl::PointCloud<NormalType>::Ptr cloud_normals(
          new pcl::PointCloud<NormalType>);
      pcl::PointCloud<PointType>::Ptr normals_points(
          new pcl::PointCloud<PointType>);
      getNormals(cloud, cloud_normals, normals_points, normalsName);

      if (show_progress)
        ++(*show_progress);

      getRotations(cloud_normals, rotName);

      if (show_progress)
        ++(*show_progress);

      if (!FLAGS_quietMode)
        std::cout << "Creating Panorama" << std::endl;

      createPanorama(pointCloud, cloud_normals, normals_points, panoName,
                     dataName);

      if (show_progress)
        ++(*show_progress);
    } else
      *show_progress += 5;
  }

  if (show_progress)
    delete show_progress;

  std::cout << "Leaving" << std::endl;
  return 0;
}

static void dispDepthMap(const Eigen::RowMatrixXd &dm) {
  double average = 0;
  int count = 0;
  const double *dataPtr = dm.data();
  for (int i = 0; i < dm.size(); ++i) {
    if (*(dataPtr + i)) {
      average += *(dataPtr + i);
      ++count;
    }
  }
  average /= count;
  double sigma = 0;
  for (int i = 0; i < dm.size(); ++i) {
    const double val = *(dataPtr + i);
    if (val) {
      sigma += (val - average) * (val - average);
    }
  }

  sigma /= count - 1;
  sigma = sqrt(sigma);

  cv::Mat heatMap(dm.rows(), dm.cols(), CV_8UC3, cv::Scalar::all(0));
  for (int j = 0; j < heatMap.rows; ++j) {
    uchar *dst = heatMap.ptr<uchar>(j);
    for (int i = 0; i < heatMap.cols; ++i) {
      if (dm(j, i)) {
        const int gray = cv::saturate_cast<uchar>(
            255.0 * ((dm(j, i) - average) / (1.0 * sigma) + 1.0) / 2.0);
        int red, green, blue;
        if (gray < 128) {
          red = 0;
          green = 2 * gray;
          blue = 255 - green;
        } else {
          blue = 0;
          red = 2 * (gray - 128);
          green = 255 - red;
        }
        dst[i * 3] = blue;
        dst[i * 3 + 1] = green;
        dst[i * 3 + 2] = red;
      }
    }
  }
  cvNamedWindow("dm", CV_WINDOW_NORMAL);
  cv::imshow("dm", heatMap);
}

static void dispSurfaceNormals(const Eigen::ArrayXV3f &sn) {
  cv::Mat heatMap(sn.rows(), sn.cols(), CV_8UC3, cv::Scalar::all(255));
  for (int j = 0; j < heatMap.rows; ++j) {
    uchar *dst = heatMap.ptr<uchar>(j);
    for (int i = 0; i < heatMap.cols; ++i) {
      if (sn(j, i) != Eigen::Vector3f::Zero()) {

        int red = 255 * std::abs(sn(j, i).dot(Eigen::Vector3f::UnitX()));
        int green = 255 * std::abs(sn(j, i).dot(Eigen::Vector3f::UnitY()));
        int blue = 255 * std::abs(sn(j, i).dot(Eigen::Vector3f::UnitZ()));

        dst[i * 3] = blue;
        dst[i * 3 + 1] = green;
        dst[i * 3 + 2] = red;
      }
    }
  }
  cvNamedWindow("sn", CV_WINDOW_NORMAL);
  cv::imshow("sn", heatMap);
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
    PTXcols = columns;
    PTXrows = rows;
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

    PTXcols = columns;
    PTXrows = rows;

    pointCloud.resize(columns * rows);
    for (auto &p : pointCloud)
      p.loadFromFile(in);

    in.close();
  }
}

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch (depth) {
  case CV_8U:
    r = "8U";
    break;
  case CV_8S:
    r = "8S";
    break;
  case CV_16U:
    r = "16U";
    break;
  case CV_16S:
    r = "16S";
    break;
  case CV_32S:
    r = "32S";
    break;
  case CV_32F:
    r = "32F";
    break;
  case CV_64F:
    r = "64F";
    break;
  default:
    r = "User";
    break;
  }

  r += "C";
  r += (chans + '0');

  return r;
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
                    const std::string &panoName, const std::string &dataName) {
  if (!FLAGS_redo && fexists(panoName) && fexists(dataName))
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

  for (auto &element : pointCloud) {
    assert(row >= 0 && row < PTXPanorama.rows);
    assert(col >= 0 && col < PTXPanorama.cols);
    _PTXPanorama(row, col)[0] = element.rgb[2];
    _PTXPanorama(row, col)[1] = element.rgb[1];
    _PTXPanorama(row, col)[2] = element.rgb[0];

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

  place::Panorama pano;
  pano.imgs.resize(1);
  pano.imgs[0] = scaledPTX;
  pano.rMap = scaledRMap.cast<float>();
  pano.surfaceNormals = scaledSurfNormals;

  SIFT->detect(pano.imgs[0], keypoints);

  std::sort(
      keypoints.begin(), keypoints.end(),
      [](const auto &a, const auto &b) { return a.response > b.response; });

  keypoints.erase(keypoints.begin() + 0.7 * keypoints.size() + 1,
                  keypoints.end());

  cv::KeyPoint::convert(keypoints, pano.keypoints);

  double startSize = pano.keypoints.size();
  pano.keypoints.erase(
      std::remove_if(pano.keypoints.begin(), pano.keypoints.end(),
                     [&pano](const auto &kp) {
                       return !pano.rMap(kp.y, kp.x) ||
                              pano.surfaceNormals(kp.y, kp.x) ==
                                  Eigen::Vector3f::Zero();
                     }),
      pano.keypoints.end());

  // pano.imgs[0] =
  //     cv::Mat(scaledPTX.size(), scaledPTX.type(), cv::Scalar::all(0));
  // GaussianBlur(scaledPTX, pano.imgs[0], cv::Size(5, 5), 0);

  if (FLAGS_save)
    pano.writeToFile(panoName, dataName);

  if (FLAGS_preview) {
    std::cout << "Well formed kps: " << pano.keypoints.size() / startSize * 100
              << "\%" << std::endl;

    dispDepthMap(scaledRMap);
    dispSurfaceNormals(scaledSurfNormals);
    cvNamedWindow("Tracking", CV_WINDOW_NORMAL);
    cv::imshow("Tracking", scaledTracking);
    cvNamedWindow("PTX", CV_WINDOW_NORMAL);
    cv::imshow("PTX", PTXPanorama);
    cv::Mat out;
    cv::drawKeypoints(scaledPTX, keypoints, out);
    cvNamedWindow("KP", CV_WINDOW_NORMAL);
    cv::imshow("KP", out);
    cvNamedWindow("Blur", CV_WINDOW_NORMAL);
    cv::imshow("Blur", pano.imgs[0]);
    cv::waitKey(0);
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

  if (!FLAGS_redo && reloadNormals(cloud_normals, normals_points, outName))
    return;

  pcl::PointCloud<PointType>::Ptr filtered_cloud(
      new pcl::PointCloud<PointType>);

  pcl::UniformSampling<PointType> uniform_sampling;
  uniform_sampling.setInputCloud(cloud);
  // 85 mm
  uniform_sampling.setRadiusSearch(0.0085f);
  uniform_sampling.filter(*filtered_cloud);

  // pcl::visualization::PCLVisualizer::Ptr viewer = rgbVis(filtered_cloud);
  // while (!viewer->wasStopped()) {
  //   viewer->spinOnce(100);
  //   boost::this_thread::sleep(boost::posix_time::microseconds(100000));
  // }

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

  if (FLAGS_save)
    saveNormals(cloud_normals, normals_points, outName);
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

  Eigen::Vector3f range(10, 10, 6);
  Eigen::Vector3f delta = 1.1 * sigma.cwiseProduct(range);

  pointMin = average - delta / 2.0;
  pointMax = average + delta / 2.0;
}
