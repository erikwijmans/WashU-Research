#include "getRotations.h"
#include "preprocessor.h"

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/StdVector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>

#include "opencv2/xfeatures2d.hpp"

#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/keypoints/impl/sift_keypoint.hpp>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/progress.hpp>

#include <dirent.h>

DEFINE_bool(ptx, false, "Tells the preprocessor to read from ptx file and convert it to binary.  Otherwise, a binary file will be read");

int PTXrows, PTXcols;

void convertToBinary(const std::string & fileNameIn,
  const std::string &,
  std::vector<scan::PointXYZRGBA> & pointCloud);
void createPanorama(const std::vector<scan::PointXYZRGBA> & pointCloud,
  pcl::PointCloud<NormalType>::Ptr & cloud_normals,
  pcl::PointCloud<PointType>::Ptr & normals_points,
  const std::string & panoName,
  const std::string & dataName);
void boundingBox(const std::vector<scan::PointXYZRGBA> & points,
  Eigen::Vector3f & pointMin, Eigen::Vector3f & pointMax);
void createPCLPointCloud(const std::vector<scan::PointXYZRGBA> & points,
  pcl::PointCloud<PointType>::Ptr & cloud);
void getNormals(const pcl::PointCloud<PointType>::Ptr & cloud,
  pcl::PointCloud<NormalType>::Ptr & cloud_normals,
  pcl::PointCloud<PointType>::Ptr & normals_points,
  const std::string & outName);
void saveNormals(const pcl::PointCloud<NormalType>::Ptr & cloud_normals,
  pcl::PointCloud<PointType>::Ptr & normals_points,
  const std::string & outName);
void getDescriptors(const pcl::PointCloud<PointType>::Ptr & cloud,
  const pcl::PointCloud<NormalType>::Ptr & cloud_normals,
  const pcl::PointCloud<PointType>::Ptr & normals_points,
  pcl::PointCloud<DescriptorType>::Ptr & cloud_descriptors,
  pcl::PointCloud<PointType>::Ptr & filtered_cloud);
void saveDescriptors(const pcl::PointCloud<PointType>::Ptr & cloud,
  const pcl::PointCloud<DescriptorType>::Ptr & cloud_descriptors,
  const std::string & outName);
void SIFT(const pcl::PointCloud<PointType>::Ptr & cloud,
  const std::string & outName);

static inline bool fexists(const std::string & file) {
  std::ifstream in (file, std::ios::in);
  return in.is_open();
}

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  prependDataPath();

  std::vector<std::string> csvFileNames;

  std::string * inFolder;
  if (FLAGS_ptx)
    inFolder = &FLAGS_PTXFolder;
  else
    inFolder = &FLAGS_binaryFolder;

  DIR *dir;
  struct dirent *ent;
  if ((dir = opendir (inFolder->data())) != NULL) {
    while ((ent = readdir (dir)) != NULL) {
      std::string fileName = ent->d_name;
      if(fileName != ".." && fileName != "."){
        csvFileNames.push_back(fileName);
      }
    }
    closedir (dir);
  }  else {
    perror ("");
    return EXIT_FAILURE;
  }

  sort(csvFileNames.begin(), csvFileNames.end(),
    [](const std::string & a, const std::string & b) {
        int numA = std::stoi(a.substr(a.find(".") - 3, 3));
        int numB = std::stoi(b.substr(b.find(".") - 3, 3));
        return numA < numB;
    }
  );

  if (FLAGS_numScans == -1 )
    FLAGS_numScans = csvFileNames.size() - FLAGS_startIndex;

  boost::progress_display * show_progress = nullptr;
  if (FLAGS_quietMode)
    show_progress = new boost::progress_display (FLAGS_numScans);

  for (int i = FLAGS_startIndex; i < FLAGS_numScans + FLAGS_startIndex; ++i) {
    const std::string number = csvFileNames[i].substr(csvFileNames[i].find(".") - 3, 3);
    const std::string buildName = csvFileNames[i].substr(csvFileNames[i].rfind("/") + 1, 3);
    const std::string csvFileName = FLAGS_PTXFolder + csvFileNames[i];

    const std::string binaryFileName = FLAGS_binaryFolder +
      buildName + "_binary_" + number + ".dat";
    const std::string normalsName = FLAGS_normalsFolder +
      buildName + "_normals_" + number + ".dat";
    const std::string dataName = FLAGS_panoFolder + "data/"
      + buildName + "_data_" + number + ".dat";
    const std::string rotName = FLAGS_rotFolder +
      buildName + "_rotations_" + number + ".dat";
    const std::string panoName = FLAGS_panoFolder + "images/"
      + buildName + "_panorama_" + number + ".png";

    if (FLAGS_redo || !(fexists(binaryFileName) && fexists(normalsName)
        && fexists(dataName) && fexists(rotName) && fexists(panoName))) {
      std::vector<scan::PointXYZRGBA> pointCloud;
      convertToBinary(csvFileName, binaryFileName, pointCloud);

      pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType>);
      createPCLPointCloud(pointCloud, cloud);

      if (!FLAGS_quietMode)
        std::cout << "Calculating Normals" << std::endl;


      pcl::PointCloud<NormalType>::Ptr cloud_normals (new pcl::PointCloud<NormalType>);
      pcl::PointCloud<PointType>::Ptr normals_points (new pcl::PointCloud<PointType>);
      getNormals(cloud, cloud_normals, normals_points, normalsName);

      getRotations(cloud_normals, rotName);

      if (!FLAGS_quietMode)
        std::cout << "Creating Panorama" << std::endl;

      createPanorama(pointCloud, cloud_normals, normals_points, panoName, dataName);
    }

    if (show_progress)
      ++(*show_progress);
  }

  if (show_progress)
    delete show_progress;

  std::cout << "Leaving" << std::endl;
  return 0;
}

static void dispDepthMap(const Eigen::RowMatrixXd & dm) {
  double average = 0;
  int count = 0;
  const double * dataPtr = dm.data();
  for (int i = 0; i < dm.size(); ++i) {
    if(*(dataPtr + i)) {
      average += *(dataPtr + i);
      ++count;
    }
  }
  average /= count;
  double sigma = 0;
  for (int i = 0; i < dm.size(); ++i) {
    const double val = *(dataPtr + i);
    if (val) {
      sigma += (val - average)*(val -average);
    }
  }

  sigma /= count - 1;
  sigma = sqrt(sigma);

  cv::Mat heatMap (dm.rows(), dm.cols(), CV_8UC3, cv::Scalar::all(0));
  for (int j = 0; j < heatMap.rows; ++j) {
    uchar * dst = heatMap.ptr<uchar>(j);
    for (int i = 0; i < heatMap.cols; ++i) {
      if(dm(j, i)){
        const int gray = cv::saturate_cast<uchar>(
          255.0 * ((dm(j, i) - average)
            / (1.0 * sigma) + 1.0)/2.0);
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
        dst[i*3] = blue;
        dst[i*3 +1] = green;
        dst[i*3 + 2] = red;
      }
    }
  }
  cvNamedWindow("dm", CV_WINDOW_NORMAL);
  cv::imshow("dm", heatMap);
}

static void dispSurfaceNormals(const Eigen::ArrayXV3f & sn) {
  cv::Mat heatMap (sn.rows(), sn.cols(), CV_8UC3, cv::Scalar::all(255));
  for (int j = 0; j < heatMap.rows; ++j) {
    uchar * dst = heatMap.ptr<uchar>(j);
    for (int i = 0; i < heatMap.cols; ++i) {
      if(sn(j, i) != Eigen::Vector3f::Zero()){

        int red = 255*std::abs(sn(j, i).dot(Eigen::Vector3f::UnitX()));
        int green = 255*std::abs(sn(j, i).dot(Eigen::Vector3f::UnitY()));
        int blue = 255*std::abs(sn(j, i).dot(Eigen::Vector3f::UnitZ()));

        dst[i*3] = blue;
        dst[i*3 +1] = green;
        dst[i*3 + 2] = red;
      }
    }
  }
  cvNamedWindow("sn", CV_WINDOW_NORMAL);
  cv::imshow("sn", heatMap);
}

void convertToBinary(const std::string & fileNameIn,
  const std::string & outName,
  std::vector<scan::PointXYZRGBA> & pointCloud) {

  if (!FLAGS_quietMode)
    std::cout << outName << std::endl;

  std::ifstream in (outName, std::ios::in | std::ios::binary);

  if (FLAGS_ptx || !in.is_open()) {
  	in.close();
    std::ifstream scanFile (fileNameIn, std::ios::in);
    int columns, rows;
    scanFile >> columns >> rows;
    std::ofstream out (outName, std::ios::out | std::ios::binary);
    out.write(reinterpret_cast<const char *>(&columns), sizeof(columns));
    out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    PTXcols = columns;
    PTXrows = rows;
    std::string line;
    if (!FLAGS_quietMode)
      std::cout << rows << "   " << columns << std::endl;
    pointCloud.reserve(columns*rows);
    for (int i = 0; i < 9; ++i) {
      getline(scanFile, line);
    }

    for (int k = 0; k < columns * rows; ++k) {
      scan::PointXYZRGBA tmp;
      scanFile >> tmp.point[0] >> tmp.point[1] >> tmp.point[2];

      scanFile >> tmp.intensity;

      double itmp [3];
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

    pointCloud.resize(columns*rows);
    for (auto & p : pointCloud)
      p.loadFromFile(in);

    in.close();
  }
}

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

static Eigen::Vector3d cartesianToPolar(const Eigen::Vector3d & coords) {
  double r = coords.norm();
  double theta = atan2(coords[1], coords[0]);
  double phi = acos(coords[2]/r);
  return Eigen::Vector3d (r, theta, phi);
}

static Eigen::Vector3d polarToCartesian(const Eigen::Vector3d & coords) {
  double x = coords[0]*cos(coords[1])*sin(coords[2]);
  double y = coords[0]*sin(coords[1])*sin(coords[2]);
  double z = coords[0]*cos(coords[2]);
  return Eigen::Vector3d (x,y,z);
}

static Eigen::Vector3d pointCloudToPanorama(const Eigen::Vector3f & worldCoord,
  const cv::Size & resolution) {
  auto polar = cartesianToPolar(worldCoord.cast<double>());
  int x = (polar[1]/PI + 1.0)*(resolution.width - 1)/2.0;
  int y = polar[2]*(resolution.height - 1)/maxPhi;
  return Eigen::Vector3d(x, y, polar[0]);
}

void fillGaps(auto & mat, auto & mask) {
  assert(mat.rows() == mask.rows());
  assert(mat.cols() == mask.cols());
  int count = 0;
  constexpr int limit = 15;
  decltype(mat.data()) current = nullptr;
  auto maskPtr = mask.data();
  auto matPtr = mat.data();

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

void createPanorama(const std::vector<scan::PointXYZRGBA> & pointCloud,
  pcl::PointCloud<NormalType>::Ptr & cloud_normals,
  pcl::PointCloud<PointType>::Ptr & normals_points,
  const std::string & panoName,
  const std::string & dataName) {

  cv::Mat trackingPanorama (PTXrows, PTXcols, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat_<cv::Vec3b> _trackingPanorama = trackingPanorama;

  cv::Mat PTXPanorama(PTXrows, PTXcols, CV_8UC3);
  cv::Mat_<cv::Vec3b> _PTXPanorama = PTXPanorama;
  Eigen::RowMatrixXd rMap = Eigen::RowMatrixXd::Zero(PTXrows, PTXcols);

  Eigen::RowMatrixXb hasNormal = Eigen::RowMatrixXb::Zero(PTXrows, PTXcols);
  Eigen::ArrayXV3f surfaceNormals (PTXrows, PTXcols);

  int row = PTXrows - 1;
  int col = 0.995*(PTXcols - 1)/2.0;
  Eigen::RowMatrixXb touched = Eigen::RowMatrixXb::Zero(PTXrows, PTXcols);

  for(auto element : pointCloud) {
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

    auto panoCoord = pointCloudToPanorama(element.point, trackingPanorama.size());
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
    auto & p = normals_points->at(i);
    auto & n = cloud_normals->at(i);
    Eigen::Vector3f coord (p.x, p.y, p.z);
    auto panoCoord = pointCloudToPanorama(coord, trackingPanorama.size());
    const double r = panoCoord[2];
    const int trackedRow = panoCoord[1];
    const int trackedCol = panoCoord[0];
    surfaceNormals(trackedRow, trackedCol) = Eigen::Vector3f (n.normal_x, n.normal_y, n.normal_z);
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

  /*for (int j = 0; j < touched.rows(); ++j) {
    const int panoRow = j;
    uchar * dst = trackingPanorama.ptr<uchar>(panoRow);
    const uchar * src = PTXPanorama.ptr<uchar>(panoRow);
    for (int i = 0; i < touched.cols(); ++i) {
      if (!touched(j, i)) {
        const int panoCol = i;
        const int tCol = 3*panoCol;
        const int pCol = 3*panoCol;
        dst[tCol + 0] = src[pCol + 0];
        dst[tCol + 1] = src[pCol + 1];
        dst[tCol + 2] = src[pCol + 2];
      }
    }
  }*/

  constexpr double scale = pow(2, -3.0/2);
  cv::Mat scaledTracking, scaledPTX;
  cv::resize(trackingPanorama, scaledTracking, cv::Size(), scale, scale, CV_INTER_AREA);
  cv::resize(PTXPanorama, scaledPTX, cv::Size(), scale, scale, CV_INTER_AREA);

  Eigen::RowMatrixXd scaledRMap (scaledPTX.rows, scaledPTX.cols);
  for (int j = 0; j < scaledRMap.rows(); ++j) {
    for (int i = 0; i < scaledRMap.cols(); ++i) {
      int row = floor(j/scale);
      int col = floor(i/scale);
      double average = 0;
      int count = 0;
      for (int y = 0; y < 1.0/scale; ++y) {
        for (int x = 0; x < 1.0/scale; ++x) {
          if (row + y < 0 || row + y >= rMap.rows())
            continue;
          if (col + x < 0 || col + x >= rMap.cols())
            continue;
          if (!rMap(row + y, col + x)) continue;
          average += rMap(row + y, col + x);
          ++count;
        }
      }
      average /= count;
      scaledRMap(j, i) = count > 0 ? average : 0;
    }
  }

  Eigen::ArrayXV3f scaledSurfNormals (scaledPTX.rows, scaledPTX.cols);
  for (int j = 0; j < scaledSurfNormals.rows(); ++j) {
    for (int i = 0; i < scaledSurfNormals.cols(); ++i) {
      int row = floor(j/scale);
      int col = floor(i/scale);
      Eigen::Vector3d average = Eigen::Vector3d::Zero();
      for (int y = 0; y < 1.0/scale; ++y) {
        for (int x = 0; x < 1.0/scale; ++x) {
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
  std::sort(keypoints.begin(), keypoints.end(),
    [](const auto & a, const auto & b) {
      return a.response > b.response;
    });
  keypoints.erase(keypoints.begin() + 0.7*keypoints.size() + 1, keypoints.end());

  pano.imgs[0] = cv::Mat(scaledPTX.size(), scaledPTX.type(), cv::Scalar::all(0));
  GaussianBlur(scaledPTX, pano.imgs[0], cv::Size(7, 7), 0, 0);

  cv::KeyPoint::convert(keypoints, pano.keypoints);

  double startSize = pano.keypoints.size();
  for (auto it = pano.keypoints.begin(); it != pano.keypoints.end();)
    if (!pano.rMap(it->y, it->x) ||
      pano.surfaceNormals(it->y, it->x) == Eigen::Vector3f::Zero())
      pano.keypoints.erase(it);
    else
      ++it;

  if (FLAGS_save)
    pano.writeToFile(panoName, dataName);

  if (FLAGS_preview) {
    std::cout << "Well formed kps: " << pano.keypoints.size()/startSize*100
      << "\%" << std::endl;

    dispDepthMap(scaledRMap);
    dispSurfaceNormals(scaledSurfNormals);
    cvNamedWindow("Tracking", CV_WINDOW_NORMAL);
    cv::imshow("Tracking", scaledTracking);
    cvNamedWindow("PTX", CV_WINDOW_NORMAL);
    cv::imshow("PTX", scaledPTX);
    cv::Mat out;
    cv::drawKeypoints(scaledPTX, keypoints, out);
    cvNamedWindow("KP", CV_WINDOW_NORMAL);
    cv::imshow("KP", out);
    cvNamedWindow("Blur", CV_WINDOW_NORMAL);
    cv::imshow("Blur", pano.imgs[0]);
    cv::waitKey(0);
  }
}

void createPCLPointCloud(const std::vector<scan::PointXYZRGBA> & points,
  pcl::PointCloud<PointType>::Ptr & cloud) {

  Eigen::Vector3f pointMin, pointMax;
  boundingBox(points, pointMin, pointMax);

  for (auto & p : points) {
    bool in = true;
    for (int i = 0; i < 3; ++i)
      if (p.point[i] < pointMin[i] || p.point[i] > pointMax[i])
        in = false;

    if (!in) continue;
    if (p.intensity < 0.2) continue;

    auto & point = p.point;
    auto & rgb = p.rgb;
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

bool reloadNormals(pcl::PointCloud<NormalType>::Ptr & cloud_normals,
  pcl::PointCloud<PointType>::Ptr & normals_points,
  const std::string & outName) {
  std::ifstream in (outName, std::ios::in | std::ios::binary);
  if (!in.is_open())
    return false;

  size_t size;
  in.read(reinterpret_cast<char *>
    (& size), sizeof(size_t));

  cloud_normals->resize(size);
  normals_points->resize(size);

  for (int i = 0; i < size; ++i) {
    in.read(reinterpret_cast<char *> (&cloud_normals->at(i)),
      sizeof(NormalType));
    in.read(reinterpret_cast<char *>(&normals_points->at(i)),
      sizeof(PointType));
  }

  return true;
}

void getNormals(const pcl::PointCloud<PointType>::Ptr & cloud,
  pcl::PointCloud<NormalType>::Ptr & cloud_normals,
  pcl::PointCloud<PointType>::Ptr & normals_points,
  const std::string & outName) {

  if (!FLAGS_redo) {
    if (reloadNormals(cloud_normals, normals_points, outName))
      return;
  }

  pcl::PointCloud<int> sampled_indices;
  pcl::PointCloud<PointType>::Ptr filtered_cloud (new pcl::PointCloud<PointType>);
  pcl::UniformSampling<PointType> uniform_sampling;

  uniform_sampling.setInputCloud(cloud);
  // 85 mm
  uniform_sampling.setRadiusSearch(0.0085f);
  uniform_sampling.compute(sampled_indices);

  pcl::copyPointCloud(*cloud,
    sampled_indices.points, *filtered_cloud);

  pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
  pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType>);
  norm_est.setInputCloud (filtered_cloud);
  norm_est.setSearchSurface(cloud);
  norm_est.setSearchMethod (tree);
  norm_est.setRadiusSearch(0.03);
  norm_est.compute (*cloud_normals);

  size_t startSize = cloud_normals->size();
  std::vector<int> indices;
  pcl::removeNaNNormalsFromPointCloud(*cloud_normals, *cloud_normals, indices);
  pcl::copyPointCloud(*filtered_cloud, indices, *normals_points);

  if (FLAGS_save)
    saveNormals(cloud_normals, normals_points, outName);
}

void saveNormals(const pcl::PointCloud<NormalType>::Ptr & cloud_normals,
  pcl::PointCloud<PointType>::Ptr & normals_points,
  const std::string & outName) {
  assert(cloud_normals->size() == normals_points->size());
  std::ofstream out (outName, std::ios::out | std::ios::binary);

  size_t size = cloud_normals->points.size();
  out.write(reinterpret_cast<const char *>
    (& size), sizeof(size_t));

  for (int i = 0; i < size; ++i) {
    out.write(reinterpret_cast<const char *> (&cloud_normals->at(i)),
      sizeof(NormalType));
    out.write(reinterpret_cast<const char *>(&normals_points->at(i)),
      sizeof(PointType));
  }

  out.close();
}

void getDescriptors(const pcl::PointCloud<PointType>::Ptr & cloud,
  const pcl::PointCloud<NormalType>::Ptr & cloud_normals,
  const pcl::PointCloud<PointType>::Ptr & normals_points,
  pcl::PointCloud<DescriptorType>::Ptr & cloud_descriptors,
  pcl::PointCloud<PointType>::Ptr & filtered_cloud) {

  pcl::PointCloud<int> sampled_indices;
  pcl::UniformSampling<PointType> uniform_sampling;

  uniform_sampling.setInputCloud(cloud);
  uniform_sampling.setRadiusSearch(0.05);
  uniform_sampling.compute(sampled_indices);

  pcl::copyPointCloud(*cloud,
    sampled_indices.points, *filtered_cloud);

  pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> shot_est;

  shot_est.setRadiusSearch (0.03);
  shot_est.setInputCloud (filtered_cloud);
  shot_est.setInputNormals (cloud_normals);
  shot_est.setSearchSurface (normals_points);
  shot_est.compute (*cloud_descriptors);
}

static double norm1344 (float * vec) {
  double norm = 0;
  #pragma omp simd
  for (int i = 0; i < 1344; ++i) {
    norm += vec[i]*vec[i];
  }
  return sqrt(norm);
}

static bool finite1344(float * vec) {
  for (int i = 0; i < 1344; ++i) {
    if (!pcl_isfinite(vec[i])) return false;
  }
  return true;
}

static double norm352 (float * vec) {
  double norm = 0;
  #pragma omp simd
  for (int i = 0; i < 352; ++i) {
    norm += vec[i]*vec[i];
  }
  return sqrt(norm);
}

static bool finite352(float * vec) {
  for (int i = 0; i < 352; ++i) {
    if (!pcl_isfinite(vec[i])) return false;
  }
  return true;
}

void saveDescriptors(const pcl::PointCloud<PointType>::Ptr & cloud,
  const pcl::PointCloud<DescriptorType>::Ptr & cloud_descriptors,
  const std::string & outName) {

  std::vector<int> wellFormedIndicies;
  for (int i = 0; i < cloud_descriptors->size(); ++i)
    if (finite352(cloud_descriptors->at(i).descriptor) &&
      norm352(cloud_descriptors->at(i).descriptor))
      wellFormedIndicies.push_back(i);

  std::ofstream out (outName, std::ios::out | std::ios::binary);
  int num = wellFormedIndicies.size();
  out.write(reinterpret_cast<const char *>(&num), sizeof(num));
  for (int i = 0; i < num; ++i) {
    int index = wellFormedIndicies[i];
    SPARSE352WithXYZ SHOTout;
    SHOTout.position = Eigen::Vector3d
      (cloud->at(index).x, cloud->at(index).y, cloud->at(index).z);
    for (int k = 0; k < SHOTout.descriptor->size(); ++k) {
      float value = cloud_descriptors->at(index).descriptor[k];
      if (value)
        SHOTout.descriptor->coeffRef(k) = value;
    }
    SHOTout.writeToFile(out);
  }
  out.close();
}

void boundingBox(const std::vector<scan::PointXYZRGBA> & points,
  Eigen::Vector3f & pointMin, Eigen::Vector3f & pointMax) {
  Eigen::Vector3f average = Eigen::Vector3f::Zero();
  Eigen::Vector3f sigma = Eigen::Vector3f::Zero();
  for (auto & point : points) {
    average += point.point;
  }
  average /= points.size();

  for (auto & point : points)
    for (int i = 0; i < 3; ++i)
      sigma[i] += (point.point[i] - average[i])*(point.point[i] - average[i]);

  sigma /= points.size() - 1;
  for (int i = 0; i < 3; ++i)
    sigma[i] = sqrt(sigma[i]);

  Eigen::Vector3f range (10, 10, 6);
  Eigen::Vector3f delta;
  for (int i = 0; i < delta.size(); ++i)
    delta[i] = 1.1*range[i]*sigma[i];

  pointMin = average - delta/2.0;
  pointMax = average + delta/2.0;
}

void saveSIFT(const pcl::PointCloud<pcl::PointXYZ>::Ptr & SIFT,
  const std::string & outName) {
  std::ofstream out (outName, std::ios::out | std::ios::binary);
  int size = SIFT->size();
  out.write(reinterpret_cast<const char *>(&size), sizeof(size));

  for (auto & F : *SIFT) {
    Eigen::Vector3f tmp (F.x, F.y, F.z);
    out.write(reinterpret_cast<const char *>(tmp.data()), sizeof(tmp));
  }
  out.close();
}

void SIFT(const pcl::PointCloud<PointType>::Ptr & cloud,
  const std::string & outName) {

  // Parameters for sift computation
  const float min_scale = 0.1f;
  const int n_octaves = 6;
  const int n_scales_per_octave = 10;
  const float min_contrast = 0.5f;

  pcl::SIFTKeypoint<PointType, pcl::PointWithScale> sift;
  pcl::PointCloud<pcl::PointWithScale> result;
  pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType> ());
  sift.setSearchMethod(tree);
  sift.setScales(min_scale, n_octaves, n_scales_per_octave);
  sift.setMinimumContrast(min_contrast);
  sift.setInputCloud(cloud);
  sift.compute(result);

  pcl::PointCloud<pcl::PointXYZ>::Ptr SIFT (new pcl::PointCloud<pcl::PointXYZ>);
  copyPointCloud(result, *SIFT);

  if (false) {
    pcl::PointCloud<PointType>::Ptr sub_sampled (new pcl::PointCloud<PointType>);
    int subSample = 0;
    const int subSampleSize = 10;
    for (auto & p : *cloud)
      if (subSample++ % subSampleSize == 0)
        sub_sampled->push_back(p);
    // Visualization of keypoints along with the original cloud
      pcl::visualization::PCLVisualizer viewer("PCL Viewer");
      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints_color_handler (SIFT, 0, 255, 0);
      pcl::visualization::PointCloudColorHandlerCustom<PointType> cloud_color_handler (cloud, 255, 255, 0);
      viewer.setBackgroundColor( 0.0, 0.0, 0.0 );
      viewer.addPointCloud(sub_sampled, "cloud");
      viewer.addPointCloud(SIFT, keypoints_color_handler, "keypoints");
      viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "keypoints");

      while (!viewer.wasStopped ()) {
        viewer.spinOnce ();
      }
  }
  saveSIFT(SIFT, outName);
  exit(1);
}