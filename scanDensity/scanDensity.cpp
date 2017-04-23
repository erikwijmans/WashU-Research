/**
  Implenets the DensityMapsManager which is responsible for
  loading in data, figuring out what type of analysis still
  needs to be done for each scan, and getting the names for
  files to be saved as.
  Also implements CloudAnalyzer2D, which is responsible for
  examing the 2D free space and point evidence. Due to the simplicity
  of this data, it is not responsible for saving it
  Implements BoundingBox which is used to remove points
  that are outliers from the pointcloud
*/

#include "scanDensity_scanDensity.h"

#include <locale>
#include <sstream>

#include <omp.h>

DensityMapsManager::DensityMapsManager(const std::string &commandLine)
    : R{NULL}, pointsWithCenter{NULL}, pointsNoCenter{NULL} {
  this->resetFlags(commandLine);
}

DensityMapsManager::DensityMapsManager(int argc, char *argv[])
    : R{NULL}, pointsWithCenter{NULL}, pointsNoCenter{NULL} {
  this->resetFlags(argc, argv);
}

void DensityMapsManager::resetFlags(const std::string &commandLine) {
  std::vector<std::string> v;
  std::istringstream is(commandLine);
  std::string tmp;
  while (is >> tmp)
    v.push_back(tmp);
  char **argv = new char *[v.size() + 1];
  for (int i = 0; i < v.size(); ++i) {
    argv[i] = &v[i][0];
  }
  argv[v.size()] = NULL;
  resetFlags(v.size() + 1, argv);
  delete[] argv;
}

void DensityMapsManager::resetFlags(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  prependDataPath();

  if (!FLAGS_2D && !FLAGS_3D)
    FLAGS_2D = FLAGS_3D = true;

  if (!FLAGS_pe && !FLAGS_fe)
    FLAGS_pe = FLAGS_fe = true;

  utils::parse_folder(FLAGS_binaryFolder, binaryNames);
  utils::parse_folder(FLAGS_rotFolder, rotationsFiles);
  utils::parse_folder(FLAGS_doorsFolder + "/pointcloud", doorsNames);

  auto[buildName, tmp] = parse_name(rotationsFiles[0]);

  if (FLAGS_scale == -1) {
    FLAGS_scale = buildingScale.getScale();
  }

  if (FLAGS_startNumber != -1)
    FLAGS_startIndex = numberToIndex(binaryNames, FLAGS_startNumber);

  if (FLAGS_numScans == -1)
    FLAGS_numScans = binaryNames.size() - FLAGS_startIndex;

  CHECK(FLAGS_startIndex + FLAGS_numScans <= binaryNames.size())
      << "Not enough binary files" << std::endl;
  CHECK(FLAGS_startIndex + FLAGS_numScans <= rotationsFiles.size())
      << "Not enough rotations files" << std::endl;
  CHECK(binaryNames.size() == rotationsFiles.size() &&
        binaryNames.size() == doorsNames.size())
      << "Must have same number of binary files, rotations files, and doors "
         "files";

  this->current = FLAGS_startIndex;
}

void DensityMapsManager::run() {
  rotationFile = rotationsFiles[current];
  fileName = binaryNames[current];
  doorName = doorsNames[current];

  std::tie(buildName, scanNumber) = parse_name(fileName);

  if (!FLAGS_redo && exists2D() && exists3D() && existsDoors())
    return;

  LOG(INFO) << "Working on number " << scanNumber << std::endl;

  CHECK(fs::exists(rotationFile))
      << "Could not open " << rotationFile << std::endl;
  std::ifstream binaryReader(rotationFile.string(),
                             std::ios::in | std::ios::binary);
  R = std::make_shared<std::vector<Eigen::Matrix3d>>(4);
  for (int i = 0; i < R->size(); ++i) {
    binaryReader.read(reinterpret_cast<char *>(R->at(i).data()),
                      sizeof(Eigen::Matrix3d));
  }
  binaryReader.close();

  CHECK(fs::exists(doorName)) << "Could not open " << doorName << std::endl;
  binaryReader.open(doorName.string(), std::ios::in | std::ios::binary);
  int num;
  binaryReader.read(reinterpret_cast<char *>(&num), sizeof(num));
  doors = std::make_shared<std::vector<place::Door>>(num);
  for (auto &d : *doors)
    d.loadFromFile(binaryReader);
  binaryReader.close();

  CHECK(fs::exists(fileName)) << "Could not open " << fileName << std::endl;
  binaryReader.open(fileName.string(), std::ios::in | std::ios::binary);

  int columns, rows;
  binaryReader.read(reinterpret_cast<char *>(&columns), sizeof(int));
  binaryReader.read(reinterpret_cast<char *>(&rows), sizeof(int));

  pointsWithCenter = std::make_shared<std::vector<Eigen::Vector3f>>();
  pointsWithCenter->reserve(columns * rows);
  pointsNoCenter = std::make_shared<std::vector<Eigen::Vector3f>>();
  pointsNoCenter->reserve(columns * rows);

  for (int k = 0; k < columns * rows; ++k) {
    scan::PointXYZRGBA tmp;
    tmp.loadFromFile(binaryReader);
    auto &point = tmp.point;

    point[1] *= -1.0;

    if (!(point[0] || point[1] || point[2]) || tmp.intensity < 0.01)
      continue;

    pointsWithCenter->push_back(point);

    if (point[0] * point[0] + point[1] * point[1] > 1)
      pointsNoCenter->push_back(point);
  }
  binaryReader.close();
}

bool DensityMapsManager::hasNext() {
  return current < FLAGS_numScans + FLAGS_startIndex;
}

void DensityMapsManager::setNext() { ++current; }

void DensityMapsManager::get2DPointNames(std::vector<fs::path> &names) {
  for (int r = 0; r < NUM_ROTS; ++r) {
    names.push_back(fs::path(FLAGS_dmFolder) / "R{}"_format(r) /
                    "{}_point_{}.png"_format(buildName, scanNumber));
  }
}

void DensityMapsManager::get2DFreeNames(std::vector<fs::path> &names) {
  for (int r = 0; r < NUM_ROTS; ++r) {
    names.push_back(fs::path(FLAGS_dmFolder) / "R{}"_format(r) /
                    "{}_freeSpace_{}.png"_format(buildName, scanNumber));
  }
}

void DensityMapsManager::get3DPointNames(std::vector<fs::path> &names) {
  for (int r = 0; r < NUM_ROTS; ++r) {
    names.push_back(fs::path(FLAGS_voxelFolder) / "R{}"_format(r) /
                    "{}_point_{}.dat"_format(buildName, scanNumber));
  }
}

void DensityMapsManager::get3DFreeNames(std::vector<fs::path> &names) {
  for (int r = 0; r < NUM_ROTS; ++r) {
    names.push_back(fs::path(FLAGS_voxelFolder) / "R{}"_format(r) /
                    "{}_freeSpace_{}.dat"_format(buildName, scanNumber));
  }
}

fs::path DensityMapsManager::getZerosName() {
  return fs::path(FLAGS_zerosFolder) /
         "{}_zeros_{}.dat"_format(buildName, scanNumber);
}

fs::path DensityMapsManager::getDoorsName() {
  return fs::path(FLAGS_doorsFolder) / "floorplan" /
         "{}_doors_{}.dat"_format(buildName, scanNumber);
}

fs::path DensityMapsManager::getMetaDataName() {
  return fs::path(FLAGS_voxelFolder) / "metaData" /
         "{}_metaData_{}.dat"_format(buildName, scanNumber);
}

bool DensityMapsManager::exists2D() {
  std::vector<fs::path> names;
  if (FLAGS_pe)
    get2DPointNames(names);
  if (FLAGS_fe)
    get2DFreeNames(names);
  for (auto &n : names)
    if (!fs::exists(n))
      return false;
  return true;
}

bool DensityMapsManager::exists3D() {
  std::vector<fs::path> names;
  if (FLAGS_pe)
    get3DPointNames(names);
  if (FLAGS_fe)
    get3DFreeNames(names);

  for (auto &n : names)
    if (!fs::exists(n))
      return false;
  return true;
}

bool DensityMapsManager::existsDoors() { return fs::exists(getDoorsName()); }

BoundingBox::BoundingBox(
    const std::shared_ptr<const std::vector<Eigen::Vector3f>> &points,
    Eigen::Vector3f &&range)
    : points{points}, range{range} {}

BoundingBox::BoundingBox(
    const std::shared_ptr<const std::vector<Eigen::Vector3f>> &points,
    Eigen::Vector3f &range)
    : points{points}, range{range} {}

void BoundingBox::run() {
  average = Eigen::Vector3f::Zero();
  sigma = Eigen::Vector3f::Zero();

  for (auto &point : *points)
    average += point;

  average /= points->size();

  for (auto &point : *points)
    for (int i = 0; i < 3; ++i)
      sigma[i] += (point[i] - average[i]) * (point[i] - average[i]);

  sigma /= points->size() - 1;
  for (int i = 0; i < 3; ++i)
    sigma[i] = sqrt(sigma[i]);
}

void BoundingBox::setRange(Eigen::Vector3f &&range) { this->range = range; }

void BoundingBox::setRange(Eigen::Vector3f &range) { this->range = range; }

void BoundingBox::getBoundingBox(Eigen::Vector3f &min,
                                 Eigen::Vector3f &max) const {
  Eigen::Vector3f delta;
  for (int i = 0; i < delta.size(); ++i)
    delta[i] = 1.1 * range[i] * sigma[i];

  min = average - delta / 2.0;
  max = average + delta / 2.0;
}

CloudAnalyzer2D::CloudAnalyzer2D(
    const std::shared_ptr<const std::vector<Eigen::Vector3f>> &points,
    const std::shared_ptr<const std::vector<Eigen::Matrix3d>> &R,
    const std::shared_ptr<const BoundingBox> &bBox,
    const DensityMapsManager::DoorsPtr &doors)
    : points{points}, R{R}, bBox{bBox}, doors{doors} {}

void CloudAnalyzer2D::initalize(double scale) {
  bBox->getBoundingBox(pointMin, pointMax);

  zScale = (float)numZ / (pointMax[2] - pointMin[2]);

  numX = scale * (pointMax[0] - pointMin[0]);
  numY = scale * (pointMax[1] - pointMin[1]);

  pointInVoxel = voxel::DirectVoxel<char>::Create(numX, numY, numZ);

  for (auto &point : *points) {
    const int x = scale * (point[0] - pointMin[0]);
    const int y = scale * (point[1] - pointMin[1]);
    const int z = zScale * (point[2] - pointMin[2]);

    if (x < 0 || x >= numX)
      continue;
    if (y < 0 || y >= numY)
      continue;
    if (z < 0 || z >= numZ)
      continue;

    pointInVoxel->at(x, y, z) = 1;
  }

  zeroZero = Eigen::Vector3d(-pointMin[0] * FLAGS_scale,
                             -pointMin[1] * FLAGS_scale, -pointMin[2] * zScale);

  newRows = sqrt(2) * std::max(numY, numX);
  newCols = newRows;
  int dX = (newCols - numX) / 2.0;
  int dY = (newRows - numY) / 2.0;
  newZZ = zeroZero;
  newZZ[0] += dX;
  newZZ[1] += dY;
  imageZeroZero = Eigen::Vector2i(newZZ[0], newZZ[1]);
}

void CloudAnalyzer2D::examinePointEvidence() {
  pointEvidence.clear();

  for (int r = 0; r < R->size(); ++r) {
    Eigen::MatrixXf total = Eigen::MatrixXf::Zero(newRows, newCols);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < total.cols(); ++i) {
      for (int j = 0; j < total.rows(); ++j) {
        for (int k = 0; k < numZ; ++k) {
          const Eigen::Vector3d coord(i, j, k);
          const Eigen::Vector3i src =
              (R->at(r) * (coord - newZZ) + zeroZero)
                  .unaryExpr([](auto v) { return std::round(v); })
                  .cast<int>();

          if (src[0] < 0 || src[0] >= numX || src[1] < 0 || src[1] >= numY ||
              src[2] < 0 || src[2] >= numZ)
            continue;

          if (pointInVoxel->at(src))
            ++total(j, i);
        }
      }
    }

    const float *dataPtr = total.data();
    auto[average, sigma] = utils::ave_and_stdev(
        dataPtr, dataPtr + total.size(), 0.0, [](auto v) { return v; },
        [](auto v) -> bool { return v; });

    cv::Mat heatMap(newRows, newCols, CV_8UC1, cv::Scalar::all(255));
    for (int j = 0; j < heatMap.rows; ++j) {
      uchar *dst = heatMap.ptr<uchar>(j);
      for (int i = 0; i < heatMap.cols; ++i) {
        const double count = total(j, i);
        if (count > 0) {
          const int gray = cv::saturate_cast<uchar>(
              255.0 *
              ((count - average - sigma) / (3.0 * sigma) - 0.0 * sigma));
          dst[i] = 255 - gray;
        }
      }
    }
    pointEvidence.push_back(heatMap);
  }
}

void CloudAnalyzer2D::examineFreeSpaceEvidence() {
  freeSpaceEvidence.clear();
  Eigen::Vector3f cameraCenter = -1.0 * pointMin;

  voxel::DirectVoxel<char> freeSpace(numX, numY, numZ);

  for (int k = 0; k < numZ; ++k) {
    for (int j = 0; j < numY; ++j) {
      for (int i = 0; i < numX; ++i) {

        if (!pointInVoxel->at(i, j, k))
          continue;

        Eigen::Vector3d ray(i, j, k);
        ray[0] -= cameraCenter[0] * FLAGS_scale;
        ray[1] -= cameraCenter[1] * FLAGS_scale;
        ray[2] -= cameraCenter[2] * zScale;
        double length = ray.norm();
        Eigen::Vector3d unitRay = ray / length;

        Eigen::Vector3i voxelHit;
        for (int a = 0; a <= ceil(length); ++a) {
          voxelHit[0] = floor(cameraCenter[0] * FLAGS_scale + a * unitRay[0]);
          voxelHit[1] = floor(cameraCenter[1] * FLAGS_scale + a * unitRay[1]);
          voxelHit[2] = floor(cameraCenter[2] * zScale + a * unitRay[2]);

          if (voxelHit[0] < 0 || voxelHit[0] >= numX)
            continue;
          if (voxelHit[1] < 0 || voxelHit[1] >= numY)
            continue;
          if (voxelHit[2] < 0 || voxelHit[2] >= numZ)
            continue;

          freeSpace(voxelHit) = 1;
        }
      }
    }
  }

  for (int r = 0; r < R->size(); ++r) {
    Eigen::MatrixXd collapsedCount = Eigen::MatrixXd::Zero(newRows, newCols);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < collapsedCount.cols(); ++i) {
      for (int j = 0; j < collapsedCount.rows(); ++j) {
        for (int k = 0; k < numZ; ++k) {
          const Eigen::Vector3d coord(i, j, k);
          const Eigen::Vector3i src =
              (R->at(r) * (coord - newZZ) + zeroZero)
                  .unaryExpr([](auto v) { return std::round(v); })
                  .cast<int>();

          if (src[0] < 0 || src[0] >= numX || src[1] < 0 || src[1] >= numY ||
              src[2] < 0 || src[2] >= numZ)
            continue;

          if (freeSpace(src))
            ++collapsedCount(j, i);
        }
      }
    }

    const double *vPtr = collapsedCount.data();
    auto[average, sigma] = utils::ave_and_stdev(
        vPtr, vPtr + collapsedCount.size(), 0.0, [](double v) { return v; },
        [](double v) -> bool { return v; });

    cv::Mat heatMap(newRows, newCols, CV_8UC1, cv::Scalar::all(255));
    for (int j = 0; j < heatMap.rows; ++j) {
      uchar *dst = heatMap.ptr<uchar>(j);
      for (int i = 0; i < heatMap.cols; ++i) {
        const double count = collapsedCount(j, i);
        if (count > 0) {
          const int gray = cv::saturate_cast<uchar>(
              255.0 * ((count - average) / sigma + 1.0));
          dst[i] = 255 - gray;
        }
      }
    }
    const double radius = 0.3;
    for (int j = -sqrt(radius) * FLAGS_scale; j < sqrt(radius) * FLAGS_scale;
         ++j) {
      uchar *dst = heatMap.ptr<uchar>(j + imageZeroZero[1]);
      for (int i = -sqrt(radius * FLAGS_scale * FLAGS_scale - j * j);
           i < sqrt(radius * FLAGS_scale * FLAGS_scale - j * j); ++i) {
        dst[i + imageZeroZero[0]] = 0;
      }
    }
    freeSpaceEvidence.push_back(heatMap);
  }
}

void CloudAnalyzer2D::rotateDoors() {
  for (int r = 0; r < NUM_ROTS; ++r) {
    std::vector<place::Door> v;
    for (const auto &d : *doors) {
      place::Door newD(d);
      newD.corner[1] *= -1;
      newD.corner = R->at(r).inverse() * newD.corner;

      newD.xAxis[1] *= -1.0;
      newD.xAxis = R->at(r).inverse() * newD.xAxis;

      newD.zAxis[1] *= -1.0;
      newD.zAxis = R->at(r).inverse() * newD.zAxis;

      v.emplace_back(newD);
    }
    rotatedDoors.emplace_back(v);
  }
}

const std::vector<cv::Mat> &CloudAnalyzer2D::getPointEvidence() {
  return pointEvidence;
}

const std::vector<cv::Mat> &CloudAnalyzer2D::getFreeSpaceEvidence() {
  return freeSpaceEvidence;
}

Eigen::Vector2i CloudAnalyzer2D::getImageZeroZero() { return imageZeroZero; }

const std::vector<std::vector<place::Door>> CloudAnalyzer2D::getRotatedDoors() {
  return rotatedDoors;
}
