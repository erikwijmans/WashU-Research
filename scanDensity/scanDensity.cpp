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

DensityMapsManager::DensityMapsManager (const std::string & commandLine):
  R {NULL},
  pointsWithCenter {NULL},
  pointsNoCenter {NULL}
{
  this->resetFlags(commandLine);
}

DensityMapsManager::DensityMapsManager(int argc, char * argv[]):
  R {NULL},
  pointsWithCenter {NULL},
  pointsNoCenter {NULL}
{
  this->resetFlags(argc, argv);
}


void DensityMapsManager::resetFlags(const std::string & commandLine) {
  std::vector<std::string> v;
  std::istringstream is (commandLine);
  std::string tmp;
  while(is >> tmp)
    v.push_back(tmp);
  char ** argv = new char * [v.size() + 1];
  for (int i = 0; i < v.size(); ++i) {
    argv[i] = &v[i][0];
  }
  argv[v.size()] = NULL;
  resetFlags(v.size() + 1, argv);
  delete [] argv;
}

void DensityMapsManager::resetFlags(int argc, char * argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  prependDataPath();

  if (!FLAGS_2D && !FLAGS_3D)
    FLAGS_2D = FLAGS_3D = true;

  if(!FLAGS_pe && !FLAGS_fe)
    FLAGS_pe = FLAGS_fe = true;

  DIR *dir;
  struct dirent *ent;
  if ((dir = opendir (FLAGS_binaryFolder.data())) != NULL) {
    /* Add all the files and directories to a std::vector */
    while ((ent = readdir (dir)) != NULL) {
      std::string fileName = ent->d_name;
      if (fileName != ".." && fileName != "."){
        binaryNames.push_back(fileName);
      }
    }
    closedir (dir);
  }  else {
    /* could not open directory */
    perror ("");
    exit(EXIT_FAILURE);
  }
  sort(binaryNames.begin(), binaryNames.end());

  if ((dir = opendir (FLAGS_rotFolder.data())) != NULL) {
    /* Add all the files and directories to a std::vector */
    while ((ent = readdir (dir)) != NULL) {
      std::string fileName = ent->d_name;
      if (fileName != ".." && fileName != "."){
        rotationsFiles.push_back(fileName);
      }
    }
    closedir (dir);
  }  else {
    /* could not open directory */
    perror ("");
    exit(EXIT_FAILURE);
  }
  sort(rotationsFiles.begin(), rotationsFiles.end());

  if (binaryNames.size() != rotationsFiles.size()) {
    std::cout << "Not the same number of binaryFiles as rotationsFiles" << std::endl;
    exit(1);
  }

  std::string buildName = rotationsFiles[0].substr(0, 3);

  if (FLAGS_scale == -1) {
    auto it = buildingToScale.find(buildName);
    if (it == buildingToScale.end()) {
      std::cout << "Could not find a scale assocaited with " << buildName << std::endl;
      exit(1);
    }
    FLAGS_scale = buildingToScale.find(buildName)->second;
  }

  if (FLAGS_numScans == -1)
    FLAGS_numScans = binaryNames.size() - FLAGS_startIndex;
  this->current = FLAGS_startIndex;
}

void DensityMapsManager::run() {
  rotationFile = FLAGS_rotFolder + rotationsFiles[current];
  fileName = FLAGS_binaryFolder + binaryNames[current];

  scanNumber = fileName.substr(fileName.find(".") - 3, 3);
  buildName = fileName.substr(fileName.rfind("/") + 1, 3);

  if (!FLAGS_redo && exists2D() && exists3D())
    return;

  if (!FLAGS_quietMode)
    std::cout << scanNumber << std::endl;

  std::ifstream binaryReader (rotationFile, std::ios::in | std::ios::binary);
  R = std::make_shared<std::vector<Eigen::Matrix3d> > (4);
  for (int i = 0; i < R->size(); ++i) {
    binaryReader.read(reinterpret_cast<char *>(R->at(i).data()),
      sizeof(Eigen::Matrix3d));
  }
  binaryReader.close();

  binaryReader.open(fileName, std::ios::in | std::ios::binary);

  int columns, rows;
  binaryReader.read(reinterpret_cast<char *> (& columns), sizeof(int));
  binaryReader.read(reinterpret_cast<char *> (& rows), sizeof(int));

  pointsWithCenter = std::make_shared<std::vector<Eigen::Vector3f> > ();
  pointsWithCenter->reserve(columns*rows);
  pointsNoCenter = std::make_shared<std::vector<Eigen::Vector3f> > ();
  pointsNoCenter->reserve(columns*rows);

  for (int k = 0; k < columns * rows; ++k) {
    scan::PointXYZRGBA tmp;
    tmp.loadFromFile(binaryReader);
    auto & point = tmp.point;

    point[1] *= -1.0;

    if (!(point[0] || point[1] || point[2]) || tmp.intensity < 0.2)
      continue;

    pointsWithCenter->push_back(point);

    if (point[0]*point[0] + point[1]*point[1] > 1)
      pointsNoCenter->push_back(point);
  }
  binaryReader.close();
}

bool DensityMapsManager::hasNext() {
  return current < FLAGS_numScans + FLAGS_startIndex;
}

void DensityMapsManager::setNext() {
  ++current;
}

void DensityMapsManager::get2DPointNames(std::vector<std::string> & names) {
  for (int r = 0; r < NUM_ROTS; ++r) {
    names.push_back(FLAGS_dmFolder + "R" + std::to_string(r)
      + "/" + buildName + "_point_" + scanNumber + ".png");
  }
}

void DensityMapsManager::get3DPointNames(std::vector<std::string> & names) {
  for (int r = 0; r < NUM_ROTS; ++r) {
    names.push_back(FLAGS_voxelFolder + "R" + std::to_string(r)
      + "/" + buildName + "_point_" + scanNumber + ".dat");
  }
}

void DensityMapsManager::get2DFreeNames(std::vector<std::string> & names) {
  for (int r = 0; r < NUM_ROTS; ++r) {
    names.push_back(FLAGS_dmFolder + "R" + std::to_string(r)
      + "/" + buildName + "_freeSpace_" + scanNumber + ".png");
  }
}

void DensityMapsManager::get3DFreeNames(std::vector<std::string> & names) {
  for (int r = 0; r < NUM_ROTS; ++r) {
    names.push_back(FLAGS_voxelFolder + "R" + std::to_string(r)
      + "/" + buildName + "_freeSpace_" + scanNumber + ".dat");
  }
}

std::string DensityMapsManager::getZerosName() {
  return FLAGS_zerosFolder + buildName + "_zeros_" + scanNumber + ".dat";
}

std::string DensityMapsManager::getMetaDataName() {
  return FLAGS_voxelFolder + "metaData/" + buildName + "_metaData_"
    + scanNumber + ".dat";
}

static bool fexists(const std::string & name) {
  std::ifstream in (name, std::ios::in);
  return in.is_open();
}

bool DensityMapsManager::exists2D() {
  std::vector<std::string> names;
  if (FLAGS_pe)
    get2DPointNames(names);
  if (FLAGS_fe)
    get2DFreeNames(names);
  for (auto & n : names)
    if (!fexists(n)) return false;
  return true;
}

bool DensityMapsManager::exists3D() {
  std::vector<std::string> names;
  if (FLAGS_pe)
    get3DPointNames(names);
  if (FLAGS_fe)
    get3DFreeNames(names);

  for (auto & n : names)
    if (!fexists(n)) return false;
  return true;
}

BoundingBox::BoundingBox(const std::shared_ptr<const std::vector<Eigen::Vector3f> > & points,
  Eigen::Vector3f && range) :
  points {points},
  range {range}
{
}

BoundingBox::BoundingBox(const std::shared_ptr<const std::vector<Eigen::Vector3f> > & points,
  Eigen::Vector3f & range) :
  points {points},
  range {range}
{
}

void BoundingBox::run() {
  this->average = Eigen::Vector3f::Zero();
  this->sigma = Eigen::Vector3f::Zero();
  for (auto & point : *points) {
    average += point;
  }
  average /= points->size();

  for (auto & point : *points)
    for (int i = 0; i < 3; ++i)
      sigma[i] += (point[i] - average[i])*(point[i] - average[i]);

  sigma /= points->size() - 1;
  for (int i = 0; i < 3; ++i)
    sigma[i] = sqrt(sigma[i]);
}

void BoundingBox::setRange(Eigen::Vector3f && range) {
  this->range = range;
}

void BoundingBox::setRange(Eigen::Vector3f & range) {
  this->range = range;
}

void BoundingBox::getBoundingBox(Eigen::Vector3f & min,
  Eigen::Vector3f & max) const {

  Eigen::Vector3f delta;
  for (int i = 0; i < delta.size(); ++i)
    delta[i] = 1.1*range[i]*sigma[i];

  min = average - delta/2.0;
  max = average + delta/2.0;
}


CloudAnalyzer2D::CloudAnalyzer2D(const std::shared_ptr<const std::vector<Eigen::Vector3f> > & points,
  const std::shared_ptr<const std::vector<Eigen::Matrix3d> > & R,
  const std::shared_ptr<const BoundingBox> & bBox) :
  points {points},
  R {R},
  bBox {bBox},
  pointsPerVoxel {nullptr}
{
  cvNamedWindow("Preview", CV_WINDOW_NORMAL);
}

void CloudAnalyzer2D::initalize(double scale) {
  bBox->getBoundingBox(pointMin, pointMax);


  zScale = (float)numZ/(pointMax[2] - pointMin[2]);

  numX = scale * (pointMax[0] - pointMin[0]);
  numY = scale * (pointMax[1] - pointMin[1]);

  pointsPerVoxel =
    voxel::HashVoxel<Eigen::Vector2i, Eigen::VectorXi>::Create(Eigen::Vector2i(0, 0),
      Eigen::Vector2i(numX, numY));

  for (auto & point : *points){
    const int x = scale*(point[0] - pointMin[0]);
    const int y = scale*(point[1] - pointMin[1]);
    const int z = zScale*(point[2] - pointMin[2]);

    if (x < 0 || x >= numX)
      continue;
    if (y < 0 || y >= numY)
      continue;
    if ( z < 0 || z >= numZ)
      continue;

    auto p = pointsPerVoxel->at(x, y);
    if (!p)
      p = pointsPerVoxel->insert(Eigen::VectorXi::Zero(numZ), x, y);

    ++(*p)[z];

  }

  zeroZero = Eigen::Vector3d(-pointMin[0]*FLAGS_scale, -pointMin[1]*FLAGS_scale, 0);
}

void CloudAnalyzer2D::examinePointEvidence() {
  pointEvidence.clear();
  Eigen::MatrixXf total = Eigen::MatrixXf::Zero (numY, numX);
  for (int i = 0; i < numX; ++i) {
    for (int j = 0; j < numY; ++j) {
      auto column = pointsPerVoxel->at(i, j);
      if (column) {
        for (int k = 0; k < numZ; ++k) {
          if ((*column)[k])
            ++total(j, i);
        }
      }
    }
  }

  double average, sigma;
  average = sigma = 0;
  int count = 0;
  const float * dataPtr = total.data();
  for (int i = 0; i < total.size(); ++i) {
    if (*(dataPtr+ i)) {
      ++count;
      average+= *(dataPtr + i);
    }
  }

  average = average/count;

  for (int i = 0; i < total.size(); ++i) {
    if (*(dataPtr + i) !=0)
      sigma += (*(dataPtr + i) - average)*(*(dataPtr + i)- average);
  }

  sigma = sigma/(count-1);
  sigma = sqrt(sigma);

  int newRows = sqrt(2)*std::max(total.rows(), total.cols());
  int newCols = newRows;
  int dX = (newCols - total.cols())/2.0;
  int dY = (newRows - total.rows())/2.0;
  Eigen::Vector3d newZZ = zeroZero;
  newZZ[0] += dX;
  newZZ[1] += dY;

  imageZeroZero = Eigen::Vector2i(newZZ[0], newZZ[1]);

  for (int r = 0; r < R->size(); ++r) {
    cv::Mat heatMap  (newRows, newCols, CV_8UC1, cv::Scalar::all(255));
    for (int j = 0; j < heatMap.rows; ++j) {
      uchar * dst = heatMap.ptr<uchar>(j);
      for (int i = 0; i < heatMap.cols; ++i) {
        const Eigen::Vector3d pixel (i, j, 0);
        const Eigen::Vector3d src = R->at(r)*(pixel - newZZ) + zeroZero;

        if (src[0] < 0 || src[0] >= total.cols())
          continue;
        if (src[1] < 0 || src[1] >= total.rows())
          continue;

        const double count = total(src[1], src[0]);
        if (count > 0) {
          const int gray = cv::saturate_cast<uchar>(
            255.0 * (count - average - sigma)
              / (3.0 * sigma));
          dst[i] = 255 - gray;
        }
      }
    }
    if (FLAGS_preview) {
      cv::imshow("Preview", heatMap);
      cv::waitKey(0);
    }

    pointEvidence.push_back(heatMap);
  }
}

void CloudAnalyzer2D::examineFreeSpaceEvidence() {
  freeSpaceEvidence.clear();
  Eigen::Vector3f cameraCenter = -1.0*pointMin;

  //voxel::HashVoxel<Eigen::Vector2i, Eigen::VectorXi> numTimesSeen (
    //Eigen::Vector2i(0, 0), Eigen::Vector2i(numX, numY));

  std::vector<Eigen::MatrixXi> numTimesSeen (numX, Eigen::MatrixXi::Zero(numZ, numY));

  for (int i = 0; i < numX; ++i) {
    for (int j = 0; j < numY; ++j) {
      auto column = pointsPerVoxel->at(i, j);
      if (!column)
        continue;
      for (int k = 0; k < numZ; ++k) {
        if (!(*column)[k])
          continue;

        float ray[3];
        ray[0] = i - cameraCenter[0]*FLAGS_scale;
        ray[1] = j - cameraCenter[1]*FLAGS_scale;
        ray[2] = k - cameraCenter[2]*zScale;
        float length = sqrt(ray[0]*ray[0] + ray[1]*ray[1] + ray[2]*ray[2]);
        float unitRay[3];
        unitRay[0] = ray[0]/length;
        unitRay[1] = ray[1]/length;
        unitRay[2] = ray[2]/length;
        int voxelHit [3];
        for (int a = 0; a <= ceil(length); ++a) {

          voxelHit[0] = floor(cameraCenter[0]*FLAGS_scale + a*unitRay[0]);
          voxelHit[1] = floor(cameraCenter[1]*FLAGS_scale + a*unitRay[1]);
          voxelHit[2] = floor(cameraCenter[2]*zScale + a*unitRay[2]);

          if (voxelHit[0] < 0 || voxelHit[0] >= numX)
            continue;
          if (voxelHit[1] < 0 || voxelHit[1] >= numY)
            continue;
          if (voxelHit[2] < 0 || voxelHit[2] >= numZ)
            continue;
          /*auto n = numTimesSeen(voxelHit[0], voxelHit[1]);
          if (!n) {
            n = numTimesSeen.insert(std::make_shared<Eigen::VectorXi>
              (Eigen::VectorXi::Zero(numZ)), voxelHit[0], voxelHit[1]);
          }

          (*n)[voxelHit[2]] +=
            (*column)[k];*/

          numTimesSeen[voxelHit[0]](voxelHit[2], voxelHit[1])
            += (*column)[k];
        }
      }
    }
  }

  Eigen::MatrixXd collapsedCount = Eigen::MatrixXd::Zero(numY, numX);

  for (int i = 0; i < numX; ++i) {
    for (int j = 0; j < numY; ++j) {
     // auto column = numTimesSeen(i, j);
      bool column = true;
      if (column) {
        for (int k = 0; k < numZ; ++k) {
          if (numTimesSeen[i](k,j)/*(*column)[k]*/) {
            ++collapsedCount(j, i);
          }
        }
      }
    }
  }

  double average, sigma;
  average = sigma = 0;
  size_t count = 0;
  const double * vPtr = collapsedCount.data();

  for (int i = 0; i < collapsedCount.size(); ++i) {
    if (*(vPtr + i) != 0) {
      average += *(vPtr + i);
      ++count;
    }
  }

  average = average/count;

  for (int i = 0; i < collapsedCount.size(); ++i){
    if (*(vPtr + i)!=0)
      sigma += (*(vPtr + i)-average)*(*(vPtr + i)-average);
  }
  sigma = sigma/(count - 1);
  sigma = sqrt(sigma);

  int newRows = sqrt(2)*std::max(collapsedCount.rows(), collapsedCount.cols());
  int newCols = newRows;
  int dX = (newCols - collapsedCount.cols())/2.0;
  int dY = (newRows - collapsedCount.rows())/2.0;
  Eigen::Vector3d newZZ = zeroZero;
  newZZ[0] += dX;
  newZZ[1] += dY;

  imageZeroZero = Eigen::Vector2i(newZZ[0], newZZ[1]);

  for (int r = 0; r < R->size(); ++r) {
    cv::Mat heatMap (newRows, newCols, CV_8UC1, cv::Scalar::all(255));
    for (int j = 0; j < heatMap.rows; ++j) {
      uchar * dst = heatMap.ptr<uchar>(j);
      for (int i = 0; i < heatMap.cols; ++i) {
        const Eigen::Vector3d pixel (i, j, 0);
        const Eigen::Vector3d src = R->at(r)*(pixel - newZZ) + zeroZero;

        if (src[0] < 0 || src[0] >= collapsedCount.cols())
          continue;
        if (src[1] < 0 || src[1] >= collapsedCount.rows())
          continue;

        const double count = collapsedCount(src[1], src[0]);
        if (count > 0) {
          const int gray = cv::saturate_cast<uchar>(
            255.0 * ((count - average)/sigma + 1.0 ));
          dst[i] = 255 - gray;
        }
      }
    }
    const double radius = 0.3;
    for (int j = -sqrt(radius)*FLAGS_scale; j < sqrt(radius)*FLAGS_scale; ++j) {
      uchar * dst = heatMap.ptr<uchar>(j + imageZeroZero[1]);
      for (int i = -sqrt(radius*FLAGS_scale*FLAGS_scale - j*j);
        i < sqrt(radius*FLAGS_scale*FLAGS_scale - j*j); ++i) {
        dst[i + imageZeroZero[0]] = 0;
      }
    }

    if (FLAGS_preview) {
      cv::imshow("Preview", heatMap);
      cv::waitKey(0);
    }

    freeSpaceEvidence.push_back(heatMap);
  }
}

const std::vector<cv::Mat> & CloudAnalyzer2D::getPointEvidence() {
  return pointEvidence;
}

const std::vector<cv::Mat> & CloudAnalyzer2D::getFreeSpaceEvidence() {
  return freeSpaceEvidence;
}

Eigen::Vector2i CloudAnalyzer2D::getImageZeroZero() {
  return imageZeroZero;
}