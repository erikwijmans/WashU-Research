#include "scanDensity_3DInfo.h"
#include <list>

voxel::CloudAnalyzer3D::CloudAnalyzer3D(const std::shared_ptr<const std::vector<Eigen::Vector3f> > & points,
  const std::shared_ptr<const std::vector<Eigen::Matrix3d> > & R,
  const std::shared_ptr<const BoundingBox> & bBox) :
  points {points},
  R {R},
  bBox {bBox}
{
  cvNamedWindow("Preview", CV_WINDOW_NORMAL);
}

void voxel::CloudAnalyzer3D::run(double voxelsPerMeter,
  double pixelsPerMeter) {

  bBox->getBoundingBox(pointMin, pointMax);
  this->voxelsPerMeter = voxelsPerMeter;
  this->pixelsPerMeter = pixelsPerMeter;

  const int numX = voxelsPerMeter * (pointMax[0] - pointMin[0]);
  const int numY = voxelsPerMeter * (pointMax[1] - pointMin[1]);

  const float zScale = voxelsPerMeter;
  const int numZ = zScale * (pointMax[2] - pointMin[2]);

  pointsPerVoxel.assign(numZ, Eigen::MatrixXi::Zero(numY, numX));

  for(auto & point : *points){
    const int x = voxelsPerMeter*(point[0] - pointMin[0]);
    const int y = voxelsPerMeter*(point[1] - pointMin[1]);
    const int z = zScale*(point[2] -pointMin[2]);

    if(x <0 || x >= numX)
      continue;
    if(y < 0 || y >= numY)
      continue;
    if( z < 0 || z >= numZ)
      continue;

    ++pointsPerVoxel[z](y, x);
  }

  //Free space evidence

  float cameraCenter [3];
  cameraCenter[0] = -1*pointMin[0];
  cameraCenter[1] = -1*pointMin[1];
  cameraCenter[2] = -1*pointMin[2];
  numTimesSeen.assign(numZ, Eigen::MatrixXi::Zero(numY, numX));

  for (int k = 0; k < numZ; ++k) {
    for (int i = 0; i < numX; ++i) {
      for (int j = 0; j < numY; ++j) {
        if(!pointsPerVoxel[k](j,i))
          continue;

        float ray[3];
        ray[0] = i - cameraCenter[0]*voxelsPerMeter;
        ray[1] = j - cameraCenter[1]*voxelsPerMeter;
        ray[2] = k - cameraCenter[2]*zScale;
        float length = sqrt(ray[0]*ray[0] + ray[1]*ray[1] + ray[2]*ray[2]);
        float unitRay[3];
        unitRay[0] = ray[0]/length;
        unitRay[1] = ray[1]/length;
        unitRay[2] = ray[2]/length;
        int stop = floor(0.95*length - 3);
        int voxelHit [3];
        for (int a = 0; a < stop; ++a) {

          voxelHit[0] = floor(cameraCenter[0]*voxelsPerMeter + a*unitRay[0]);
          voxelHit[1] = floor(cameraCenter[1]*voxelsPerMeter + a*unitRay[1]);
          voxelHit[2] = floor(cameraCenter[2]*zScale + a*unitRay[2]);

          if(voxelHit[0] < 0 || voxelHit[0] >= numX)
            continue;
          if(voxelHit[1] < 0 || voxelHit[1] >= numY)
            continue;
          if(voxelHit[2] < 0 || voxelHit[2] >= numZ)
            continue;

          numTimesSeen[voxelHit[2]](voxelHit[1], voxelHit[0])
            += pointsPerVoxel[k](j,i);
        }
      }
    }
  }

  zeroZeroD = Eigen::Vector3d(-pointMin[0]*voxelsPerMeter,
    -pointMin[1]*voxelsPerMeter, 0);
  zeroZero = Eigen::Vector3i(-pointMin[0]*voxelsPerMeter,
    -pointMin[1]*voxelsPerMeter, -pointMin[2]*zScale);

  /*std::unordered_map<Eigen::Vector3i, int> xyzToCount;
  for (auto& feature : *featureVectors) {
    auto point = feature.position;
    point[1] *= -1.0;
    const int x = voxelsPerMeter*(point[0] - pointMin[0]);
    const int y = voxelsPerMeter*(point[1] - pointMin[1]);
    const int z = zScale*(point[2] -pointMin[2]);

    if(x <0 || x >= numX)
      continue;
    if(y < 0 || y >= numY)
      continue;
    if( z < 0 || z >= numZ)
      continue;

    Eigen::Vector3i position (x, y, z);
    auto it = xyzToSHOT.find(position);
    if (it == xyzToSHOT.end()) {
      xyzToSHOT.emplace(position, feature.descriptor);
      xyzToCount.emplace(position, 1);
    } else {
      *it->second += *feature.descriptor;
      ++xyzToCount.find(position)->second;
    }
  }
  for (auto feat : xyzToSHOT) {
    *feat.second /= xyzToCount.find(feat.first)->second;
  }*/
}

static void displayVoxelGrid(const auto & voxelB) {

  Eigen::MatrixXd collapsed = Eigen::MatrixXd::Zero (voxelB[0].rows(), voxelB[0].cols());

  for (int k = 0; k < voxelB.size(); ++k)
    for (int i = 0; i < voxelB[0].cols(); ++i)
      for (int j = 0; j < voxelB[0].rows(); ++j)
        collapsed(j, i) += voxelB[k](j,i) ? 1 : 0;

  double average, sigma;
  average = sigma = 0;
  int count = 0;
  const double * dataPtr = collapsed.data();
  for(int i = 0; i < collapsed.size(); ++i) {
    if(*(dataPtr+ i)) {
      ++count;
      average+= *(dataPtr + i);
    }
  }

  average = average/count;

  for(int i = 0; i < collapsed.size(); ++i)
    if(*(dataPtr + i) !=0 )
      sigma += (*(dataPtr + i) - average)*(*(dataPtr + i)- average);

  sigma = sigma/(count-1);
  sigma = sqrt(sigma);

  cv::Mat heatMap (collapsed.rows(), collapsed.cols(), CV_8UC3, cv::Scalar::all(255));
  for (int i = 0; i < heatMap.rows; ++i) {
    uchar * dst = heatMap.ptr<uchar>(i);
    for (int j = 0; j < heatMap.cols; ++j) {
      if(collapsed(i,j)){
        const int gray = cv::saturate_cast<uchar>(
          255.0 * (collapsed(i,j) - average)
            / (1.0 * sigma));
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
        dst[j*3] = blue;
        dst[j*3 +1] = green;
        dst[j*3 + 2] = red;
      }
    }
  }

  cv::imshow("Preview", heatMap);
  cv::waitKey(0);
}

template<typename Scalar>
static void labelNeigbors(const voxel::FeatureVoxel<Scalar> & features,
  typename voxel::FeatureVoxel<Scalar>::DescripPtr & currentDescrip,
  int ID, std::list<Eigen::Vector3i> & toLabel,
  voxel::FeatureVoxel<Scalar> & minimalFeatures, int & numCondensed) {

  if (toLabel.empty()) return;

  auto current = toLabel.front();
  toLabel.pop_front();
  int x = current[0];
  int y = current[1];
  int z = current[2];
  for (int k = -1; k <= 1; ++k) {
    for (int i = -1; i <= 1; ++i) {
      for (int j = -1; j <= 1; ++j) {
        auto descriptor = features.getFeatureVector(x + i, y + j, z + k);
        if (descriptor && !minimalFeatures[z + k](y + j, x + i)) {
          if (acos(currentDescrip->dot(*descriptor)/
              (currentDescrip->norm()*descriptor->norm())) < 0.2) {
            minimalFeatures[z + k](y + j, x + i) = ID;
            toLabel.push_front(Eigen::Vector3i(x + i, y + j, z + k));
            *currentDescrip += *descriptor;
            ++numCondensed;
          }
        }
      }
    }
  }

  labelNeigbors(features, currentDescrip, ID, toLabel, minimalFeatures, numCondensed);
}

static void createMinimalFeatures(const voxel::FeatureVoxel<auto> & features,
  voxel::FeatureVoxel<auto> & minimalFeatures) {

  minimalFeatures.setVoxelGrid(std::vector<Eigen::MatrixXi> (features.getNumZ(),
    Eigen::MatrixXi::Zero(features.getNumY(), features.getNumX())));

  std::list<Eigen::Vector3i> toLabel;

  for (int k = 0; k < features.getNumZ(); ++k) {
    for (int i = 0; i < features.getNumX(); ++i) {
      for (int j = 0; j < features.getNumY(); ++j) {
        auto currentDescrip = features.getFeatureVector(i, j, k);
        if (currentDescrip && !minimalFeatures[k](j,i)) {
          int currentID = minimalFeatures.addFeatureVector(i, j, k,
            currentDescrip);
          int numCondensed = 1;
          toLabel.push_front(Eigen::Vector3i (i, j, k));
          labelNeigbors(features, currentDescrip, currentID, toLabel, minimalFeatures, numCondensed);
          *currentDescrip /= numCondensed;
        } else if (features[k](j, i) && !minimalFeatures[k](j, i))
          minimalFeatures[k](j, i) = features[k](j, i);
      }
    }
  }
}

/* This method is destructive to the things created by run */
void voxel::CloudAnalyzer3D::saveVoxelGrids(const std::vector<std::string> & pointNames,
    const std::vector<std::string> & freeNames,
    const std::string & metaData) {
  std::vector<Eigen::MatrixXi> & pointGrid = pointsPerVoxel;
  std::vector<Eigen::MatrixXi> & freeSpace = numTimesSeen;

  const int z = pointGrid.size();
  const int y = pointGrid[0].rows();
  const int x = pointGrid[0].cols();

  double averageP = 0.0, averageF = 0.0;
  int countP = 0, countF = 0;

  for(int i = 0; i < z; ++i) {
    const int * dataPtr = pointGrid[i].data();
    const int * fPtr = freeSpace[i].data();
    for(int j = 0; j < pointGrid[i].size(); ++j) {
      const int value = *(dataPtr + j);
      if(value) {
        averageP += value;
        ++countP;
      }
      if(*(fPtr + j)) {
        averageF += *(fPtr + j);
        ++countF;
      }
    }
  }
  averageP /= countP;
  averageF /= countF;
  double sigmaP = 0.0, sigmaF = 0.0;
  for(int i = 0; i < z; ++i) {
    const int * dataPtr = pointGrid[i].data();
    const int * fPtr = freeSpace[i].data();
    for(int j = 0; j < pointGrid[i].size(); ++j) {
      const int value = *(dataPtr + j);
      if(value)
        sigmaP += (value - averageP)*(value - averageP);
      if(*(fPtr + j))
        sigmaF += (*(fPtr + j) - averageF)*(*(fPtr + j) - averageF);
    }
  }
  sigmaP /= countP - 1;
  sigmaP = sqrt(sigmaP);
  sigmaF /= countF - 1;
  sigmaF = sqrt(sigmaF);

  std::vector<Eigen::MatrixXi> threshHoldedPoint (z, Eigen::MatrixXi::Zero(y, x));
  std::vector<Eigen::MatrixXb> threshHoldedFree (z, Eigen::MatrixXb::Zero(y,x));
  size_t numNonZeros = 0, nonZeroPoint = 0;
  for (int k = 0; k < z; ++k) {
    const int * pointSrc = pointGrid[k].data();
    int * pointDst = threshHoldedPoint[k].data();

    const int * freeSrc = freeSpace[k].data();
    char * freeDst = threshHoldedFree[k].data();
    for (int i = 0; i < x*y; ++i) {
      if (*(pointSrc + i)) {
        double normalized = (*(pointSrc + i) - averageP)/sigmaP;
        *(pointDst + i) = normalized > -1.0 ? 1 : 0;
        nonZeroPoint += normalized > -1.0 ? 1 : 0;
      }

      if (*(freeSrc + i)) {
        double normalized = (*(freeSrc + i) - averageP)/sigmaP;
        *(freeDst + i) = normalized > -1.0 ? 1 : 0;
        numNonZeros +=  normalized > -1.0 ? 1 : 0;
      }
    }
  }
  pointGrid.clear();
  freeSpace.clear();

  /*FeatureVoxel<float> features (threshHoldedPoint);
  for(int k = 0; k < z; ++k) {
    for(int i = 0; i < x; ++i) {
      for(int j = 0; j < y; ++j) {
        if (threshHoldedPoint[k](j,i)) {
          Eigen::Vector3i pos (i, j, k);
          auto it = xyzToSHOT.find(pos);
          if (it != xyzToSHOT.end()) {
            features.addFeatureVector(i, j, k, it->second);
          }
        }
      }
    }
  }
  threshHoldedPoint.clear();*/

  int newRows = sqrt(2)*std::max(y, x);
  int newCols = newRows;
  int dX = (newCols - x)/2.0;
  int dY = (newRows - y)/2.0;
  Eigen::Vector3d newZZ = zeroZeroD;
  newZZ[0] += dX;
  newZZ[1] += dY;

  std::ofstream metaDataWriter (metaData, std::ios::out | std::ios::binary);
  for (int r = 0; r < NUM_ROTS; ++r) {

    place::voxelGrid rotatedFree, rotatedPoint;
    rotatedFree.v = std::vector<Eigen::MatrixXb> (z, Eigen::MatrixXb::Zero(newRows, newCols));
    rotatedFree.c = numNonZeros;
    /*FeatureVoxel<float> rotatedFeatures (
      std::vector<Eigen::MatrixXi> (z, Eigen::MatrixXi::Zero(newRows, newCols)));*/

    rotatedPoint.v = std::vector<Eigen::MatrixXb> (z, Eigen::MatrixXb::Zero(newRows, newCols));
    rotatedPoint.c = nonZeroPoint;
    for(int k = 0; k < z; ++k) {
      for(int i = 0; i < newCols; ++i) {
        for(int j = 0; j < newRows; ++j) {

          Eigen::Vector3d point (i, j, 0);
          Eigen::Vector3d src = R->at(r)*(point - newZZ) + zeroZeroD;

          if(src[0] < 0 || src[0] >= x)
            continue;
          if(src[1] < 0 || src[1] >= y)
            continue;

          /*int ID = features[k](src[1], src[0]);
          if(ID != 0) {
            auto v = features.getFeatureVector(ID);
            if (v)
              rotatedFeatures.addFeatureVector(i, j, k, v);
            else
              rotatedFeatures[k](j,i) = ID;
          }*/
          rotatedFree.v[k](j, i) = threshHoldedFree[k](src[1], src[0]);
          rotatedPoint.v[k](j, i) = threshHoldedPoint[k](src[1], src[0]);
        }
      }
    }

    int minCol = newCols;
    int minRow = newRows;
    int maxCol = 0;
    int maxRow = 0;
    int minZ = z;
    int maxZ = 0;
    for(int k = 0; k < z; ++k) {
      for(int i = 0; i < newCols; ++i) {
        for(int j = 0; j < newRows; ++j) {
          if(rotatedPoint.v[k](j,i)) {
            minCol = std::min(minCol, i);
            maxCol = std::max(maxCol, i);

            minRow = std::min(minRow, j);
            maxRow = std::max(maxRow, j);

            minZ = std::min(minZ, k);
            maxZ = std::max(maxZ, k);
          }
        }
      }
    }

    const int newZ = maxZ - minZ + 1;
    const int newY = maxRow - minRow + 1;
    const int newX = maxCol - minCol + 1;

    place::voxelGrid trimmedFree, trimmedPoint;
    trimmedFree.v = std::vector<Eigen::MatrixXb> (newZ);
    trimmedFree.c = rotatedFree.c;
   /* FeatureVoxel<float> trimmedFeatures
      (std::vector<Eigen::MatrixXi> (newZ, Eigen::MatrixXi ()),
        rotatedFeatures.getAllFeatureVectors());*/

    trimmedPoint.v = std::vector<Eigen::MatrixXb> (newZ);
    trimmedPoint.c = rotatedPoint.c;

    for (int k = 0; k < newZ; ++k) {
      trimmedFree.v[k] = rotatedFree.v[k + minZ].block(minRow, minCol, newY, newX);
      trimmedPoint.v[k] = rotatedPoint.v[k + minZ].block(minRow, minCol, newY, newX);
    }
    rotatedFree.v.clear();
    rotatedPoint.v.clear();

    /*FeatureVoxel<float> minimalFeatures;
    createMinimalFeatures(trimmedFeatures, minimalFeatures);

    std::cout << trimmedFeatures.getNumFeatures() << " has been reduced to " << minimalFeatures.getNumFeatures() << std::endl;
    trimmedFeatures.clear();*/

    place::metaData meta {zeroZero, newX, newY, newZ, voxelsPerMeter, pixelsPerMeter};
    meta.zZ[0] += dX;
    meta.zZ[1] += dY;
    meta.zZ[0] -= minCol;
    meta.zZ[1] -= minRow;
    meta.zZ[2] -= minZ;
    meta.writeToFile(metaDataWriter);

    trimmedPoint.zZ = meta.zZ;
    trimmedFree.zZ = meta.zZ;

    std::ofstream out (freeNames[r], std::ios::out | std::ios::binary);
    trimmedFree.writeToFile(out);
    out.close();

    out.open(pointNames[r], std::ios::out | std::ios::binary);
    trimmedPoint.writeToFile(out);
    out.close();

    if (FLAGS_preview) {
      displayVoxelGrid(trimmedPoint.v);
      displayVoxelGrid(trimmedFree.v);
    }
  }
  metaDataWriter.close();
}