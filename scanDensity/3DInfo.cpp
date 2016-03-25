#include "scanDensity_3DInfo.h"
#include <FeatureVoxel.hpp>

voxel::CloudAnalyzer3D::CloudAnalyzer3D(const std::vector<Eigen::Vector3f> * points,
  const std::vector<Eigen::Matrix3d> * R, const BoundingBox * bBox) {
  this->points = points;
  this->R = R;
  this->bBox = bBox;
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
}


static void displayVoxelGrid(const std::vector<Eigen::MatrixXb> & voxelB) {
  Eigen::MatrixXd collapsed (voxelB[0].rows(), voxelB[0].cols());

  for(int i = 0; i < collapsed.cols(); ++i) {
    for(int j = 0; j < collapsed.rows(); ++j) {
      double sum = 0;
      for(int k = 0; k < voxelB.size(); ++k) {
        sum += voxelB[k](j,i);
      }
      collapsed(j,i) = sum;
    }
  }

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

  for(int i = 0; i < collapsed.size(); ++i) {
    if(*(dataPtr + i) !=0)
      sigma += (*(dataPtr + i) - average)*(*(dataPtr + i)- average);
  }

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
          blue = 2 * gray;
          green = 255 - blue;
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

static void writeGrid(const std::vector<Eigen::MatrixXb> & toWrite, 
  const std::string & outName, const size_t numNonZeros) {
  std::ofstream out (outName, std::ios::out | std::ios::binary);

  const int vZ = toWrite.size();
  const int vY = toWrite[0].rows();
  const int vX = toWrite[0].cols();

  out.write(reinterpret_cast<const char *>(& vZ), sizeof(vZ));
  out.write(reinterpret_cast<const char *>(& vY), sizeof(vY));
  out.write(reinterpret_cast<const char *>(& vX), sizeof(vX));

  for(int k = 0; k < vZ; ++k)
    out.write(toWrite[k].data(), toWrite[k].size());

  out.write(reinterpret_cast<const char *>(&numNonZeros), sizeof(numNonZeros));

  out.close();
}

void voxel::CloudAnalyzer3D::saveVoxelGrids(const std::vector<std::string> & pointNames,
    const std::vector<std::string> & freeNames,
    const std::string & metaData) {
  const std::vector<Eigen::MatrixXi> & pointGrid = pointsPerVoxel;
  const std::vector<Eigen::MatrixXi> & freeSpace = numTimesSeen;

  int x, y, z;
  z = pointGrid.size();
  y = pointGrid[0].rows();
  x = pointGrid[0].cols();

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

  
  std::ofstream metaDataWriter (metaData, std::ios::out | std::ios::binary);

  int newRows = std::max(y, x);
  int newCols = newRows;
  int dX = (newCols - x)/2.0;
  int dY = (newRows - y)/2.0;
  Eigen::Vector3d newZZ = zeroZeroD;
  newZZ[0] += dX;
  newZZ[1] += dY;

  for(int r = 0; r < NUM_ROTS; ++r) {
    const std::string & outNamePoint = pointNames[r];
    const std::string & outNameFree = freeNames[r];

    std::vector<Eigen::MatrixXb> voxelPoint (z, Eigen::MatrixXb::Zero(newRows,newCols));
    std::vector<Eigen::MatrixXb> voxelFree (z, Eigen::MatrixXb::Zero(newRows, newCols));

    size_t numNonZerosPoint = 0, numNonZerosFreeSpace = 0;
    for(int k = 0; k < voxelPoint.size(); ++k) {
      for(int i = 0; i < voxelPoint[0].cols(); ++i) {
        for(int j = 0; j < voxelPoint[0].rows(); ++j) {
          
          Eigen::Vector3d point (i,j,0);
          Eigen::Vector3d src = R->at(r)*(point - newZZ) + zeroZeroD;

          if(src[0] < 0 || src[0] >= x)
            continue;
          if(src[1] < 0 || src[1] >= y)
            continue;

          if(pointGrid[k](src[1], src[0]) != 0) {
            double normalized = (pointGrid[k](src[1],src[0]) - averageP)/sigmaP;
            voxelPoint[k](j,i) = normalized > -1.0 ? 1 : 0;
            numNonZerosPoint += normalized > -1.0 ? 1 : 0;
          }
          if(freeSpace[k](src[1],src[0]) != 0) {
            double normalized = (freeSpace[k](src[1],src[0]) - averageF)/sigmaF;
            voxelFree[k](j,i) = normalized > -1.0 ? 1 : 0;
            numNonZerosFreeSpace += normalized > -1.0 ? 1 : 0;
          }
        }
      }
    }

    int minCol = x;
    int minRow = y;
    int maxCol = 0;
    int maxRow = 0;
    int minZ = z;
    int maxZ = 0;
    for(int k = 0; k < z; ++k) {
      for(int i = 0; i < x; ++i) {
        for(int j = 0; j < y; ++j) {
          if(voxelPoint[k](j,i)) {
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

    std::vector<Eigen::MatrixXb> newPoint (newZ), 
      newFree (newZ);

   
    for (int i = 0; i < newZ; ++i) {
      newPoint[i] = voxelPoint[i + minZ].block(minRow, minCol, newY, newX);
      newFree[i] = voxelFree[i + minZ].block(minRow, minCol, newY, newX);
    }

    if(FLAGS_preview) {
      displayVoxelGrid(newPoint);
      displayVoxelGrid(newFree);
    }
    
    writeGrid(newPoint, outNamePoint, numNonZerosPoint);
    writeGrid(newFree, outNameFree, numNonZerosFreeSpace);

    voxel::metaData meta {zeroZero, newX, newY, newZ, voxelsPerMeter, pixelsPerMeter};
    meta.zZ[0] += dX;
    meta.zZ[1] += dY;
    meta.zZ[0] -= minCol;
    meta.zZ[1] -= minRow;
    meta.zZ[2] -= minZ;
    metaDataWriter.write(reinterpret_cast<const char *>(&meta), sizeof(meta));
  }
  metaDataWriter.close();
}