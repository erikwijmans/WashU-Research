#include "scanDensity_3DInfo.h"


void voxel::analyzeScan3D(const std::string & fileName,
  const std::string & rotationFile) {
  const std::string scanNumber = fileName.substr(fileName.find(".") - 3, 3);
  std::cout << scanNumber << std::endl;

  if(!FLAGS_redo) {
    std::string outName = FLAGS_voxelFolder + "R1" + "/DUC_point_" + scanNumber + ".dat";
    std::ifstream in1 (outName, std::ios::in | std::ios::binary);

    outName = FLAGS_voxelFolder + "R1" + "/DUC_freeSpace_" + scanNumber + ".dat";
    std::ifstream in2 (outName, std::ios::in | std::ios::binary);
    if(in1.is_open() && in2.is_open())
      return;
  }

  std::ifstream scanFile (fileName, std::ios::in | std::ios::binary);

  int columns, rows;
  scanFile.read(reinterpret_cast<char *> (& columns), sizeof(int));
  scanFile.read(reinterpret_cast<char *> (& rows), sizeof(int));

  std::vector<Eigen::Vector3f> points;
  points.reserve(columns*rows/10);
  int numCenter = 0;
  for (int k = 0; k < columns * rows; ++k) {
    Eigen::Vector3f point;
    scanFile.read(reinterpret_cast<char *> (point.data()), sizeof(point));

    point[1] *= -1.0;

    if(k%10)
      continue;

    if(point[0]*point[0] + point[1]*point[1] < 1) {
      if(numCenter%3 == 0)
        points.push_back(point);
      ++numCenter;
    } else if(point[0] || point[1] || point[2])
      points.push_back(point);
  }
  scanFile.close();

  voxel::createVoxelGrids(points, rotationFile, scanNumber);
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


void voxel::saveVoxelGrid(std::vector<Eigen::MatrixXi> & grid,
  const std::vector<Eigen::Matrix3d> & R,
  const Eigen::Vector3d & zeroZero,
  const std::string & scanNumber, const std::string & type) {

  int x, y, z;
  z = grid.size();
  y = grid[0].rows();
  x = grid[0].cols();

  double average = 0.0;
  int count = 0;

  for(int i = 0; i < z; ++i) {
    const int * dataPtr = grid[i].data();
    for(int j = 0; j < grid[i].size(); ++j) {
      const int value = *(dataPtr + j);
      if(value) {
        average += value;
        ++count;
      }
    }
  }
  average /= count;
  double sigma = 0.0;
  for(int i = 0; i < z; ++i) {
    const int * dataPtr = grid[i].data();
    for(int j = 0; j < grid[i].size(); ++j) {
      const int value = *(dataPtr + j);
      if(value)
        sigma += (value - average)*(value - average);
    }
  }
  sigma /= count - 1;
  sigma = sqrt(sigma);

  for(int r = 0; r < NUM_ROTS; ++r) {
    const std::string outName = FLAGS_voxelFolder + "R" + std::to_string(r) + "/"
      + "DUC_" + type + "_" + scanNumber + ".dat";

    size_t numNonZeros = 0;
    std::vector<Eigen::MatrixXb> voxelB (z, Eigen::MatrixXb::Zero(y,x));
    for(int k = 0; k < voxelB.size(); ++k) {
      for(int i = 0; i < voxelB[0].cols(); ++i) {
        for(int j = 0; j < voxelB[0].rows(); ++j) {
          
          Eigen::Vector3d point (i,j,0);
          Eigen::Vector3d src = R[r]*(point - zeroZero) + zeroZero;

          if(src[0] < 0 || src[0] >= x)
            continue;
          if(src[1] < 0 || src[1] >= y)
            continue;

          if(grid[k](src[1],src[0]) != 0) {
            double normalized = (grid[k](src[1],src[0]) - average)/sigma;
            voxelB[k](j,i) = normalized > -0.5 ? 1 : 0;
            numNonZeros += normalized > -0.5 ? 1 : 0;
          }
        }
      }
    }

    if(FLAGS_preview)
      displayVoxelGrid(voxelB);

    std::ofstream out (outName, std::ios::out | std::ios::binary);

    const int vZ = voxelB.size();
    const int vY = voxelB[0].rows();
    const int vX = voxelB[0].cols();

    out.write(reinterpret_cast<const char *>(& vZ), sizeof(vZ));
    out.write(reinterpret_cast<const char *>(& vY), sizeof(vY));
    out.write(reinterpret_cast<const char *>(& vX), sizeof(vX));


    for(int k = 0; k < vZ; ++k)
      out.write(voxelB[k].data(), voxelB[k].size());
    out.write(reinterpret_cast<const char *>(&numNonZeros), sizeof(numNonZeros));

    out.close();
  }

  

  
}

void voxel::createVoxelGrids(const std::vector<Eigen::Vector3f> & points,
  const std::string & rotationFile, const std::string & scanNumber) {

  std::ifstream binaryReader (rotationFile, std::ios::in | std::ios::binary);
  std::vector<Eigen::Matrix3d> R (NUM_ROTS);
  for (int i = 0; i < R.size(); ++i) {
    binaryReader.read(reinterpret_cast<char *>(R[i].data()),
      sizeof(Eigen::Matrix3d));
  }
  binaryReader.close();

  

  if(!FLAGS_quiteMode)
    std::cout << "Voxel Grids" << std::endl;

  float pointMin [3], pointMax[3];
  voxel::createBoundingBox(pointMin, pointMax, points);

  voxel::pointBased(points, R, pointMin, pointMax, scanNumber);
  
}

void voxel::pointBased(const std::vector<Eigen::Vector3f> & points,
  const std::vector<Eigen::Matrix3d> & R,
  float * pointMin, float * pointMax,
  const std::string & scanNumber) {

  if (!FLAGS_quiteMode)
    std::cout << "point based" << std::endl;

  /*const int numZ = 100;
  const float zScale = numZ/(pointMax[2] - pointMin[2]);*/

 /* float tmpMin [] = {pointMin[0], pointMin[1], -10};
  float tmpMax [] = {pointMax[0], pointMax[1], 10};

  pointMin = tmpMin;
  pointMax = tmpMax;*/
  
  const int numX = voxelsPerMeter * (pointMax[0] - pointMin[0]);
  const int numY = voxelsPerMeter * (pointMax[1] - pointMin[1]);

  const float zScale = voxelsPerMeter;
  const int numZ = zScale * (pointMax[2] - pointMin[2]);

  std::vector<Eigen::MatrixXi> numTimesSeen3D (numZ, Eigen::MatrixXi::Zero(numY, numX));

  for(auto & point : points){
    const int x = voxelsPerMeter*(point[0] - pointMin[0]);
    const int y = voxelsPerMeter*(point[1] - pointMin[1]);
    const int z = zScale*(point[2] -pointMin[2]);
       
    if(x <0 || x >= numX)
      continue;
    if(y < 0 || y >= numY)
      continue; 
    if( z < 0 || z >= numZ)
      continue;

    ++numTimesSeen3D[z](y, x); 
  }

 /* int minCol = numX;
  int minRow = numY;
  int maxCol = 0;
  int maxRow = 0;
  int minZ = numZ;
  int maxZ = 0;
  for(int k = 0; k < numZ; ++k) {
    for(int i = 0; i < numX; ++i) {
      for(int j = 0; j < numY; ++j) {
        if(numTimesSeen3D[k](j,i)) {
          minCol = std::min(minCol, i);
          maxCol = std::max(maxCol, i);

          minRow = std::min(minRow, j);
          maxRow = std::max(maxRow, j);

          minZ = std::min(minZ, k);
          maxZ = std::max(maxZ, k);
        }
      }
    }
  }*/

  const int trimmedX = numX;
  const int trimmedY = numY;
  const int trimmedZ = numZ;
  std::vector<Eigen::MatrixXi> & trimmed = numTimesSeen3D;

 /* const int trimmedZ = maxZ - minZ + 1;

  std::vector<Eigen::MatrixXi > trimmed (trimmedZ);
  for(int i = minZ; i < maxZ + 1; ++i) {
    trimmed[i - minZ] = numTimesSeen3D[i];
  }*/

  const Eigen::Vector3d zeroZeroD (-pointMin[0]*voxelsPerMeter,
    -pointMin[1]*voxelsPerMeter, 0);
  voxel::saveVoxelGrid(trimmed, R, zeroZeroD, scanNumber, "point");

  Eigen::Vector3i zeroZero (-pointMin[0]*voxelsPerMeter, 
    -pointMin[1]*voxelsPerMeter, -pointMin[2]*zScale);
  /*zeroZero[0] -= minCol;
  zeroZero[1] -= minRow; */
  // zeroZero[2] -= minZ;

  voxel::metaData tmp = {zeroZero, trimmedX, trimmedY, trimmedZ};

  std::string metaData = FLAGS_voxelFolder + "metaData/" + "DUC_scan_" + scanNumber + ".dat";
  std::ofstream metaDataWriter (metaData, std::ios::out | std::ios::binary);

  for(int i = 0; i < NUM_ROTS; ++i)
    metaDataWriter.write(reinterpret_cast<const char *>(&tmp), sizeof(tmp));

  metaDataWriter.close();

  //Free space evidence

  float cameraCenter [3];
  cameraCenter[0] = -1*pointMin[0];
  cameraCenter[1] = -1*pointMin[1];
  cameraCenter[2] = -1*pointMin[2];
  std::vector<Eigen::MatrixXi> numTimesSeen (trimmedZ, Eigen::MatrixXi::Zero(trimmedY, trimmedX));

  for (int k = 0; k < trimmedZ; ++k) {
    for (int i = 0; i < trimmedX; ++i) {
      for (int j = 0; j < trimmedY; ++j) {
        if(!trimmed[k](j,i))
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
        int stop = floor(length - 2);
        int voxelHit [3];
        for (int a = 0; a < stop; ++a) {
      
          voxelHit[0] = floor(cameraCenter[0]*voxelsPerMeter + a*unitRay[0]);
          voxelHit[1] = floor(cameraCenter[1]*voxelsPerMeter + a*unitRay[1]);
          voxelHit[2] = floor(cameraCenter[2]*zScale + a*unitRay[2]);

          if(voxelHit[0] < 0 || voxelHit[0] >= trimmedX)
            continue;
          if(voxelHit[1] < 0 || voxelHit[1] >= trimmedY)
            continue;
          if(voxelHit[2] < 0 || voxelHit[2] >= trimmedZ)
            continue;
          
          numTimesSeen[voxelHit[2]](voxelHit[1], voxelHit[0])
            += trimmed[k](j,i);
        }
      }
    }
  }

   
  
  voxel::saveVoxelGrid(numTimesSeen, R, zeroZeroD, scanNumber, "freeSpace");
}

void voxel::freeSpace(const std::vector<Eigen::Vector3f> & points,
  const float * pointMin, const float * pointMax,
  const std::string & scanNumber, const int rotNumber) {

  if (!FLAGS_quiteMode)
    std::cout << "freeSpace" << std::endl;

  const int numZ = voxelsPerMeter *(pointMax[2] - pointMin[2]);
  const int numX = voxelsPerMeter * (pointMax[0] - pointMin[0]);
  const int numY = voxelsPerMeter * (pointMax[1] - pointMin[1]);
  
  float cameraCenter [3];
  cameraCenter[0] = -1*pointMin[0];
  cameraCenter[1] = -1*pointMin[1];
  cameraCenter[2] = -1*pointMin[2];

  std::vector<Eigen::MatrixXi> pointsPerVoxel (numZ, Eigen::MatrixXi::Zero(numY, numX));
  std::vector<Eigen::MatrixXi> numTimesSeen (numZ, Eigen::MatrixXi::Zero(numY, numX));

  for(auto & point : points) {
    int x = floor((point[0]- pointMin[0]) * voxelsPerMeter);
    int y = floor((point[1] - pointMin[1]) * voxelsPerMeter);
    int z = floor((point[2] - pointMin[2]) * voxelsPerMeter);

    if(x < 0 || x >= numX)
      continue;
    if(y < 0 || y >= numY)
      continue;
    if(z < 0 || z >= numZ)
      continue;

    ++pointsPerVoxel[z](y,x);
  }

  
  

  int minCol = numX;
  int minRow = numY;
  int maxCol = 0;
  int maxRow = 0;
  for(int k = 0; k < numZ; ++k) {
    for(int i = 0; i < numX; ++i) {
      for(int j = 0; j < numY; ++j) {
        if(numTimesSeen[k](j,i)) {
          minCol = std::min(minCol, i);
          maxCol = std::max(maxCol, i);

          minRow = std::min(minRow, j);
          maxRow = std::max(maxRow, j);
        }
      }
    }
  }
  std::vector<Eigen::MatrixXi> trimmed (numZ);
  for(int i = 0; i < numZ; ++i) {
    trimmed[i] = numTimesSeen[i].block(minRow, minCol, maxRow - minRow + 1,
      maxCol - minCol + 1);
  }
  numTimesSeen.clear();
}


void voxel::createBoundingBox(float * pointMin, float * pointMax,
  const std::vector<Eigen::Vector3f> & points){
  double averageX, averageY, sigmaX, sigmaY, averageZ, sigmaZ;
  averageX = averageY = sigmaX = sigmaY = averageZ = sigmaZ = 0;

  for (auto & point : points) {
    averageX += point[0];
    averageY += point[1];
    averageZ += point[2];
  }
  averageX = averageX/points.size();
  averageY = averageY/points.size();
  averageZ = averageZ/points.size();

  for (auto & point : points) {
    sigmaX += (point[0] - averageX)*
      (point[0] - averageX);
    sigmaY += (point[1] - averageY)*
      (point[1] - averageY);
    sigmaZ += (point[2] - averageZ)*
      (point[2] - averageZ);
  }
  sigmaX = sigmaX/(points.size()-1);
  sigmaY = sigmaY/(points.size()-1);
  sigmaZ = sigmaZ/(points.size()-1);
  sigmaX = sqrt(sigmaX);
  sigmaY = sqrt(sigmaY);
  sigmaZ = sqrt(sigmaZ);

  if(!FLAGS_quiteMode)
  {
    std::cout << "averageX: " << averageX << std::endl;
    std::cout << "averageY: " << averageY << std::endl;
    std::cout << "averageZ: " << averageZ << std::endl;
    std::cout << "sigmaX: " << sigmaX << std::endl;
    std::cout << "sigmaY: " << sigmaY << std::endl;
    std::cout << "sigmaZ: " << sigmaZ << std::endl;
  }

  double dX = 1.1*10*sigmaX;
  double dY = 1.1*10*sigmaY;
  double dZ = 1.1*6*sigmaZ;


  pointMin[0] = averageX - dX/2;
  pointMin[1] = averageY - dY/2;
  pointMin[2] = averageZ - dZ/2;

  pointMax[0] = averageX + dX/2;
  pointMax[1] = averageY + dY/2;
  pointMax[2] = averageZ + dZ/2;
} 