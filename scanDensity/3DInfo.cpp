#include "scanDensity_3DInfo.h"


void voxel::saveVoxelGrid(std::vector<Eigen::MatrixXi> & grid,
  const std::string & outName) {

  std::ofstream out (outName, std::ios::out | std::ios::binary);

  int x, y, z;
  z = grid.size();
  y = grid[0].rows();
  x = grid[0].cols();

  out.write(reinterpret_cast<const char *>(& z), sizeof(z));
  out.write(reinterpret_cast<const char *>(& y), sizeof(y));
  out.write(reinterpret_cast<const char *>(& x), sizeof(x));

  double average = 0.0;
  int count = 0;

  for(int i = 0; i < z; ++i) {
    const int * dataPtr = grid[i].data();
    for(int j = 0; j < grid[i].outerSize(); ++j) {
      const double value = *(dataPtr + j);
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
    for(int j = 0; j < grid[i].outerSize(); ++j) {
      const double value = *(dataPtr + j);
      if(value)
        sigma += (value - average)*(value - average);
    }
  }
  sigma /= count - 1;
  sigma = sqrt(sigma);

  for(int i = 0; i < z; ++i) {
    const int * valuePtr = grid[i].data();
    for(int j = 0; j < grid[i].size(); ++j) {
      const double normalized = (*(valuePtr + i) - average)/(1.0*sigma);
      const char tmp = normalized > -1.0 ? 1 : 0;
      out.write(&tmp, sizeof(tmp));
    }
  }
  out.close();
}



void voxel::createVoxelGrids(const std::vector<Eigen::Vector3f> & points,
  const float * pointMin, const float * pointMax,  
  const std::string & rotationFile, const std::string & scanNumber) {

  std::ifstream binaryReader (rotationFile, std::ios::in | std::ios::binary);
  std::vector<Eigen::Matrix3d> R (NUM_ROTS);
  for (int i = 0; i < R.size(); ++i) {
    binaryReader.read(reinterpret_cast<char *>(&R[i]),
      sizeof(Eigen::Matrix3d));
  }
  binaryReader.close();

  int i = 0;
  for(auto & rot : R) {
    Eigen::Vector3d pMin (pointMin[0], pointMin[1], pointMin[2]);
    pMin = rot*pMin;

    Eigen::Vector3d pMax (pointMax[0], pointMax[1], pointMax[2]);
    pMax = rot*pMax;

    voxel::pointBased(points, pMin, pMax, rot, scanNumber, rotNumber);
    voxel::freeSpace(points, pMin, pMax, rot, scanNumber, rotNumber);
    ++i;
  }
}

void voxel::pointBased(const std::vector<Eigen::Vector3f> & points,
  const Eigen::Vector3d & pointMin, const Eigen::Vector3d & pointMax,
  const Eigen::Matrix3d & rot, const std::string & scanNumber,
  const int rotNumber) {

  const int numZ = 100;
  const float zScale = (float)numZ/(pointMax[2] - pointMin[2]);
  const int numX = FLAGS_scale * (pointMax[0] - pointMin[0]);
  const int numY = FLAGS_scale * (pointMax[1] - pointMin[1]);

  std::vector<Eigen::MatrixXi> numTimesSeen3D (numZ, Eigen::MatrixXi::Zero(numY, numX));


  for(auto point : points){
    Eigen::Vector3d pointD (point[0], point[1], point[2]);
    pointD = rot*pointD;
    const int x = FLAGS_scale*(pointD[0] - pointMin[0]);
    const int y = FLAGS_scale*(pointD[1] - pointMin[1]);
    const int z = zScale*(pointD[2] - pointMin[2]);
       
    if(x <0 || x >= numX)
      continue;
    if(y < 0 || y >= numY)
      continue; 
    if( z < 0 || z >= numZ)
      continue;

    ++numTimesSeen3D[y](x, z); 
  }
  const std::string outName = FLAGS_voxelFolder + "R" + rotNumber + "/DUC_point_" + scanNumber + ".dat";
  voxel::saveVoxelGrid(numTimesSeen3D, outName);
}

void voxel::freeSpace(const std::vector<Eigen::Vector3f> & points,
  const Eigen::Vector3d & pointMin, const Eigen::Vector3d & pointMax,
  const Eigen::Matrix3d & rot, const std::string & scanNumber,
  const int rotNumber) {

  const int numX = FLAGS_scale * (pointMax[0] - pointMin[0]);
  const int numY = FLAGS_scale * (pointMax[1] - pointMin[1]);
  const int numZ = 100;

  const float zScale = (float)numZ/(pointMax[2] - pointMin[2]);
  float cameraCenter [3];
  cameraCenter[0] = -1*pointMin[0];
  cameraCenter[1] = -1*pointMin[1];
  cameraCenter[2] = -1*pointMin[2];

  std::vector<Eigen::MatrixXi> pointsPerVoxel (numZ, Eigen::MatrixXi::Zero(numY, numX));
  std::vector<Eigen::MatrixXi> numTimesSeen (numZ, Eigen::MatrixXi::Zero(numY, numX));

  for(auto & point : points) {
    Eigen::Vector3d pointD (point[0], point[1], point[2]);
    pointD = rot*pointD;
    int x = floor((pointD[0]- pointMin[0]) * FLAGS_scale);
    int y = floor((pointD[1] - pointMin[1]) * FLAGS_scale);
    int z = floor((pointD[2] - pointMin[2]) * zScale);

    if(x < 0 || x >= numX)
      continue;
    if(y < 0 || y >= numY)
      continue;
    if(z < 0 || z >= numZ)
      continue;

    ++pointsPerVoxel[z](y,x);
  }


  for (int k = 0; k < numZ; ++k) {
    for (int i = 0; i < numX; ++i) {
      for (int j = 0; j < numY; ++j) {
        if(!pointsPerVoxel[k](j,i))
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
        for (int a = 0; a < floor(length-1); ++a) {
      
          voxelHit[0] = floor(cameraCenter[0]*FLAGS_scale + a*unitRay[0]);
          voxelHit[1] = floor(cameraCenter[1]*FLAGS_scale + a*unitRay[1]);
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
  const std::string outName = FLAGS_voxelFolder + "R" + rotNumber + "/DUC_freeSpace_" + scanNumber + ".dat";
  voxel::saveVoxelGrid(numTimesSeen, outName);
}


