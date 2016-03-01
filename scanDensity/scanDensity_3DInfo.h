#ifndef SCAN_DENSITY_3D_INFO_H
#define SCAN_DENSITY_3D_INFO_H

#include "scanDensity_scanDensity.h"

#define NUM_ROTS 4

extern double voxelsPerMeter;

namespace Eigen {
  typedef Matrix<char, Dynamic, Dynamic> MatrixXb;
}


class VoxelInfo {
  private:
    std::vector<std::string> binaryNames, rotationsFiles;
  public:
    VoxelInfo (int argc, char * argv[]);
    void run();
    void run(int startIndex, int numScans);
    void setScale(double newScale) {voxelsPerMeter = newScale;};
    double getScale() {return voxelsPerMeter;};
};

namespace voxel {

  typedef struct {
    Eigen::Vector3i zZ;
    int x, y, z;
  } metaData;

  void analyzeScan3D(const std::string & fileName,
  const std::string & rotationFile);

 void saveVoxelGrids(std::vector<Eigen::MatrixXi> & pointGrid,
   std::vector<Eigen::MatrixXi> & freeSpace,
   const std::vector<Eigen::Matrix3d> & R,
   const Eigen::Vector3d & zeroZeroD,
   const Eigen::Vector3i & zeroZero,
   const std::string & scanNumber);

 void writeGrid(const std::vector<Eigen::MatrixXb> & toWrite, 
  const std::string & outName, const size_t numNonZeros);


  void createVoxelGrids(const std::vector<Eigen::Vector3f> & points,
    const std::string & rotationFile, const std::string & scanNumber);

  void pointBased(const std::vector<Eigen::Vector3f> & points,
    const std::vector<Eigen::Matrix3d> & R,
    float * pointMin, float * pointMax,
    const std::string & scanNumber);

  void freeSpace(const std::vector<Eigen::Vector3f> & points,
    const float * pointMin, const float * pointMax,
    const std::string & scanNumber, const int rotNumber);

  void createBoundingBox(float * pointMin, float * pointMax,
    const std::vector<Eigen::Vector3f> & points);

} // 3D




#endif // SCAN_DENSITY_3D_INFO_H
