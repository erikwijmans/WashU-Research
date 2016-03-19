#ifndef SCAN_DENSITY_3D_INFO_H
#define SCAN_DENSITY_3D_INFO_H

#include "scanDensity_scanDensity.h"

#define NUM_ROTS 4

namespace Eigen {
  typedef Matrix<char, Dynamic, Dynamic> MatrixXb;
}

namespace voxel {

  typedef struct {
    Eigen::Vector3i zZ;
    int x, y, z;
    double vox, s;
  } metaData;

  class CloudAnalyzer3D {
    private:
      const std::vector<Eigen::Vector3f> * points;
      const std::vector<Eigen::Matrix3d> * R;
      const BoundingBox * bBox;
      std::vector<Eigen::MatrixXi> pointsPerVoxel, numTimesSeen;
      Eigen::Vector3f pointMin, pointMax;
      double voxelsPerMeter, pixelsPerMeter;
      Eigen::Vector3d zeroZeroD;
      Eigen::Vector3i zeroZero;
    public:
      CloudAnalyzer3D(const std::vector<Eigen::Vector3f> * points,
        const std::vector<Eigen::Matrix3d> * R, const BoundingBox * bBox);
      void run(double voxelsPerMeter, double pixelsPerMeter);
      void saveVoxelGrids(const std::vector<std::string> & pointNames,
        const std::vector<std::string> & freeNames,
        const std::string & metaData);
  };

} // voxel




#endif // SCAN_DENSITY_3D_INFO_H
