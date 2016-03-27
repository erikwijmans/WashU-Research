#ifndef SCAN_DENSITY_3D_INFO_H
#define SCAN_DENSITY_3D_INFO_H

#include "scanDensity_scanDensity.h"

namespace voxel {

  class CloudAnalyzer3D {
    private:
      std::shared_ptr<const std::vector<Eigen::Vector3f> > points;
      std::shared_ptr<const std::vector<Eigen::Matrix3d> > R;
      std::shared_ptr<const BoundingBox> bBox;
      std::vector<Eigen::MatrixXi> pointsPerVoxel, numTimesSeen;
      Eigen::Vector3f pointMin, pointMax;
      double voxelsPerMeter, pixelsPerMeter;
      Eigen::Vector3d zeroZeroD;
      Eigen::Vector3i zeroZero;
    public:
      CloudAnalyzer3D(const std::shared_ptr<const std::vector<Eigen::Vector3f> > & points,
        const std::shared_ptr<const std::vector<Eigen::Matrix3d> > & R, 
        const std::shared_ptr<const BoundingBox> & bBox);
      void run(double voxelsPerMeter, double pixelsPerMeter);
      void saveVoxelGrids(const std::vector<std::string> & pointNames,
        const std::vector<std::string> & freeNames,
        const std::string & metaData);
  };

} // voxel


#endif // SCAN_DENSITY_3D_INFO_H
