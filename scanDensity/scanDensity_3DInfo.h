#ifndef SCAN_DENSITY_3D_INFO_H
#define SCAN_DENSITY_3D_INFO_H

#include "scanDensity_scanDensity.h"

#include <unordered_map>

namespace std {
  template <>
  struct hash<Eigen::Vector3i>
  {
    std::size_t operator()(const Eigen::Vector3i & k) const {
      size_t seed = 0;
      for (int i = 0; i < k.size(); ++i) {
        seed ^= k[i] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      }
      return seed;
    }
  };
} // std

namespace voxel {

  class CloudAnalyzer3D {
    private:
      std::shared_ptr<const std::vector<Eigen::Vector3f> > points;
      std::shared_ptr<const std::vector<Eigen::Matrix3d> > R;
      std::shared_ptr<const BoundingBox> bBox;
      std::shared_ptr<const std::vector<SHOT1344WithXYZ> > featureVectors;
      std::unordered_map<Eigen::Vector3i, std::shared_ptr<Eigen::Vector1344f> > xyzToSHOT1334;
      std::vector<Eigen::MatrixXi> pointsPerVoxel, numTimesSeen;
      Eigen::Vector3f pointMin, pointMax;
      double voxelsPerMeter, pixelsPerMeter;
      Eigen::Vector3d zeroZeroD;
      Eigen::Vector3i zeroZero;
    public:
      CloudAnalyzer3D(const std::shared_ptr<const std::vector<Eigen::Vector3f> > & points,
        const std::shared_ptr<const std::vector<Eigen::Matrix3d> > & R,
        const std::shared_ptr<const std::vector<SHOT1344WithXYZ> > & featureVectors,
        const std::shared_ptr<const BoundingBox> & bBox);
      void run(double voxelsPerMeter, double pixelsPerMeter);
      void saveVoxelGrids(const std::vector<std::string> & pointNames,
        const std::vector<std::string> & freeNames,
        const std::string & metaData);
  };

} // voxel


#endif // SCAN_DENSITY_3D_INFO_H
