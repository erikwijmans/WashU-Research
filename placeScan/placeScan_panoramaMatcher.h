#pragma once
#ifndef PLACE_SCAN_PANORAMA_MATCHER_H
#define PLACE_SCAN_PANORAMA_MATCHER_H

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/StdVector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <scan_typedefs.hpp>

extern double totatlCount, numCalls;

namespace pano {

  double compareSIFT(const cv::Mat & panoA, const cv::Mat & panoB,
    const std::vector<Eigen::Vector3d> & pointsInA,
    const std::vector<Eigen::Vector3d> & pointsInB);

  double compareSIFT2(place::Panorama & panoA,
    place::Panorama & panoB, const Eigen::Matrix3d & RA,
    const Eigen::Matrix3d & RB, const Eigen::Vector3d & aToB,
    const Eigen::Vector3d & bToA);

  double compare( const cv::Mat & panoA, const cv::Mat & panoB,
    const Eigen::Matrix3d & RA, const Eigen::Matrix3d & RB,
    const std::vector<Eigen::Vector3d> & pointsInA,
    const Eigen::Vector3d & translation,
    const std::vector<Eigen::Vector3d> & pointsInB,
    const Eigen::RowMatrixXd & rMap);

  void voxelGridToWorld(std::vector<Eigen::Vector3d> & points,
    const Eigen::Matrix3d & R, const Eigen::Vector3i & zeroZero,
    const double metersPerVoxel);

  double compareNCC2(place::Panorama & panoA,
    place::Panorama & panoB, const Eigen::Matrix3d & RA,
    const Eigen::Matrix3d & RB, const Eigen::Vector3d & aToB,
    const Eigen::Vector3d & bToA);

} // pano


#endif // PLACE_SCAN_PANORAMA_MATCHER_H
