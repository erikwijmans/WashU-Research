#pragma once
#ifndef PLACE_SCAN_DOOR_DETECTOR_H
#define PLACE_SCAN_DOOR_DETECTOR_H

#include "placeScan_placeScan.h"
#include <opencv/cv.hpp>

namespace place {
class DoorDetector {
public:
  DoorDetector(const std::vector<Eigen::SparseMatrix<double>> &fpPyramid,
               const std::vector<Eigen::SparseMatrix<double>> &erodedFpPyramid,
               const std::vector<Eigen::MatrixXb> &fpMasks);

private:
  cv::Mat_<double> response;
  std::vector<cv::Mat> symbols;
};
} // place

#endif // PLACE_SCAN_DOOR_DETECTOR_H
