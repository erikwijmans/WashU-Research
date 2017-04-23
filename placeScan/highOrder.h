#pragma once
#ifndef PLACE_SCAN_HIGH_ORDER_H
#define PLACE_SCAN_HIGH_ORDER_H

#include "placeScan_multiLabeling.h"
#include "placeScan_placeScanHelper2.h"

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/StdVector>
#include <opencv2/core.hpp>
#include <unordered_map>

namespace place {

void createHigherOrderTermsV2(
    const std::vector<std::vector<Eigen::MatrixXb>> &freeSpace,
    const std::vector<std::vector<Eigen::Vector2i>> &zeroZeros,
    const std::vector<place::node> &nodes,
    multi::Labeler::HighOrderV2 &highOrder);

} // place

#endif // PLACE_SCAN_HIGH_ORDER_H
