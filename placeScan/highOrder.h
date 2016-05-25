#pragma once
#ifndef PLACE_SCAN_HIGH_ORDER_H
#define PLACE_SCAN_HIGH_ORDER_H

#include "placeScan_placeScanHelper2.h"

#include <opencv2/core.hpp>
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/StdVector>
#include <unordered_map>

#include <scan_typedefs.hpp>
#include <scan_gflags.h>

namespace std {
  template <>
  struct hash<std::vector<int> >
  {
    size_t operator()(const std::vector<int> & k) const {
      constexpr double A = 1.6180339887498948482*1e5;
      size_t seed = 0;
      for (auto v : k) {
        seed ^= static_cast<size_t>(v*A) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      }
      return seed;
    }
  };
} // std

namespace place {
  void createHigherOrderTerms(const std::vector<std::vector<Eigen::MatrixXb> > & scans,
    const std::vector<std::vector<Eigen::Vector2i> > & zeroZeros,
    const std::vector<place::node> & nodes,
    std::unordered_map<std::vector<int>, double> & highOrder);

  void displayHighOrder(const std::unordered_map<std::vector<int>, double> highOrder,
    const std::vector<place::node> & nodes,
    const std::vector<std::vector<Eigen::MatrixXb> > & scans,
    const std::vector<std::vector<Eigen::Vector2i> > & zeroZeros);

  void MIPSolver(const Eigen::MatrixXE & adjacencyMatrix,
    const std::unordered_map<std::vector<int>, double> & highOrder, const std::vector<place::node> & nodes,
    std::vector<const place::node *> & bestNodes);
} // place


#endif // PLACE_SCAN_HIGH_ORDER_H
