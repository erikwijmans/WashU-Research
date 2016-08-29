#pragma once
#ifndef PLACE_SCAN_HIGH_ORDER_H
#define PLACE_SCAN_HIGH_ORDER_H

#include "placeScan_multiLabeling.h"
#include "placeScan_placeScanHelper2.h"

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/StdVector>
#include <opencv2/core.hpp>
#include <unordered_map>

#include <scan_gflags.h>
#include <scan_typedefs.hpp>

namespace place {
void createHigherOrderTerms(
    const std::vector<std::vector<Eigen::MatrixXb>> &scans,
    const std::vector<std::vector<Eigen::Vector2i>> &zeroZeros,
    const std::vector<place::R2Node> &nodes,
    const std::unordered_map<int, std::unordered_set<int>> &unwantedNeighbors,
    multi::Labeler::map &highOrder);

void displayHighOrder(
    const multi::Labeler::map highOrder,
    const std::vector<place::R2Node> &nodes,
    const std::vector<std::vector<Eigen::MatrixXb>> &scans,
    const std::vector<std::vector<Eigen::Vector2i>> &zeroZeros);

void MIPSolver(const Eigen::MatrixXE &adjacencyMatrix,
               const multi::Labeler::map &highOrder,
               const std::vector<place::R2Node> &nodes,
               std::vector<place::SelectedNode> &bestNodes);

void MIPSolver(const Eigen::MatrixXE &adjacencyMatrix,
               const std::vector<place::R2Node> &nodes,
               std::vector<place::SelectedNode> &bestNodes);
void getDirections();
} // place

#endif // PLACE_SCAN_HIGH_ORDER_H
