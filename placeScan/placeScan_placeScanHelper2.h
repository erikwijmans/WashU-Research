#pragma once
#ifndef PLACESCAN_PLACESCANHELPER2_H_
#define PLACESCAN_PLACESCANHELPER2_H_

#include "placeScan_placeScanHelper.h"

#include <scan_typedefs.hpp>
#include <FeatureVoxel.hpp>

extern const int minScans;

namespace place {

  void createWeightedFloorPlan (Eigen::SparseMatrix<double> & weightedFloorPlan);

  void loadInPlacement(const std::string & scanName,
    std::vector<place::moreInfo> & scoreVec, const int scanNum);

  void displayWeightedFloorPlan(Eigen::SparseMatrix<double> & weightedFloorPlan);

  void weightEdges(const std::vector<place::node> & nodes,
    const std::vector<std::vector<place::metaData> > & voxelInfo,
    const std::vector<std::string> & pointVoxelFileNames,
    const std::vector<std::string> & freeVoxelFileNames,
    const std::vector<std::vector<Eigen::Matrix3d> > & rotationMatricies,
    std::vector<place::Panorama> & panoramas,
    Eigen::MatrixXE & adjacencyMatrix);

  void loadInPlacementGraph(const std::string & imageName,
    std::vector<place::node> & nodes, const int num);

  void trimScansAndMasks(const std::vector<cv::Mat> & toTrimScans,
    const std::vector<cv::Mat> & toTrimMasks, std::vector<cv::Mat> & trimmedScans,
    std::vector<cv::Mat> & trimmedMasks, std::vector<Eigen::Vector2i> & zeroZero);

  void displayGraph(const Eigen::MatrixXE & adjacencyMatrix,
    const std::vector<place::node> & nodes,
    const std::vector<std::vector<Eigen::MatrixXb> > & scans,
    const std::vector<std::vector<Eigen::Vector2i> > & zeroZeros);

  void displayBest(const std::vector<place::SelectedNode> & bestNodes,
    const std::vector<std::vector<Eigen::MatrixXb> > & scans,
    const std::vector<std::vector<Eigen::Vector2i> > & zeroZeros);

  place::edge compare3D(const place::voxelGrid & aPoint,
    const place::voxelGrid & bPoint,
    const place::voxelGrid & aFree,
    const place::voxelGrid & bFree,
    const place::cube & aRect, const place::cube & bRect);

  void loadInVoxel(const std::string & name,
    place::voxelGrid & dst);

  void loadInScansGraph(const std::vector<std::string> & pointFileNames,
    const std::vector<std::string> & freeFileNames,
    const std::vector<std::string> & zerosFileNames,
    std::vector<std::vector<Eigen::MatrixXb> > & scans,
    std::vector<std::vector<Eigen::MatrixXb> > & masks,
    std::vector<std::vector<Eigen::Vector2i> > & zeroZeros);

  void TRWSolver(const Eigen::MatrixXE & adjacencyMatrix,
    const std::vector<place::node> & nodes,
    std::vector<SelectedNode> & bestNodes);

  bool reloadGraph(Eigen::MatrixXE & adjacencyMatrix);
  void saveGraph(Eigen::MatrixXE & adjacencyMatrix);

  template<typename T>
  bool localGroup(T & toCheck, const int yOffset,
    const int xOffset, const int range) {
    for (int i = -range; i <= range; ++i) {
      for (int j = -range; j <= range; ++j) {
        if (yOffset + j < 0 || yOffset + j >= toCheck.rows())
          continue;
        if (xOffset + i < 0 || xOffset + i >= toCheck.cols())
          continue;
        if (toCheck(yOffset + j, xOffset + i))
          return true;
      }
    }
    return false;
  }

  void normalizeWeights(Eigen::MatrixXE & adjacencyMatrix,
    std::vector<place::node> & nodes);
}

#endif
