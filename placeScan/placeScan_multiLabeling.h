#pragma once
#ifndef PLACE_SCAN_MULTI_LABELING_H
#define PLACE_SCAN_MULTI_LABELING_H

#include "placeScan_placeScan.h"
#include "placeScan_placeScanHelper.h"
#include "placeScan_placeScanHelper2.h"

#include <unordered_set>

extern const int minScans;

namespace multi {

class Labeler {
public:
  Labeler();
  void weightEdges();
  void solveTRW();
  void solveMIP();
  void displaySolution();
  void displayGraph();
  void getLabeling(std::vector<const place::node *> &labels);
  void saveFinal(int index);
  struct HighOrderEle {
    double w;
    int c;
    HighOrderEle(double w, int c) : w{w}, c{c} {};
  };
  typedef std::unordered_map<std::vector<int>, HighOrderEle> HighOrder;
  typedef std::unordered_map<std::vector<int>, Eigen::VectorXd> HighOrderV2;

private:
  Eigen::MatrixXE adjacencyMatrix;
  std::vector<place::node> nodes, R1Nodes;
  std::vector<std::vector<Eigen::Vector2i>> zeroZeros;
  std::vector<place::SelectedNode> bestNodes;
  std::vector<std::string> pointFileNames, zerosFileNames, freeFileNames;
  std::vector<std::string> pointVoxelFileNames, freeVoxelFileNames;
  std::vector<std::string> metaDataFiles, rotationsFiles, panoFiles;
  std::vector<std::vector<place::MetaData>> voxelInfo;
  std::vector<std::vector<Eigen::MatrixXb>> scans, masks;
  HighOrder highOrder;
  HighOrderV2 highOrderV2;
  std::vector<std::vector<Eigen::Matrix3d>> rotationMatricies;
  std::vector<place::Panorama> panoramas;
  std::unordered_map<int, std::unordered_set<int>> unwantedNeighbors;
  void load();
};

} // multi

#endif // PLACE_SCAN_MULTI_LABELING_H
