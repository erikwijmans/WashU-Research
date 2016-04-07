#ifndef PLACESCAN_PLACESCANHELPER2_H_
#define PLACESCAN_PLACESCANHELPER2_H_

#include "placeScan_placeScanHelper.h"

#include <scan_typedefs.hpp>
#include <FeatureVoxel.hpp>

extern const int minScans;

#include <unordered_map>

namespace std {
  template <>
  struct hash<std::vector<int> >
  {
    std::size_t operator()(const std::vector<int> & k) const {
      size_t seed = 0;
      for (auto v : k) {
        seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      }
      return seed;
    }
  };
} // std

namespace place {
  
	void createWeightedFloorPlan (Eigen::SparseMatrix<double> & weightedFloorPlan);

	void loadInPlacement(const std::string & scanName,
	  std::vector<place::moreInfo> & scoreVec, const int scanNum);

  void displayWeightedFloorPlan(Eigen::SparseMatrix<double> & weightedFloorPlan);

  void createGraph(Eigen::MatrixXE & adjacencyMatrix,
    std::vector<place::node> & nodes,
    std::vector<std::vector<Eigen::Vector2i> > & zeroZeros,
    std::vector<const place::node *> & bestNodes);

  void weightEdges(const std::vector<place::node> & nodes,
    const std::vector<std::vector<place::metaData> > & voxelInfo,
    const std::vector<std::string> & pointVoxelFileNames,
    const std::vector<std::string> & freeVoxelFileNames,
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

  void displayBest(const std::vector<const place::node *> & bestNodes,
    const std::vector<std::vector<Eigen::MatrixXb> > & scans, 
    const std::vector<std::vector<Eigen::Vector2i> > & zeroZeros);

  place::edge compare3D(const voxel::SparseFeatureVoxel<float> & aPoint,
    const voxel::SparseFeatureVoxel<float> & bPoint,
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
    std::vector<const place::node * > & bestNodes);

  bool reloadGraph(Eigen::MatrixXE & adjacencyMatrix);
  void saveGraph(Eigen::MatrixXE & adjacencyMatrix);

  bool localGroup(auto & toCheck, const int yOffset, 
    const int xOffset, const int range);

  void createHigherOrderTerms(const std::vector<std::vector<Eigen::MatrixXb> > & scans,
    const std::vector<std::vector<Eigen::Vector2i> > & zeroZeros,
    const std::vector<place::node> & nodes, std::unordered_map<std::vector<int>, double> &
    highOrder);

  void displayHighOrder(const std::unordered_map<std::vector<int>, double> highOrder, 
    const std::vector<place::node> & nodes, 
    const std::vector<std::vector<Eigen::MatrixXb> > & scans, 
    const std::vector<std::vector<Eigen::Vector2i> > & zeroZeros);

  void MIPSolver(const Eigen::MatrixXE & adjacencyMatrix, 
    const std::unordered_map<std::vector<int>, double> & highOrder, const std::vector<place::node> & nodes,
    std::vector<const place::node *> & bestNodes);

  void normalizeWeights(Eigen::MatrixXE & adjacencyMatrix, 
    std::vector<place::node> & nodes);
}

#endif
