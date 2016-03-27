#ifndef PLACESCAN_PLACESCANHELPER2_H_
#define PLACESCAN_PLACESCANHELPER2_H_

#include "placeScan_placeScanHelper.h"

#include <scan_typedefs.hpp>

extern const int minScans;


namespace place {
  class cube {
    public:
      int X1, Y1, Z1;
      int X2, Y2, Z2;
      cube();
      ~cube();
      int volume();
  };

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
    const std::vector<cv::Mat> & panoramas,
    const std::vector<std::vector<Eigen::Matrix3d> > & rotationMatricies,
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


  place::edgeWeight compare3D(const place::voxel & aPoint,
    const place::voxel & bPoint,
    const place::voxel & aFree,
    const place::voxel & bFree, 
    const place::cube & aRect, const place::cube & bRect,
    std::vector<Eigen::Vector3d> & aOverlap,  std::vector<Eigen::Vector3d> & bOverlap);

  void loadInVoxel(const std::string & name, 
    place::voxel & dst);

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

  bool localGroup(const Eigen::MatrixXb & toCheck, const int yOffset, 
    const int xOffset, const int range);

  void createHigherOrderTerms(const std::vector<std::vector<Eigen::MatrixXb> > & scans,
    const std::vector<std::vector<Eigen::Vector2i> > & zeroZeros,
    const std::vector<place::node> & nodes, std::map<std::vector<int>, double> &
    highOrder);

  void displayHighOrder(const std::map<std::vector<int>, double> highOrder, 
    const std::vector<place::node> & nodes, 
    const std::vector<std::vector<Eigen::MatrixXb> > & scans, 
    const std::vector<std::vector<Eigen::Vector2i> > & zeroZeros);

  void MIPSolver(const Eigen::MatrixXE & adjacencyMatrix, 
    const std::map<std::vector<int>, double> & highOrder, const std::vector<place::node> & nodes,
    std::vector<const place::node *> & bestNodes);

  void normalizeWeights(Eigen::MatrixXE & adjacencyMatrix, 
    std::vector<place::node> & nodes);
}


#endif