#ifndef PLACESCAN_PLACESCANHELPER2_H_
#define PLACESCAN_PLACESCANHELPER2_H_

#include "placeScan_placeScanHelper.h"


const double pixelsPerMeter = 73.5;
const double voxelsPerMeter = 20.0;

namespace place {
  typedef struct {
    double pA, feA, feB, fx;
    double w;
  } edgeWeight;

  typedef struct {
    double weight = 0.0;
    std::vector<int> incident;
  } higherOrder;
} // place


namespace Eigen {
  typedef Matrix<place::edgeWeight, Dynamic, Dynamic> MatrixXS;
  typedef Array<place::higherOrder, Dynamic, Dynamic> ArrayXH;
} // Eigen
  
namespace place {
  typedef struct {
    std::vector<Eigen::MatrixXb> v;
    size_t c;
  } voxel;

	typedef struct {
		posInfo s;
		int scanNum;
	} moreInfo;

	typedef struct {
		posInfo s;
		int color;
	} node;

  typedef struct {
    int X1;
    int Y1;
    int X2;
    int Y2;
  } rect;

  typedef struct {
    int X1, Y1, Z1;
    int X2, Y2, Z2;
  } cube;

   typedef struct {
    Eigen::Vector3i zZ;
    int x, y, z;
  } metaData;

	void createWeightedFloorPlan (Eigen::SparseMatrix<double> & weightedFloorPlan);

	void loadInPlacement(const std::string & scanName,
	  std::vector<place::moreInfo> & scoreVec, const int scanNum);

  void displayWeightedFloorPlan(Eigen::SparseMatrix<double> & weightedFloorPlan);

  void createGraph(Eigen::MatrixXS & adjacencyMatrix,
    std::vector<place::node> & nodes,
    std::vector<std::vector<Eigen::Vector2i> > & zeroZeros,
    std::vector<const place::node *> & bestNodes);

  void weightEdges(const std::vector<place::node> & nodes, 
    const std::vector<std::vector<place::metaData> > & voxelInfo,
    const std::vector<std::string> & pointVoxelFileNames,
    const std::vector<std::string> & freeVoxelFileNames,
    Eigen::MatrixXS & adjacencyMatrix);

  void loadInPlacementGraph(const std::string & imageName, 
    std::vector<place::node> & nodes, const int num);

  void trimScansAndMasks(const std::vector<cv::Mat> & toTrimScans, 
    const std::vector<cv::Mat> & toTrimMasks, std::vector<cv::Mat> & trimmedScans,
    std::vector<cv::Mat> & trimmedMasks, std::vector<Eigen::Vector2i> & zeroZero);

  void displayGraph(const Eigen::MatrixXS & adjacencyMatrix, 
    const std::vector<place::node> & nodes,
    const std::vector<std::vector<Eigen::MatrixXb> > & scans, 
    const std::vector<std::vector<Eigen::Vector2i> > & zeroZeros);

  void findLongestPath(const Eigen::MatrixXS & adjacencyMatrix,
    const std::vector<place::node> & nodes,
    std::vector<place::node> & longestPath);

  void pathFinder(const Eigen::MatrixXS & adjacencyMatrix,
    const std::vector<place::node> & nodes,
    int currentNode, bool * colorMap, int numColors, double currentLength,
    std::vector<place::node> & longestPath);


  void displayTRW(const std::vector<const place::node *> & bestNodes,
    const std::vector<std::vector<Eigen::MatrixXb> > & scans, 
    const std::vector<std::vector<Eigen::Vector2i> > & zeroZeros);

  void findBestLabels(const Eigen::MatrixXS & adjacencyMatrix, 
    const std::vector<place::node> & nodes, std::vector<place::node> & bestLabels);

  place::edgeWeight compare3D(const place::voxel & aPoint,
    const place::voxel & bPoint,
    const place::voxel & aFree,
    const place::voxel & bFree, 
    const place::cube & aRect, const place::cube & bRect, bool display);

  void loadInVoxel(const std::string & name, 
    place::voxel & dst);

  void loadInScansGraph(const std::vector<std::string> & pointFileNames,
  const std::vector<std::string> & freeFileNames,
  const std::vector<std::string> & zerosFileNames,
  std::vector<std::vector<Eigen::MatrixXb> > & scans,
  std::vector<std::vector<Eigen::MatrixXb> > & masks,
  std::vector<std::vector<Eigen::Vector2i> > & zeroZeros);

  void TRWSolver(const Eigen::MatrixXS & adjacencyMatrix,
    const std::vector<place::node> & nodes,
    std::vector<const place::node * > & bestNodes);

  bool reloadGraph(Eigen::MatrixXS & adjacencyMatrix);
  void saveGraph(Eigen::MatrixXS & adjacencyMatrix);

  bool localGroup(const Eigen::MatrixXb & toCheck, const int yOffset, 
  const int xOffset);

  void createHigherOrderTerms(const std::vector<std::vector<Eigen::MatrixXb> > & scans,
    const std::vector<place::node> & nodes, Eigen::ArrayXH & map);

  void MIPSolver(const Eigen::MatrixXS & adjacencyMatrix, 
    const Eigen::ArrayXH & map, const std::vector<place::node> & nodes,
    std::vector<const place::node *> & bestNodes);
}


#endif