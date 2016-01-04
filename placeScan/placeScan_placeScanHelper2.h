#ifndef PLACESCAN_PLACESCANHELPER2_H_
#define PLACESCAN_PLACESCANHELPER2_H_

#include "placeScan_placeScanHelper.h"


namespace place {
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

	void createWeightedFloorPlan (Eigen::SparseMatrix<double> & weightedFloorPlan);

	void loadInPlacement(const std::string & scanName,
	  std::vector<place::moreInfo> & scoreVec, const int scanNum);

  void displayWeightedFloorPlan(Eigen::SparseMatrix<double> & weightedFloorPlan);

  void createGraph(Eigen::MatrixXd & adjacencyMatrix,
    std::vector<place::node> & nodes,
    std::vector<std::vector<Eigen::Vector2i> > & zeroZeros);

  void weightEdges(const std::vector<place::node> & nodes, 
    const std::vector<std::vector<Eigen::MatrixXb> > & scans,
    const std::vector<std::vector<Eigen::Vector2i> > & zeroZeros,
    const std::vector<std::string> & pointVoxelFileNames,
    const std::vector<std::string> & freeVoxelFileNames,
    Eigen::MatrixXd & adjacencyMatrix);

  void loadInPlacementGraph(const std::string & imageName, 
    std::vector<place::node> & nodes, const int num);

  void trimScansAndMasks(const std::vector<cv::Mat> & toTrimScans, 
    const std::vector<cv::Mat> & toTrimMasks, std::vector<cv::Mat> & trimmedScans,
    std::vector<cv::Mat> & trimmedMasks, std::vector<Eigen::Vector2i> & zeroZero);

  void displayGraph(const Eigen::MatrixXd & adjacencyMatrix, 
    const std::vector<place::node> & nodes,
    const std::vector<std::vector<Eigen::MatrixXb> > & scans, 
    const std::vector<std::vector<Eigen::Vector2i> > & zeroZeros);

  void findLongestPath(const Eigen::MatrixXd & adjacencyMatrix,
    const std::vector<place::node> & nodes,
    std::vector<place::node> & longestPath);

  void pathFinder(const Eigen::MatrixXd & adjacencyMatrix,
    const std::vector<place::node> & nodes,
    int currentNode, bool * colorMap, int numColors, double currentLength,
    std::vector<place::node> & longestPath);


  void displayLongestPath(const std::vector<place::node> & longestPath,
    const std::vector<std::vector<Eigen::MatrixXb> > & scans, 
    const std::vector<std::vector<Eigen::Vector2i> > & zeroZeros);

  void findBestLabels(const Eigen::MatrixXd & adjacencyMatrix, 
    const std::vector<place::node> & nodes, std::vector<place::node> & bestLabels);

  double compare3D(const std::vector<Eigen::MatrixXb> & aPoint,
    const std::vector<Eigen::MatrixXb> & bPoint,
    const std::vector<Eigen::MatrixXb> & aFree,
    const std::vector<Eigen::MatrixXb> & bFree, 
    const place::rect & aRect, const place::rect & bRect);

  void loadInVoxel(const std::string & name, 
    std::vector<Eigen::MatrixXb> & dst);
}



#endif