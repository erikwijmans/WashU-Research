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
    int numSameColor;
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
    const std::vector<std::vector<Eigen::MatrixXb> > & masks,
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

}



#endif