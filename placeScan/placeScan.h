#ifndef PLACE_SCAN_H
#define PLACE_SCAN_H

#include <vector>
#include <string>

#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <Eigen/Sparse>
#include <opencv2/core.hpp>
#include <gflags/gflags.h>
#include <omp.h>


DECLARE_bool(visulization);
DECLARE_bool(previewIn);
DECLARE_bool(previewOut);
DECLARE_bool(replace);
DECLARE_bool(quiteMode);
DECLARE_bool(tinyPreviewIn);
DECLARE_bool(save);
DECLARE_bool(debugMode);
DECLARE_string(floorPlan);
DECLARE_string(dmFolder);
DECLARE_string(rotFolder);
DECLARE_string(preDone);
DECLARE_string(zerosFolder);
DECLARE_int32(startIndex);
DECLARE_int32(numScans);
DECLARE_int32(numLevels);

typedef struct
{
	double score = 0;
	double yasu = 0;
	double yasu2 = 0;
	double entropy = 0;
	int rotation = 0;
	int fpPixels = 0;
	int scanPixels = 0;
	int x = 0;
	int y = 0;
	
} posInfo;

#pragma omp declare reduction (merge : std::vector<posInfo> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp declare reduction (merge : std::vector<int> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))



void analyzePlacement(const cv::Mat &, const std::string &, 
const std::string &, const std::string &);

void findLocalMinima(const std::vector<posInfo> &, std::vector<int> &, const float);

void createPyramid(std::vector<Eigen::SparseMatrix<double> > &, 
	std::vector<std::vector<Eigen::SparseMatrix<double> > > &);

void trimScanPryamids(const std::vector<std::vector<Eigen::SparseMatrix<double> > > &, 
	std::vector<std::vector<Eigen::SparseMatrix<double> > > &, 
	const std::vector<std::vector<Eigen::SparseMatrix<double> > > &, 
	std::vector<std::vector<Eigen::SparseMatrix<double> > > &);

void findPlacementPointBasedV2(const Eigen::SparseMatrix<double> &, 
	const std::vector<Eigen::SparseMatrix<double> > &, const Eigen::SparseMatrix<double> &, 
	const std::vector<Eigen::SparseMatrix<double> > &, 
	std::vector<posInfo> &, const std::vector<Eigen::Vector3i> &);

void findPointsToAnalyze(const std::vector<posInfo> &, const std::vector<int> &, 
	std::vector<Eigen::Vector3i> &);

void findGlobalMinima(const std::vector<posInfo> &, const std::vector<int> &);

void findPointsToAnalyzeV2(const std::vector<posInfo> &, std::vector<Eigen::Vector3i> &);

void displayOutput(const std::vector<Eigen::SparseMatrix<double> > &, const std::vector<int> & ,
	const std::vector<posInfo> &);

void savePlacement(const std::vector<posInfo> &, const std::vector<int> &, 
	const std::string & outName);

bool reshowPlacement(const std::string &, const std::string &, const std::string &);

void loadInScans(const std::string &, const std::string &, 
	std::vector<cv::Mat> &, const std::string &);

bool notInLocalMin(const int, const std::vector<int> &);

void scanToSparse(const cv::Mat &, Eigen::SparseMatrix<double> &);

void blurMinima(std::vector<Eigen::MatrixXd> &, const std::vector<posInfo> &, 
	const Eigen::Vector4i &, const Eigen::Vector4i &);

void trimScans(const std::vector<cv::Mat> &, std::vector<cv::Mat> &);

void loadInTruePlacement(const std::string &);

void displayTruePlacement(const std::vector<Eigen::SparseMatrix<double> > &,
	const std::vector<posInfo> &);





#endif