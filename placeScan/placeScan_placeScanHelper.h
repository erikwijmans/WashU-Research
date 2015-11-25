#ifndef PLACESCAN_PLACE_SCAN_HELPER_HPP_
#define PLACESCAN_PLACE_SCAN_HELPER_HPP_ 


#include <vector>
#include <string>

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/StdVector>
#include <eigen3/Eigen/Sparse>
#include <opencv2/core.hpp>
#include <gflags/gflags.h>

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
DECLARE_int32(metricNumber);
DECLARE_int32(stopNumber);

extern std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > zeroZero;
extern std::vector<int> globalMins;
extern cv::Mat fpColor, floorPlan;
extern std::vector<Eigen::Vector3i> truePlacement;


typedef struct
{
	double score = 0;
	double scanFP = 0;
	double fpScan = 0;
	int rotation = 0;
	int fpPixels = 0;
	int scanPixels = 0;
	int x = 0;
	int y = 0;
} posInfo;

typedef Eigen::Matrix<char, Eigen::Dynamic, Eigen::Dynamic> Matrixb;


namespace place{
	void parseFolders(std::vector<std::string> & pointFileNames, 
		std::vector<std::string> & rotationFileNames,
		std::vector<std::string> & zerosFileNames,
		std::vector<std::string> * freeFileNames);

	void loadInScans(const std::string & scanName, const std::string & rotationFile, 
		const std::string * zerosFile, std::vector<cv::Mat> & rotatedScans);
	
	void loadInScansAndMasks(const std::string & scanName, 
		const std::string & rotationFile, const std::string & zerosFile, 
		const std::string & maskName, std::vector<cv::Mat> & rotatedScans,
		std::vector<cv::Mat> & masks);

	void trimScans(const std::vector<cv::Mat> & toTrim, 
		std::vector<cv::Mat> & trimmedScans);

	void savePlacement(const std::vector<posInfo> & scores, const std::vector<int> & localMinima, 
		const std::string & outName);

	bool reshowPlacement(const std::string & scanName, const std::string & rotationFile,
		const std::string & zerosFile);	

	void displayOutput(const std::vector<Eigen::SparseMatrix<double> > & rSSparseTrimmed, 
		const std::vector<int> & localMinima, const std::vector<posInfo> & scores);

	void loadInTruePlacement(const std::string & scanName);

	void displayTruePlacement(const std::vector<Eigen::SparseMatrix<double> > &,
		const std::vector<posInfo> &);

	cv::Mat sparseToImage(const Eigen::SparseMatrix<double> & toImage);

	void sparseToImage(const Eigen::SparseMatrix<double> & toImage, cv::Mat & imageOut);

	void scanToSparse(const cv::Mat & scan, Eigen::SparseMatrix<double> & sparse);

	Eigen::SparseMatrix<double> scanToSparse(const cv::Mat & scan);

	void displayMostConfidentScan(const std::string & imageName, 
		const std::string & rotationName, const std::string & zerosFile,
		posInfo & scoreInfo);

	void displayScanAndMask(const std::vector<std::vector<Eigen::SparseMatrix<double> > > & rSSparsePyramidTrimmed,
	const std::vector<std::vector<Eigen::MatrixXd> > & eMaskPyramidTrimmedNS);
}


#endif