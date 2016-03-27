#ifndef PLACESCAN_PLACE_SCAN_HELPER_HPP_
#define PLACESCAN_PLACE_SCAN_HELPER_HPP_ 

#include <scan_gflags.h>

#include <string>

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/StdVector>
#include <eigen3/Eigen/Sparse>
#include <opencv2/core.hpp>
#include <gflags/gflags.h>
#include <scan_typedefs.hpp>

extern const double maxDelta, maxTotal;

extern cv::Mat fpColor, floorPlan;
extern std::vector<Eigen::Vector3i> truePlacement;

namespace place {

	void parseFolders(std::vector<std::string> & pointFileNames, 
		std::vector<std::string> & zerosFileNames,
		std::vector<std::string> * freeFileNames);

	void loadInScans(const std::string & scanName,
		const std::string & zerosFile, std::vector<cv::Mat> & rotatedScans,
	  std::vector<Eigen::Vector2i> & zeroZero);

	void loadInScans(const std::string & scanName,
		 const std::string & zerosFile, std::vector<cv::Mat> & rotatedScans);

	void loadInScansAndMasks(const std::string & scanName, 
	  const std::string & zerosFile, 
	  const std::string & maskName, std::vector<cv::Mat> & rotatedScans,
	  std::vector<cv::Mat> & masks, std::vector<Eigen::Vector2i> & zeroZero);

	void trimScans(const std::vector<cv::Mat> & toTrim, 
		std::vector<cv::Mat> & trimmedScans, std::vector<Eigen::Vector2i> & zeroZero);

	void savePlacement(const std::vector<const place::posInfo *> & minima,
		const std::string & outName, const std::vector<Eigen::Vector2i> & zeroZero);

	bool reshowPlacement(const std::string & scanName,
		const std::string & zerosFile, const std::string & preDone);	

	void displayOutput(const std::vector<Eigen::SparseMatrix<double> > & rSSparseTrimmed, 
		const std::vector<const place::posInfo *> & minima);

	void loadInTruePlacement(const std::string & scanName, 
		const std::vector<Eigen::Vector2i> & zeroZero);

	void displayTruePlacement(const std::vector<Eigen::SparseMatrix<double> > &,
		const std::vector<posInfo> & scores, const std::vector<Eigen::Vector2i> & zeroZero);

	cv::Mat sparseToImage(const Eigen::SparseMatrix<double> & toImage);

	void sparseToImage(const Eigen::SparseMatrix<double> & toImage, cv::Mat & imageOut);

	void scanToSparse(const cv::Mat & scan, Eigen::SparseMatrix<double> & sparse);

	Eigen::SparseMatrix<double> scanToSparse(const cv::Mat & scan);

	void displayScanAndMask(const std::vector<std::vector<Eigen::SparseMatrix<double> > > & rSSparsePyramidTrimmed,
		const std::vector<std::vector<Eigen::MatrixXb> > & eMaskPyramidTrimmedNS);

	void erodeSparse(const Eigen::SparseMatrix<double> & src,
		Eigen::SparseMatrix<double> & dst);

	void displayOutput(const Eigen::SparseMatrix<double> & fp,
		const std::vector<Eigen::SparseMatrix<double> > & rSSparseTrimmed, 
		const std::vector<const place::posInfo *> & minima);

	void removeMinimumConnectedComponents(cv::Mat & image);

} //namespace place

std::ostream & operator<<(std::ostream & os, const place::posInfo * print);


#endif