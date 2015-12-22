#ifndef PLACESCAN_PLACE_SCAN_H_
#define PLACESCAN_PLACE_SCAN_H_

#include "placeScan_placeScanHelper.h"
#include "placeScan_placeScanHelper2.h"



namespace place{
	void analyzePlacement(const std::vector<Eigen::SparseMatrix<double> > & fpPyramid,
	  const std::vector<Eigen::SparseMatrix<double> > & erodedFpPyramid,
	  const std::vector<Eigen::MatrixXb> & fpMasks,
	  const std::string & scanName, const std::string & rotationFile,
	  const std::string & zerosFile, const std::string & maskName);

	const posInfo *** findLocalMinima(const std::vector<place::posInfo> & scores,
	  const float bias, const int rows, const int cols, const double exclusionSize);

	void createPyramid(std::vector<Eigen::SparseMatrix<double> > & pyramid);

	void createPyramid( std::vector<std::vector<Eigen::SparseMatrix<double> > >  & pyramid);

	void trimScanPryamids(const std::vector<std::vector<Eigen::SparseMatrix<double> > >  & rSSparsePyramid,
		std::vector<std::vector<Eigen::SparseMatrix<double> > > & rSSparsePyramidTrimmed, 
		const std::vector<std::vector<Eigen::SparseMatrix<double> > >  & erodedSparsePyramid,
		std::vector<std::vector<Eigen::SparseMatrix<double> > > & erodedSparsePyramidTrimmed,
		const std::vector<std::vector<Eigen::SparseMatrix<double> > >  & eMaskPyramid,
		std::vector<std::vector<Eigen::SparseMatrix<double> > >  & eMaskPyramidTrimmed,
		std::vector<Eigen::Vector2i> & zeroZero);

	void findPlacement(const Eigen::SparseMatrix<double> & fp, 
    const std::vector<Eigen::SparseMatrix<double> > & scans, const Eigen::SparseMatrix<double> & fpE, 
    const std::vector<Eigen::SparseMatrix<double> > & scansE, 
    const std::vector<Eigen::MatrixXb> & masks, const Eigen::Vector4d & numPixelsUnderMask,
    const Eigen::MatrixXb & fpMask, const std::vector<Eigen::Vector3i> & points,
    const double maxFpWeight, std::vector<place::posInfo> & scores);

	void findPointsToAnalyze(const std::vector<posInfo> & scores, const std::vector<int> & localMinima,
		std::vector<Eigen::Vector3i> & pointsToAnalyze);

	void findPointsToAnalyzeV2(const std::vector<const place::posInfo *> & minima, 
  	std::vector<Eigen::Vector3i> & pointsToAnalyze);

	void findGlobalMinima(const std::vector<posInfo> & scores, const std::vector<int> & localMinima);

	void findGlobalMinimaV2(const std::vector<place::posInfo> & scores, 
	  const place::posInfo *** map, const int rows, const int cols, const double exclusionSize, 
	  std::vector<const place::posInfo *> & minima);

	Eigen::MatrixXd distanceTransform(const Eigen::SparseMatrix<double> & image);

	void createFPPyramids(const cv::Mat & floorPlan, 
		std::vector<Eigen::SparseMatrix<double> > & fpPyramid,  
		std::vector<Eigen::SparseMatrix<double> > & erodedFpPyramid);

	void findNumPixelsUnderMask(const std::vector<std::vector<Eigen::SparseMatrix<double> > > & rSSparsePyramidTrimmed,
		const std::vector<std::vector<Eigen::MatrixXb> > & eMaskPyramidTrimmedNS,
		std::vector<Eigen::Vector4d> & numPixelsUnderMask);

	/*void blurMinima(const std::vector<posInfo> & scores, 
		const Eigen::Vector4i & rows, const Eigen::Vector4i & cols,
		std::vector<Eigen::MatrixXd> & scoreMatricies);*/

	void analyzePlacementWeighted(const std::vector<Eigen::SparseMatrix<double> > & fpPyramid,
		const std::vector<Eigen::SparseMatrix<double> > & erodedFpPyramid,
		const std::string & scanName, const std::string & rotationFile,
		const std::string & zerosFile, const std::string & maskName);

	void createFPPyramidsWeighted(const Eigen::SparseMatrix<double> & weightedFloorPlan,
		std::vector<Eigen::SparseMatrix<double> > & fpPyramid,
		std::vector<Eigen::SparseMatrix<double> > & erodedFpPyramid);
} //namespace place





#endif