#ifndef PLACESCAN_PLACE_SCAN_H_
#define PLACESCAN_PLACE_SCAN_H_

#include "placeScan_placeScanHelper.h"

#include <omp.h>


#pragma omp declare reduction (merge : std::vector<posInfo> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp declare reduction (merge : std::vector<int> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))


namespace place{
	void analyzePlacement(const std::vector<Eigen::SparseMatrix<double> > & fpPyramid,
		const std::vector<Eigen::SparseMatrix<double> > & erodedFpPyramid, const std::string & scanName, 
		const std::string & rotationFile, const std::string & zerosFile,
		const std::string & maskName);

	void findLocalMinima(const std::vector<posInfo> & scores, std::vector<int> & localMinima, 
		const float bias);

	void createPyramid(std::vector<Eigen::SparseMatrix<double> > & pyramid);

	void createPyramid( std::vector<std::vector<Eigen::SparseMatrix<double> > >  & pyramid);

	void trimScanPryamids(const std::vector<std::vector<Eigen::SparseMatrix<double> > >  & rSSparsePyramid,
		std::vector<std::vector<Eigen::SparseMatrix<double> > > & rSSparsePyramidTrimmed, 
		const std::vector<std::vector<Eigen::SparseMatrix<double> > >  & erodedSparsePyramid,
		std::vector<std::vector<Eigen::SparseMatrix<double> > > & erodedSparsePyramidTrimmed,
		const std::vector<std::vector<Eigen::SparseMatrix<double> > >  & eMaskPyramid,
		std::vector<std::vector<Eigen::SparseMatrix<double> > >  & eMaskPyramidTrimmed);

	void findPlacement(const Eigen::SparseMatrix<double> & fp, 
		const std::vector<Eigen::SparseMatrix<double> > & scans,const Eigen::SparseMatrix<double> & fpE, 
		const std::vector<Eigen::SparseMatrix<double> > & scansE, 
		const std::vector<Eigen::MatrixXd> & masks, const Eigen::Vector4d numPixelsUnderMask,
		const std::vector<Eigen::Vector3i> & points,
		std::vector<posInfo> & scores);

	void findPointsToAnalyze(const std::vector<posInfo> & scores, const std::vector<int> & localMinima,
		std::vector<Eigen::Vector3i> & pointsToAnalyze);

	void findGlobalMinima(const std::vector<posInfo> & scores, const std::vector<int> & localMinima);

	Eigen::MatrixXd distanceTransform(const Eigen::SparseMatrix<double> & image);

	void createFPPyramids(const cv::Mat & floorPlan, 
		std::vector<Eigen::SparseMatrix<double> > & fpPyramid,  
		std::vector<Eigen::SparseMatrix<double> > & erodedFpPyramid);

	void findNumPixelsUnderMask(const std::vector<std::vector<Eigen::SparseMatrix<double> > > & rSSparsePyramidTrimmed,
		const std::vector<std::vector<Eigen::MatrixXd> > & eMaskPyramidTrimmedNS,
		std::vector<Eigen::Vector4d> & numPixelsUnderMask);

	void blurMinima(const std::vector<posInfo> & scores, 
		const Eigen::Vector4i & rows, const Eigen::Vector4i & cols,
		std::vector<Eigen::MatrixXd> & scoreMatricies);
}





#endif