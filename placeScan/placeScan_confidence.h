#ifndef PLACESCAN_CONFIDENCE_HPP_
#define PLACESCAN_CONFIDENCE_HPP_

#include "placeScan_placeScanHelper.h"



namespace confidence{
	typedef struct{
		posInfo s;
		double tKernScore;
		int scanNum;
	} moreInfo;
	
	void findMostConfidentPlacements(std::vector<moreInfo> & scoreInfo);

	void loadInPlacement(const std::string & scanName,
		std::vector<moreInfo> & scoreVec, const int scanNum);

	void findLocalMinima(const std::vector<moreInfo> & scoreInfo,
    std::vector<int> & localMins,
    std::vector<std::string> & pointFileNames, 
    std::vector<std::string> & rotationFileNames, 
    std::vector<std::string> & zerosFileNames);

	double TKernel(Eigen::MatrixXd & currentScanNS, 
  	Eigen::MatrixXd & currentFPNS);
}


#endif