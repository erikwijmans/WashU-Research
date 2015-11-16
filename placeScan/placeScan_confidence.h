#ifndef PLACESCAN_CONFIDENCE_HPP_
#define PLACESCAN_CONFIDENCE_HPP_

#include "placeScan_placeScanHelper.h"

typedef struct{
	posInfo s;
	int scanNum;
} moreInfo;

namespace confidence{
	void findMostConfidentPlacements(std::vector<moreInfo> & scoreInfo);

	void loadInPlacement(const std::string & scanName,
		std::vector<moreInfo> & scoreVec, const int scanNum);

	void findLocalMinima(const std::vector<moreInfo> & scoreInfo,
		std::vector<int> & localMins);
}


#endif