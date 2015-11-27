#ifndef PLACESCAN_PLACESCANHELPER2_H_
#define PLACESCAN_PLACESCANHELPER2_H_

#include "placeScan_placeScanHelper.h"



namespace place {
	typedef struct {
		posInfo s;
		int scanNum;
	} moreInfo;

	void createWeightedFloorPlan (Eigen::SparseMatrix<double> & weightedFloorPlan);

	void loadInPlacement(const std::string & scanName,
	  std::vector<place::moreInfo> & scoreVec, const int scanNum);

	void displayWeightedFloorPlan(Eigen::SparseMatrix<double> & weightedFloorPlan);
}



#endif