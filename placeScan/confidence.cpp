#include "placeScan_confidence.h"

#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>


void confidence::findMostConfidentPlacements(std::vector<moreInfo> & info) {
  std::vector<std::string> pointFileNames;
  std::vector<std::string> rotationFileNames;
  std::vector<std::string> zerosFileNames;

  place::parseFolders(pointFileNames, rotationFileNames, zerosFileNames);
  const int numScans = pointFileNames.size();

  std::vector<moreInfo> scoreInfo;

  for (int i = 0; i < FLAGS_stopNumber; ++i) {
    const std::string imageName = FLAGS_dmFolder + pointFileNames[i];
    confidence::loadInPlacement(imageName, scoreInfo, i);
  }

  std::vector<int> localMins;
  confidence::findLocalMinima(scoreInfo, localMins);

  for (auto & index : localMins) {
    info.push_back(scoreInfo[index]);
  }
}

void confidence::loadInPlacement(const std::string & scanName,
  std::vector<moreInfo> & scoreVec, const int scanNum) {
  const std::string placementName = FLAGS_preDone +
  scanName.substr(scanName.find("_")-3, 3)
  + "_placement_" + scanName.substr(scanName.find(".")-3, 3) + ".dat";
  std::ifstream in(placementName, std::ios::in | std::ios::binary);

  int numLoc, numGlob;
  in.read(reinterpret_cast<char *>(&numLoc), sizeof(numLoc));
  in.read(reinterpret_cast<char *>(&numGlob), sizeof(numGlob));
  std::vector<posInfo> scoretmp;
  for (int i = 0; i < numLoc + numGlob; ++i) {
    posInfo tmp;
    in.read(reinterpret_cast<char *>(&tmp), sizeof(tmp));
    scoretmp.push_back(tmp);
  }
  double minScore = 2e20;
  for (auto & s : scoretmp) {
    minScore = std::min(s.score, minScore);
  }

  for (auto s : scoretmp)
    if (s.score == minScore)
      scoreVec.push_back({s, scanNum});
  }

  void confidence::findLocalMinima(const std::vector<moreInfo> & scoreInfo,
    std::vector<int> & localMins) {
    double aveUnexp = 0;
    for (int i = 0; i < scoreInfo.size(); ++i) {
      const posInfo * current = &scoreInfo[i].s;
      aveUnexp += current->scanFP/current->scanPixels;
    }
    aveUnexp /= scoreInfo.size();
    double sigUnexp = 0;
    for (int i = 0; i < scoreInfo.size(); ++i) {
      const posInfo * current = &scoreInfo[i].s;
      const double tmp = current->scanFP/current->scanPixels - aveUnexp;
      sigUnexp += tmp*tmp;
    }

    sigUnexp /= scoreInfo.size();
    sigUnexp = sqrt(sigUnexp);
    const double cutOff = aveUnexp + 0.5*sigUnexp;

    std::cout << aveUnexp << std::endl;
    for (int i = 0; i < scoreInfo.size(); ++i) {
      const posInfo * current = &scoreInfo[i].s;
      if (current->scanFP/current->scanPixels < cutOff)
        localMins.push_back(i);
    }
  }

/*void confidence::lineInersectionKernel(){
  Eigen::MatrixXd kernel (5,5);
  for(int i = 0; i < 5; ++i) {
    kernel(i,2) = 1;
    kernel(2,i) = -1;
  }
  kernel(3,3) = 0;
  Eigen::MatrixXd kernelT = kernel.transpose();
  Eigen::MatrixXd currentFPNS = Eigen::MatrixXd(currentFP);
  Eigen::MatrixXd currentScanNS = Eigen::MatrixXd(currentScan);


  Eigen::MatrixXd diff = currentScanNS - currentFPNS;
  
  double score1, score2;
  score1 = score2 = 0.0;

  for(int j = 0; j < diff.cols(); ++j) {
    for(int i = 0; i < diff.rows(); ++i) {
      for(int k = 0; k < kernel.cols(); ++k) {
        for(int l = 0; l < kernel.rows(); ++l) {
          if(i+l-2 < 0 || i+l-2 >= diff.rows())
            continue;
          if(k+j-2 < 0 || k+j-2 >= diff.cols())
            continue;
          score1 += std::abs(kernel(l,k)*diff(i+l-2, k+j-2));
          score2 += std::abs(kernelT(l,k)*diff(i+l-2, k+j-2));
        }
      }
    }
  }
  score = std::max(score1, score2);
}*/