#include "placeScan_confidence.h"

#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>


void confidence::findMostConfidentPlacements(std::vector<confidence::moreInfo> & info) {
  std::vector<std::string> pointFileNames;
  std::vector<std::string> rotationFileNames;
  std::vector<std::string> zerosFileNames;

  place::parseFolders(pointFileNames, rotationFileNames, zerosFileNames, NULL);
  const int numScans = pointFileNames.size();

  std::vector<confidence::moreInfo> scoreInfo;

  for (int i = 0; i < numScans; ++i) {
    const std::string imageName = FLAGS_dmFolder + pointFileNames[i];
    confidence::loadInPlacement(imageName, scoreInfo, i);
  }

  std::vector<int> localMins;
  confidence::findLocalMinima(scoreInfo, localMins,
   pointFileNames, rotationFileNames, zerosFileNames);

  for (auto & index : localMins) {
    info.push_back(scoreInfo[index]);
  }
}

void confidence::loadInPlacement(const std::string & scanName,
  std::vector<confidence::moreInfo> & scoreVec, const int scanNum) {
  const std::string placementName = FLAGS_preDone +
  scanName.substr(scanName.find("_")-3, 3)
  + "_placement_" + scanName.substr(scanName.find(".")-3, 3) + ".dat";
  std::ifstream in(placementName, std::ios::in | std::ios::binary);

  int numLoc, numGlob;
  in.read(reinterpret_cast<char *>(&numLoc), sizeof(numLoc));
  in.read(reinterpret_cast<char *>(&numGlob), sizeof(numGlob));
  std::vector<place::posInfo> scoretmp;
  for (int i = 0; i < numLoc + numGlob; ++i) {
    place::posInfo tmp;
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

void confidence::findLocalMinima(const std::vector<confidence::moreInfo> & scoreInfo,
  std::vector<int> & localMins,
  std::vector<std::string> & pointFileNames, 
  std::vector<std::string> & rotationFileNames, 
  std::vector<std::string> & zerosFileNames) {

    Eigen::MatrixXd fpNS = Eigen::MatrixXd(place::scanToSparse(floorPlan));
    for(int i = 0; i < scoreInfo.size(); ++i) {
      const std::string scanName = FLAGS_dmFolder + pointFileNames[scoreInfo[i].scanNum];
      const std::string rotName = FLAGS_rotFolder + rotationFileNames[scoreInfo[i].scanNum];
      const std::string zeroName = FLAGS_zerosFolder + zerosFileNames[scoreInfo[i].scanNum];
      std::vector<cv::Mat> rotatedScans;
      std::vector<Eigen::Vector2i> zeroZero;
      place::loadInScans(scanName, rotName, zeroName, rotatedScans, zeroZero);


    }


    double aveUnexp = 0;
    for (int i = 0; i < scoreInfo.size(); ++i) {
      const place::posInfo * current = &scoreInfo[i].s;
      aveUnexp += current->scanFP/current->scanPixels;
    }
    aveUnexp /= scoreInfo.size();
    double sigUnexp = 0;
    for (int i = 0; i < scoreInfo.size(); ++i) {
      const place::posInfo * current = &scoreInfo[i].s;
      const double tmp = current->scanFP/current->scanPixels - aveUnexp;
      sigUnexp += tmp*tmp;
    }

    sigUnexp /= scoreInfo.size();
    sigUnexp = sqrt(sigUnexp);
    const double cutOff = aveUnexp + 0.5*sigUnexp;

    std::cout << aveUnexp << std::endl;
    for (int i = 0; i < scoreInfo.size(); ++i) {
      const place::posInfo * current = &scoreInfo[i].s;
      if (current->scanFP/current->scanPixels < cutOff)
        localMins.push_back(i);
    }
  }

double confidence::TKernel(Eigen::MatrixXd & currentScanNS, 
  Eigen::MatrixXd & currentFPNS) {
  Eigen::MatrixXd kernel (5,5);
  for(int i = 0; i < 5; ++i) {
    kernel(i,2) = 1;
    kernel(2,i) = -1;
  }
  kernel(3,3) = 0;
  Eigen::MatrixXd kernelT = kernel.transpose();


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
  double score = std::max(score1, score2);
  return score;
}