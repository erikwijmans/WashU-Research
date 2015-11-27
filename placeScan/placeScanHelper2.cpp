#include "placeScan_placeScanHelper2.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>

#include <math.h>

#include <opencv2/core/eigen.hpp>

void place::createWeightedFloorPlan (Eigen::SparseMatrix<double> & weightedFloorPlan) {
	std::vector<std::string> pointFileNames;
  std::vector<std::string> rotationFileNames;
  std::vector<std::string> zerosFileNames;

  place::parseFolders(pointFileNames, rotationFileNames, zerosFileNames, NULL);
  const int numScans = pointFileNames.size();

  std::vector<std::vector<place::moreInfo> > scoreInfo;
  scoreInfo.resize(numScans);

  for (int i = 0; i < numScans; ++i) {
    const std::string imageName = FLAGS_dmFolder + pointFileNames[i];
    place::loadInPlacement(imageName, scoreInfo[i], i);
  }

  Eigen::MatrixXd weightedFp = Eigen::MatrixXd::Zero(floorPlan.rows, floorPlan.cols);
  for(int i = 0; i < floorPlan.rows; ++i) {
    const uchar * src = floorPlan.ptr<uchar>(i);
    for(int j = 0; j < floorPlan.cols; ++j) {
      if(src[j] != 255)
        weightedFp(i,j) = src[j]/255.0;
    }
  }
  

  for(auto & vec : scoreInfo) {
  	const std::string scanName = FLAGS_dmFolder + pointFileNames[vec[0].scanNum];
  	const std::string rotationFile = FLAGS_rotFolder + rotationFileNames[vec[0].scanNum];
  	const std::string zerosFile = FLAGS_zerosFolder + zerosFileNames[vec[0].scanNum];

  	std::vector<cv::Mat> rotatedScans, toTrim;
  	place::loadInScans(scanName, rotationFile, &zerosFile, toTrim);
  	place::trimScans(toTrim, rotatedScans);
  	std::vector<Eigen::SparseMatrix<double> > rSSparse;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));

  	for(auto & scan : rotatedScans) {
      cv::Mat dst;
      cv::erode(scan, dst, element);
  		rSSparse.push_back(scanToSparse(dst));
  	}
  	toTrim.clear();
  	rotatedScans.clear();

  	for(auto & m : vec) {
  		const Eigen::SparseMatrix<double> & currentScan = rSSparse[m.s.rotation];
  		const int xOffset = m.s.x - zeroZero[m.s.rotation][0];
  		const int yOffset = m.s.y - zeroZero[m.s.rotation][1];

  		for(int i = 0; i < currentScan.outerSize(); ++i) {
  			for(Eigen::SparseMatrix<double>::InnerIterator it (currentScan, i); it; ++it) {
          if(yOffset + it.row() >= weightedFp.rows() || yOffset + it.row() < 0)
            continue;
          if(xOffset + it.col() >= weightedFp.cols() || xOffset + it.col() < 0)
            continue;

          if(weightedFp(yOffset + it.row(), xOffset + it.col()) != 0)
  				  weightedFp(yOffset + it.row(), xOffset + it.col()) += it.value();
  			}
  		}
  	}
  }

  std::vector<Eigen::Triplet<double> > tripletList;
  const double * weightedFpPTR = weightedFp.data();
  const int rows = weightedFp.rows();
  
  for(int i = 0; i < weightedFp.size(); ++i) {
    if(*(weightedFpPTR + i) != 0) {
      int x = floor(i/static_cast<double>(rows));
      int y = i%rows;
      tripletList.push_back(Eigen::Triplet<double>(y, x, *(weightedFpPTR + i)));
    }
  }

  weightedFloorPlan = Eigen::SparseMatrix<double> (floorPlan.rows, floorPlan.cols);
  weightedFloorPlan.setFromTriplets(tripletList.begin(), tripletList.end());

}


void place::loadInPlacement(const std::string & scanName,
  std::vector<place::moreInfo> & scoreVec, const int scanNum) {
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

  for (auto s : scoretmp)
    scoreVec.push_back({s, scanNum});
  
}


void place::displayWeightedFloorPlan(Eigen::SparseMatrix<double> & weightedFloorPlan) {
	double maxV = 0;

	for(int i = 0; i < weightedFloorPlan.outerSize(); ++i) {
		for(Eigen::SparseMatrix<double>::InnerIterator it (weightedFloorPlan, i); it; ++it ) {
			maxV = std::max(maxV, it.value());
		}
	}

	cv::Mat out (weightedFloorPlan.rows(), weightedFloorPlan.cols(), CV_8UC3, cv::Scalar::all(255));
	cv::Mat_<cv::Vec3b> _out = out;


	for(int i = 0; i < weightedFloorPlan.outerSize(); ++i) {
		for(Eigen::SparseMatrix<double>::InnerIterator it (weightedFloorPlan, i); it; ++it ) {
			
			if (it.value() > 1) {
				const int gray = cv::saturate_cast<uchar>(255*it.value()/maxV);
				int red, green, blue;
        if (gray < 128) {
          red = 0;
          green = 2 * gray;
          blue = 255 - blue;
        } else {
          blue = 0;
          red = 2 * (gray - 128);
          green = 255 - red;
        }
        _out(it.row(), it.col())[0] = blue;
        _out(it.row(), it.col())[1] = green;
        _out(it.row(), it.col())[2] = red;
      } else if(it.value() > 0) {
        _out(it.row(), it.col())[0] = 0;
        _out(it.row(), it.col())[1] = 0;
        _out(it.row(), it.col())[2] = 0;
      }
    }
  } 
  out = _out;



  cv::imwrite("Out.png", out);

  cv::resize(out, out, cv::Size(out.cols/8, out.rows/8));

  cvNamedWindow("Preview", CV_WINDOW_NORMAL);
  cv::imshow("Preview", out);
  cv::waitKey(0);
}
