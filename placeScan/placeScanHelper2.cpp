#include "placeScan_placeScanHelper2.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>

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

  
  Eigen::MatrixXd weightedFp = Eigen::MatrixXd(place::scanToSparse(floorPlan));

  
  for(auto & vec : scoreInfo) {
  	const std::string scanName = FLAGS_dmFolder + pointFileNames[vec[0].scanNum];
  	const std::string rotationFile = FLAGS_rotFolder + rotationFileNames[vec[0].scanNum];
  	const std::string zerosFile = FLAGS_zerosFolder + zerosFileNames[vec[0].scanNum];

  	std::vector<cv::Mat> rotatedScans, toTrim;
    std::vector<Eigen::Vector2i> zeroZero;
  	place::loadInScans(scanName, rotationFile, zerosFile, toTrim, zeroZero);
  	place::trimScans(toTrim, rotatedScans, zeroZero);
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

  weightedFloorPlan = weightedFp.sparseView();
}


void place::loadInPlacement(const std::string & scanName,
  std::vector<place::moreInfo> & scoreVec, const int scanNum) {
  const std::string placementName = FLAGS_preDone +
  scanName.substr(scanName.find("_")-3, 3)
  + "_placement_" + scanName.substr(scanName.find(".")-3, 3) + ".dat";
  std::ifstream in(placementName, std::ios::in | std::ios::binary);

  int num;
  in.read(reinterpret_cast<char *>(&num), sizeof(num));
  std::vector<place::posInfo> scoretmp;
  for (int i = 0; i < num; ++i) {
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


void place::displayWeightedFloorPlan(Eigen::SparseMatrix<double> & weightedFloorPlan) {
	if(!FLAGS_previewOut)
    return;
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
      if(it.value() > 0) {
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
      }
			
    }
  } 
  out = _out;

  cvNamedWindow("Preview", CV_WINDOW_NORMAL);
  cv::imshow("Preview", out);
  cv::waitKey(0);
}

void place::createGraph(Eigen::MatrixXd & adjacencyMatrix,
  std::vector<place::node> & nodes, 
  std::vector<std::vector<Eigen::Vector2i> > & zeroZeros) {

  std::vector<std::string> pointFileNames;
  std::vector<std::string> rotationFileNames;
  std::vector<std::string> zerosFileNames;
  std::vector<std::string> freeFileNames;

  place::parseFolders(pointFileNames, rotationFileNames, zerosFileNames, &freeFileNames);
  const int numScans = pointFileNames.size();

  for (int i = 0; i < numScans; ++i) {
    const std::string imageName = FLAGS_dmFolder + pointFileNames[i];
    place::loadInPlacementGraph(imageName, nodes, i);
  }

  std::vector<std::vector<Eigen::MatrixXb> > scans, masks;
  zeroZeros.resize(numScans);

  for(int i = 0; i < numScans; ++i) {

    const std::string scanName = FLAGS_dmFolder + pointFileNames[i];
    const std::string rotationFile = FLAGS_rotFolder + rotationFileNames[i];
    const std::string zerosFile = FLAGS_zerosFolder + zerosFileNames[i];
    const std::string maskName = FLAGS_dmFolder + freeFileNames[i];

    std::vector<cv::Mat> toTrimScans, toTrimMasks, 
      trimmedScans, trimmedMasks, toTrimMasksD;
    place::loadInScansAndMasks(scanName, rotationFile, zerosFile, 
    maskName, toTrimScans, toTrimMasks, zeroZeros[i]);

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(13,13));
    for(auto & src : toTrimMasks) {
      cv::Mat dst;
      cv::dilate(src, dst, element);
      toTrimMasksD.push_back(dst);
    }

    place::trimScansAndMasks(toTrimScans, toTrimMasksD, 
      trimmedScans, trimmedMasks, zeroZeros[i]);


    std::vector<Eigen::MatrixXb> tmpvec;
    for(auto & scan : trimmedScans) {
      Eigen::MatrixXb tmp = Eigen::MatrixXb::Zero(scan.rows, scan.cols);
      for(int j = 0; j < scan.rows; ++j) {
        const uchar * src = scan.ptr<uchar>(j);
        for(int k = 0; k < scan.cols; ++k) {
          if(src[k] != 255) {
            const double confidence = 1.0 - src[k]/255.0;
              if(confidence > 0.75)
                tmp(j,k) = static_cast<char>(1);
          }
        }
      }
      tmpvec.push_back(tmp);
    }
    scans.push_back(tmpvec);
    tmpvec.clear();

    for(auto & mask : trimmedMasks) {
      Eigen::MatrixXb tmp = Eigen::MatrixXb::Zero(mask.rows, mask.cols);
      for(int j = 0; j < mask.rows; ++j) {
        const uchar * src = mask.ptr<uchar>(j);
        for(int k = 0; k < mask.cols; ++k) {
          if(src[k] != 255) 
            tmp(j,k) = static_cast<char>(1);
        }
      }
      tmpvec.push_back(tmp);
    }
    masks.push_back(tmpvec);
  }

  const int numNodes = nodes.size();
  adjacencyMatrix = Eigen::MatrixXd(numNodes, numNodes);

  place::weightEdges(nodes, scans, masks, zeroZeros, adjacencyMatrix);

  // place::displayGraph(adjacencyMatrix, nodes, scans, zeroZeros);

  std::vector<place::node> bestLabels;
  place::findBestLabels(adjacencyMatrix, nodes, bestLabels);
  place::displayLongestPath(bestLabels, scans, zeroZeros);
}

void place::weightEdges(const std::vector<place::node> & nodes, 
  const std::vector<std::vector<Eigen::MatrixXb> > & scans, 
  const std::vector<std::vector<Eigen::MatrixXb> > & masks,
  const std::vector<std::vector<Eigen::Vector2i> > & zeroZeros,
  Eigen::MatrixXd & adjacencyMatrix) {
  const int rows = adjacencyMatrix.rows();
  const int cols = adjacencyMatrix.cols();

  for(int i = 0; i < cols; ++i) {
    for(int j = 0; j < rows; ++j) {
      const place::node & nodeA = nodes[i];
      const place::node & nodeB = nodes[j];
      if(nodeA.color == nodeB.color) {
        adjacencyMatrix(j,i) = 0;
        continue;
      }
      if(i > j) {
        adjacencyMatrix(j,i) = adjacencyMatrix(i,j);
        continue;
      }
      

      const Eigen::MatrixXb & aScan = scans[nodeA.color][nodeA.s.rotation];
      const Eigen::MatrixXb & aMask = masks[nodeA.color][nodeA.s.rotation];

      const Eigen::MatrixXb & bScan = scans[nodeB.color][nodeB.s.rotation];
      const Eigen::MatrixXb & bMask = masks[nodeB.color][nodeB.s.rotation];

      const Eigen::Vector2i & zeroZeroA = zeroZeros[nodeA.color][nodeA.s.rotation];
      const Eigen::Vector2i & zeroZeroB = zeroZeros[nodeB.color][nodeB.s.rotation];

      place::rect aBox, bBox;
      aBox.X1 = nodeA.s.x - zeroZeroA[0];
      aBox.X2 = aBox.X1 + aScan.cols() - 1;
      aBox.Y1 = nodeA.s.y - zeroZeroA[1];
      aBox.Y2 = aBox.Y1 + aScan.rows() - 1;

      bBox.X1 = nodeB.s.x - zeroZeroB[0];
      bBox.X2 = bBox.X1 + bScan.cols() - 1;
      bBox.Y1 = nodeB.s.y - zeroZeroB[1];
      bBox.Y2 = bBox.Y1 + bScan.rows() - 1;

      place::rect XSection;
      XSection.X1 = std::max(aBox.X1, bBox.X1);
      XSection.Y1 = std::max(aBox.Y1, bBox.Y1);
      XSection.X2 = std::min(aBox.X2, bBox.X2);
      XSection.Y2 = std::min(aBox.Y2, bBox.Y2);

      if (XSection.X1 > XSection.X2 || 
        XSection.Y1 > XSection.Y2) {
        const double weight = 3000*(std::exp(-nodeA.s.score) + std::exp(-nodeB.s.score));
        adjacencyMatrix(j,i) = weight;
      } else {
        const int Xrows = XSection.Y2 - XSection.Y1 + 1;
        const int Xcols = XSection.X2 - XSection.X1 + 1;
        Eigen::MatrixXb XSectionAScan, XSectionAMask,
          XSectionBScan, XSectionBMask;
        place::rect crossWRTA, crossWRTB;

        crossWRTA.X1 = XSection.X1 - aBox.X1;
        crossWRTA.X2 = XSection.X2 - aBox.X1;
        crossWRTA.Y1 = XSection.Y1 - aBox.Y1;
        crossWRTA.Y2 = XSection.Y2 - aBox.Y1;

        crossWRTB.X1 = XSection.X1 - bBox.X1;
        crossWRTB.X2 = XSection.X2 - bBox.X1;
        crossWRTB.Y1 = XSection.Y1 - bBox.Y1;
        crossWRTB.Y2 = XSection.Y2 - bBox.Y1;

        XSectionAMask = aMask.block(crossWRTA.Y1, crossWRTA.X1,
          Xrows, Xcols);
        XSectionAScan = aScan.block(crossWRTA.Y1, crossWRTA.X1,
          Xrows, Xcols);

        XSectionBMask = bMask.block(crossWRTB.Y1, crossWRTB.X1,
          Xrows, Xcols);
        XSectionBScan = bScan.block(crossWRTB.Y1, crossWRTB.X1,
          Xrows, Xcols);

        double pointAgreement = 0;
        for(int k = 0; k < Xcols; ++k) {
          for(int l = 0; l < Xrows; ++l) {
            if(XSectionAScan(l,k) != 0 && XSectionBScan(l,k) != 0)
              ++pointAgreement;
          }
        }

        double freeSpaceAgreementA = 0;
        for(int k = 0; k < Xcols; ++k) {
          for(int l = 0; l < Xrows; ++l) {
            if(XSectionAScan(l,k) != 0 && XSectionBMask(l,k) != 0)
              --freeSpaceAgreementA;
          }
        }

        double freeSpaceAgreementB = 0;
        for(int k = 0; k < Xcols; ++k) {
          for(int l = 0; l < Xrows; ++l) {
            if(XSectionBScan(l,k) != 0 && XSectionAMask(l,k) != 0)
              --freeSpaceAgreementB;
          }
        }

        const double weight = 3000*(std::exp(-nodeA.s.score) + std::exp(-nodeB.s.score)) 
          + 2.0*pointAgreement + 1/5.0*freeSpaceAgreementA + 1/5.0*freeSpaceAgreementB;
        adjacencyMatrix(j,i) = weight;
      }
    }
  }
}

void place::loadInPlacementGraph(const std::string & imageName, 
  std::vector<place::node> & nodes, const int num) {
  const std::string placementName = FLAGS_preDone +
    imageName.substr(imageName.find("_")-3, 3)
    + "_placement_" + imageName.substr(imageName.find(".")-3, 3) + ".dat";
  std::ifstream in(placementName, std::ios::in | std::ios::binary);

  int numToLoad;
  in.read(reinterpret_cast<char *>(&numToLoad), sizeof(numToLoad));
  std::vector<place::posInfo> scoretmp;
  for (int i = 0; i < numToLoad; ++i) {
    place::posInfo tmp;
    in.read(reinterpret_cast<char *>(&tmp), sizeof(tmp));
    scoretmp.push_back(tmp);
  }

  for(int i = 0; i < scoretmp.size(); ++ i)
    nodes.push_back({scoretmp[i], num});
}

void place::trimScansAndMasks(const std::vector<cv::Mat> & toTrimScans, 
  const std::vector<cv::Mat> & toTrimMasks, std::vector<cv::Mat> & trimmedScans,
  std::vector<cv::Mat> & trimmedMasks, std::vector<Eigen::Vector2i> & zeroZero) {
  for(int i = 0; i < toTrimScans.size(); ++i) {
    const cv::Mat & currentScan = toTrimScans[i];
    const cv::Mat & currentMask = toTrimMasks[i];
    int minRow = currentScan.rows;
    int minCol = currentScan.cols;
    int maxRow = 0;
    int maxCol = 0;

    for (int i = 0; i < currentScan.rows; ++i) {
      const uchar * src = currentScan.ptr<uchar>(i);
      for (int j = 0; j < currentScan.cols; ++j) {
        if(src[j]!=255) {
          minRow = std::min(i, minRow);
          minCol = std::min(j, minCol);
          maxRow = std::max(i, maxRow);
          maxCol = std::max(j, maxCol);
        }
      }
    }

    cv::Mat trimmedMask (maxRow - minRow + 1, maxCol - minCol + 1, CV_8UC1);
    cv::Mat trimmedScan (maxRow - minRow + 1, maxCol - minCol + 1, CV_8UC1);
    for (int i = minRow; i < maxRow + 1; ++i) {
      const uchar * scanSrc = currentScan.ptr<uchar>(i);
      const uchar * maskSrc = currentMask.ptr<uchar>(i);
      uchar * scanDst = trimmedScan.ptr<uchar>(i-minRow);
      uchar * maskDst = trimmedMask.ptr<uchar>(i - minRow);
      for (int j = minCol; j < maxCol + 1; ++j) {
        scanDst[j-minCol] = scanSrc[j];
        maskDst[j-minCol] = maskSrc[j];
      }
    }
    zeroZero[i][0] -= minCol;
    zeroZero[i][1] -= minRow;
    trimmedScans.push_back(trimmedScan);
    trimmedMasks.push_back(trimmedMask);
  }
}

void place::displayGraph(const Eigen::MatrixXd & adjacencyMatrix, 
  const std::vector<place::node> & nodes,
  const std::vector<std::vector<Eigen::MatrixXb> > & scans, 
  const std::vector<std::vector<Eigen::Vector2i> > & zeroZeros) {

  const int rows = adjacencyMatrix.rows();
  const int cols = adjacencyMatrix.cols();

  for(int i = 0; i < cols; ++i) {
    for(int j = 0; j < rows; ++j) {
      if(adjacencyMatrix(j,i) == 0)
        continue;

      const place::node & nodeA = nodes[i];
      const place::node & nodeB = nodes[j];

      const Eigen::MatrixXb & aScan = scans[nodeA.color][nodeA.s.rotation];
      const Eigen::MatrixXb & bScan = scans[nodeB.color][nodeB.s.rotation];

      const Eigen::Vector2i & zeroZeroA = zeroZeros[nodeA.color][nodeA.s.rotation];
      const Eigen::Vector2i & zeroZeroB = zeroZeros[nodeB.color][nodeB.s.rotation];

      cv::Mat output (fpColor.rows, fpColor.cols, CV_8UC3);
      fpColor.copyTo(output);

      cv::Mat_<cv::Vec3b> _output = output;


      int yOffset = nodeA.s.y - zeroZeroA[1];
      int xOffset = nodeA.s.x - zeroZeroA[0];
      for (int k = 0; k < aScan.cols(); ++k) {
        for(int l = 0; l < aScan.rows(); ++l) {
          if(aScan(l, k) != 0) {
            _output(l + yOffset, k + xOffset)[0]=0;
            _output(l + yOffset, k + xOffset)[1]=0;
            _output(l + yOffset, k + xOffset)[2]=255;
          }
        }
      }

      yOffset = nodeB.s.y - zeroZeroB[1];
      xOffset = nodeB.s.x - zeroZeroB[0];
      for (int k = 0; k < bScan.cols(); ++k) {
        for(int l = 0; l < bScan.rows(); ++l) {
          if(bScan(l, k) != 0) {
            _output(l + yOffset, k + xOffset)[0]=0;
            _output(l + yOffset, k + xOffset)[1]=0;
            _output(l + yOffset, k + xOffset)[2]=255;
          }
        }
      }

      cvNamedWindow("Preview", CV_WINDOW_NORMAL);
      cv::imshow("Preview", output);
      if(!FLAGS_quiteMode) {
        std::cout << "Color A: " << nodeA.color << "  Color B: " << nodeB.color << std::endl;
        std::cout << "Edge weight: " << adjacencyMatrix(j,i) << std::endl;
      }
      cv::waitKey(0);
      ~output;
    }
  }
}

void place::displayLongestPath(const std::vector<place::node> & longestPath,
  const std::vector<std::vector<Eigen::MatrixXb> > & scans, 
  const std::vector<std::vector<Eigen::Vector2i> > & zeroZeros) {
 
  std::cout << "Displaying longestPath with a length of: " << longestPath.size() << std::endl;
 
  for(auto & n : longestPath) {
    cv::Mat output(fpColor.rows, fpColor.cols, CV_8UC3);
    fpColor.copyTo(output);
    cv::Mat_<cv::Vec3b> _output = output;
    const Eigen::MatrixXb & scan = scans[n.color][n.s.rotation];
    const Eigen::Vector2i zeroZero = zeroZeros[n.color][n.s.rotation];
    const int xOffset = n.s.x - zeroZero[0];
    const int yOffset = n.s.y - zeroZero[1];
    
    for(int i = 0; i < scan.cols(); ++i) {
      for(int j = 0; j < scan.rows(); ++j) {
        if(scan(j,i) != 0) {
          _output(yOffset + j, xOffset + i)[0] = 0;
          _output(yOffset + j, xOffset + i)[1] = 0;
          _output(yOffset + j, xOffset + i)[2] = 255;
        }
      }
    }

    cvNamedWindow("Preview", CV_WINDOW_NORMAL);
    cv::imshow("Preview", output);
    cv::waitKey(0);
  }
  
}

void place::findBestLabels(const Eigen::MatrixXd & adjacencyMatrix, 
  const std::vector<place::node> & nodes, std::vector<place::node> & bestLabels) {
  int currentColor = 0;
  place::node currentBest;
  double currentBestScore = 0.0;
  for(int i = 0; i < adjacencyMatrix.cols(); ++i) {
    const place::node * currentNode = &nodes[i];
    if(currentNode->color == currentColor) {
      double score = adjacencyMatrix.col(i).sum();
      if(score > currentBestScore) {
        currentBestScore = score;
        currentBest = *currentNode;
      }
    } else {
      bestLabels.push_back(currentBest);
      currentColor = currentNode->color;
      currentBestScore = adjacencyMatrix.col(i).sum();
      currentBest = *currentNode;
    }
  }
}