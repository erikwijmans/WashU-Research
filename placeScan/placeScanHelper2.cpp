 #include "placeScan_placeScanHelper2.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>

#include <math.h>
#include <dirent.h>
#include <omp.h>

#include <opencv2/core/eigen.hpp>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/discretespace.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/inference/trws/trws_trws.hxx>
#include <unordered_map>
#include "gurobi_c++.h"

static std::ostream & operator<<(std::ostream & stream, const place::cube & print) {
  stream << "(" << print.X1 << ", " << print.Y1 << ", " << print.Z1 << ")" << std::endl;
  stream << "      " << "(" << print.X2 << ", " << print.Y2 << ", " << print.Z2 <<  ")";
  return stream;
}

static std::ostream & operator<<(std::ostream & stream, const place::edgeWeight & print) {
  stream << "edgeWeight: " << print.w << std::endl;
  stream << print.pA << "  " << print.feA << std::endl;
  stream << print.fx << "  " << print.feB;
  return stream;
}

#pragma omp declare reduction (summer: double : omp_out += omp_in)

static void displayVoxelGrid(const std::vector<Eigen::MatrixXb> & voxelB,
  const std::string & windowName) {
  Eigen::MatrixXd collapsed (voxelB[0].rows(), voxelB[0].cols());

  for(int i = 0; i < collapsed.cols(); ++i) {
    for(int j = 0; j < collapsed.rows(); ++j) {
      double sum = 0;
      for(int k = 0; k < voxelB.size(); ++k) {
        sum += voxelB[k](j,i);
      }
      collapsed(j,i) = sum;
    }
  }

  double average, sigma;
  average = sigma = 0;
  int count = 0;
  const double * dataPtr = collapsed.data();
  for(int i = 0; i < collapsed.size(); ++i) {
    if(*(dataPtr+ i)) {
      ++count;
      average+= *(dataPtr + i);
    }
  }

  average = average/count;

  for(int i = 0; i < collapsed.size(); ++i) {
    if(*(dataPtr + i) !=0)
      sigma += (*(dataPtr + i) - average)*(*(dataPtr + i)- average);
  }

  sigma = sigma/(count-1);
  sigma = sqrt(sigma);
  

  cv::Mat heatMap (collapsed.rows(), collapsed.cols(), CV_8UC3, cv::Scalar::all(255));
  for (int i = 0; i < heatMap.rows; ++i) {
    uchar * dst = heatMap.ptr<uchar>(i);
    for (int j = 0; j < heatMap.cols; ++j) {
      if(collapsed(i,j)){
        const int gray = cv::saturate_cast<uchar>(
          255.0 * (collapsed(i,j) - average) 
            / (1.0 * sigma));
        int red, green, blue;
        if (gray < 128) {
          red = 0;
          blue = 2 * gray;
          green = 255 - blue;
        } else {
          blue = 0;
          red = 2 * (gray - 128);
          green = 255 - red;
        }
        dst[j*3] = blue;
        dst[j*3 +1] = green;
        dst[j*3 + 2] = red;
      }
    } 
  }

  cv::imshow(windowName, heatMap);
}

static void displayVoxelGrid(const std::vector<Eigen::MatrixXb> & voxelA,
  const std::vector<Eigen::MatrixXb> & voxelB,
  const place::cube & aRect, const place::cube & bRect) {

  const int z = aRect.Z2 - aRect.Z1 + 1;
  const int Xrows = aRect.Y2 - aRect.Y1 + 1;
  const int Xcols = aRect.X2 - aRect.X1 + 1;

  for(int k = 0; k < z; ++k) {
  Eigen::MatrixXd collapsedA (Xrows, Xcols);
  Eigen::MatrixXd collapsedB (Xrows, Xcols);

   
  for(int i = 0; i < Xcols; ++i) {
    for(int j = 0; j < Xrows; ++j) {
      double sumA = 0, sumB = 0;
     
        sumA += voxelA[k + aRect.Z1](j + aRect.Y1, i + aRect.X1);
        sumB += voxelB[k + bRect.Z1](j + bRect.Y1, i + bRect.X1);
      collapsedA(j,i) = sumA;
      collapsedB(j,i) = sumB;
    }
  }

  if(collapsedB.sum() == 0 && collapsedA.sum() == 0)
    continue;

  double averageA = 0, sigmaA = 0, averageB = 0, sigmaB = 0;
  int countA = 0, countB = 0;
  const double * dataPtrA = collapsedA.data();
  const double * dataPtrB = collapsedB.data();
  for(int i = 0; i < collapsedA.size(); ++i) {
    if(*(dataPtrA+ i)) {
      ++countA;
      averageA += *(dataPtrA + i);
    }
    if(*(dataPtrB+ i)) {
      ++countB;
      averageB += *(dataPtrB + i);
    }
  }

  averageA = averageA/countA;
  averageB = averageB/countB;

  for(int i = 0; i < collapsedA.size(); ++i) {
    if(*(dataPtrA + i) !=0)
      sigmaA += (*(dataPtrA + i) - averageA)*(*(dataPtrA + i)- averageA);

    if(*(dataPtrB + i) !=0)
      sigmaB += (*(dataPtrB + i) - averageB)*(*(dataPtrB + i)- averageB);
  }

  sigmaA = sigmaA/(countA-1);
  sigmaA = sqrt(sigmaA);

  sigmaB = sigmaB/(countB-1);
  sigmaB = sqrt(sigmaB);

  


  cv::Mat heatMap (Xrows, Xcols, CV_8UC3, cv::Scalar::all(255));
  for (int i = 0; i < heatMap.rows; ++i) {
    uchar * dst = heatMap.ptr<uchar>(i);
    for (int j = 0; j < heatMap.cols; ++j) {
      if(collapsedA(i,j) && collapsedB(i,j)) {
        const int grayA = cv::saturate_cast<uchar>(
          255.0 * (collapsedA(i,j) - averageA) 
            / (1.0 * sigmaA));
        const int grayB = cv::saturate_cast<uchar>(
          255.0 * (collapsedB(i,j) - averageB) 
            / (1.0 * sigmaB));
        int red = 255, green = 0, blue = 255;
        dst[j*3] = blue;
        dst[j*3 +1] = green;
        dst[j*3 + 2] = red;
      } else if(collapsedA(i,j)){
        const int gray = cv::saturate_cast<uchar>(
          255.0 * (collapsedA(i,j) - averageA) 
            / (1.0 * sigmaA));
        int red = 255, green = 0, blue = 0;
        dst[j*3] = blue;
        dst[j*3 +1] = green;
        dst[j*3 + 2] = red;
      } else if(collapsedB(i,j)){
        const int gray = cv::saturate_cast<uchar>(
          255.0 * (collapsedB(i,j) - averageB) 
            / (1.0 * sigmaB));
        int red = 0, green = 0, blue =255;
        dst[j*3] = blue;
        dst[j*3 +1] = green;
        dst[j*3 + 2] = red;
      }
    } 
  }
  cvNamedWindow("Preview", CV_WINDOW_NORMAL);
  cv::imshow("Preview", heatMap);
  cv::waitKey(0);
}

}

void place::createGraph(Eigen::MatrixXS & adjacencyMatrix,
  std::vector<place::node> & nodes, 
  std::vector<std::vector<Eigen::Vector2i> > & zeroZeros,
  std::vector<const place::node *> & bestNodes) {



  const double startTime = omp_get_wtime();

  std::vector<std::string> pointFileNames, zerosFileNames, freeFileNames;

  place::parseFolders(pointFileNames,
    zerosFileNames, &freeFileNames);

 std::vector<std::string > pointVoxelFileNames, freeVoxelFileNames;
 {
    std::string folder = FLAGS_voxelFolder + "R" + std::to_string(0) + "/";
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (folder.data())) != NULL) {
      while ((ent = readdir (dir)) != NULL) {
        std::string fileName = ent->d_name;
        if(fileName != ".." && fileName != "." 
          && fileName.find("point") != std::string::npos){
          pointVoxelFileNames.push_back(fileName);
        } else if (fileName != ".." && fileName != "." 
          && fileName.find("freeSpace") != std::string::npos) {
          freeVoxelFileNames.push_back(fileName);
        }
      }
      closedir (dir);
    }  else {
      /* could not open directory */
      perror ("");
      exit(-1);
    }
    std::sort(pointVoxelFileNames.begin(), pointVoxelFileNames.end());
    std::sort(freeVoxelFileNames.begin(), freeVoxelFileNames.end());
  }
  

  std::vector<std::string> metaDataFiles;
  DIR *dir;
  struct dirent *ent;
  const std::string metaDataFolder = FLAGS_voxelFolder + "metaData/";
  if ((dir = opendir (metaDataFolder.data())) != NULL) {
    while ((ent = readdir (dir)) != NULL) {
      std::string fileName = ent->d_name;
      if(fileName != ".." && fileName != ".")
        metaDataFiles.push_back(fileName);
    }
    closedir (dir);
  }  else {
    /* could not open directory */
    perror ("");
    exit(-1);
  }
  std::sort(metaDataFiles.begin(), metaDataFiles.end());

  const int numScans = pointFileNames.size();

  std::vector<std::vector<place::metaData> > voxelInfo;
  for(int i = 0; i < numScans; ++i) {
    const std::string metaName = metaDataFolder + metaDataFiles[i];
    std::ifstream in (metaName, std::ios::in | std::ios::binary);
    place::metaData tmp;
    std::vector<place::metaData> tmpVec;
    for(int i = 0; i < NUM_ROTS; ++i) {
      in.read(reinterpret_cast<char *>(&tmp), sizeof(tmp));
      tmpVec.push_back(tmp);
    }
    voxelInfo.push_back(tmpVec);
  }

  std::vector<std::vector<Eigen::MatrixXb> > scans, masks;
  zeroZeros.resize(numScans);
  place::loadInScansGraph(pointFileNames, freeFileNames,
    zerosFileNames, scans, masks, zeroZeros);
  
  const int numToParse = 10;
  const int nodeStart = 0;
  for (int i = nodeStart; i < numToParse + nodeStart; ++i) {
    const std::string imageName = FLAGS_dmFolder + pointFileNames[i];
    place::loadInPlacementGraph(imageName, nodes, i);
  }


  const int numNodes = nodes.size();
  adjacencyMatrix = Eigen::MatrixXS (numNodes, numNodes);

 
  place::weightEdges(nodes, voxelInfo, 
    pointVoxelFileNames, freeVoxelFileNames, adjacencyMatrix);

  const double endTime = omp_get_wtime();
  std::cout << "Time: " << endTime - startTime << std::endl;

  // place::displayGraph(adjacencyMatrix, nodes, scans, zeroZeros);

  std::map<std::vector<int>, double> highOrder;
  place::createHigherOrderTerms(scans, zeroZeros, nodes, highOrder);

  // while(true){
    /*bestNodes.clear();
    place::TRWSolver(adjacencyMatrix, nodes, bestNodes);
    place::displayTRW(bestNodes, scans, zeroZeros);*/

    bestNodes.clear();
    place::MIPSolver(adjacencyMatrix, highOrder, nodes, bestNodes);
    place::displayBest(bestNodes, scans, zeroZeros);
  // }
}

void place::loadInScansGraph(const std::vector<std::string> & pointFileNames,
  const std::vector<std::string> & freeFileNames,
  const std::vector<std::string> & zerosFileNames,
  std::vector<std::vector<Eigen::MatrixXb> > & scans,
  std::vector<std::vector<Eigen::MatrixXb> > & masks,
  std::vector<std::vector<Eigen::Vector2i> > & zeroZeros) {

  for(int i = 0; i < zeroZeros.size(); ++i) {

    const std::string scanName = pointFileNames[i];
    const std::string zerosFile = FLAGS_zerosFolder + zerosFileNames[i];
    const std::string maskName = freeFileNames[i];

    std::vector<cv::Mat> toTrimScans, toTrimMasks, 
      trimmedScans, trimmedMasks, toTrimMasksD;
    place::loadInScansAndMasks(scanName, zerosFile, 
    maskName, toTrimScans, toTrimMasks, zeroZeros[i]);

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
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
}

void place::weightEdges(const std::vector<place::node> & nodes,
  const std::vector<std::vector<place::metaData> > & voxelInfo,
  const std::vector<std::string> & pointVoxelFileNames,
  const std::vector<std::string> & freeVoxelFileNames,
  Eigen::MatrixXS & adjacencyMatrix) {

  if(!FLAGS_redo && place::reloadGraph(adjacencyMatrix))
    return;

  const int rows = adjacencyMatrix.rows();
  const int cols = adjacencyMatrix.cols();

  #pragma omp parallel
  {
    int voxelAColor = 1e6, voxelARot = 5;
    place::voxel aPoint, aFree;
    
    #pragma omp for nowait schedule (dynamic)
    for(int i = 0; i < cols ; ++i) {
      const place::node & nodeA = nodes[i];
      int voxelBcolor = 1e6, voxelBRot = 5;
      place::voxel bPoint, bFree;
      for(int j = 0; j < rows; ++j) {
        const place::node & nodeB = nodes[j];

        if(nodeA.color == nodeB.color) {
          adjacencyMatrix(j, i) = {0, 0, 0, 0, 0};
          continue;
        }
        if(i > j) {
          adjacencyMatrix(j, i) = adjacencyMatrix(i,j);
          continue;
        }

        /*if(nodeA.color != 20 && nodeB.color != 20)
          continue;*/

        const place::metaData & metaA = voxelInfo[nodeA.color][nodeA.s.rotation];
        
        const place::metaData & metaB = voxelInfo[nodeB.color][nodeB.s.rotation];

        place::cube aBox, bBox;
        aBox.X1 = nodeA.s.x*(voxelsPerMeter/pixelsPerMeter) - metaA.zZ[0];
        aBox.X2 = aBox.X1 + metaA.x - 1;
        aBox.Y1 = nodeA.s.y*(voxelsPerMeter/pixelsPerMeter) - metaA.zZ[1];
        aBox.Y2 = aBox.Y1 + metaA.y - 1;
        aBox.Z1 = 0 - metaA.zZ[2];
        aBox.Z2 = aBox.Z1 + metaA.z - 1;

        bBox.X1 = nodeB.s.x*(voxelsPerMeter/pixelsPerMeter) - metaB.zZ[0];
        bBox.X2 = bBox.X1 + metaB.x - 1;
        bBox.Y1 = nodeB.s.y*(voxelsPerMeter/pixelsPerMeter) - metaB.zZ[1];
        bBox.Y2 = bBox.Y1 + metaB.y - 1;
        bBox.Z1 = 0 - metaB.zZ[2];
        bBox.Z2 = bBox.Z1 + metaB.z - 1;



        place::cube XSection;
        XSection.X1 = std::max(aBox.X1, bBox.X1);
        XSection.Y1 = std::max(aBox.Y1, bBox.Y1);
        XSection.X2 = std::min(aBox.X2, bBox.X2);
        XSection.Y2 = std::min(aBox.Y2, bBox.Y2);
        XSection.Z1 = std::max(aBox.Z1, bBox.Z1);
        XSection.Z2 = std::min(aBox.Z2, bBox.Z2);


        if (XSection.X1 > XSection.X2 || 
          XSection.Y1 > XSection.Y2 || 
          XSection.Z1 > XSection.Z2) {
          adjacencyMatrix(j, i) = {0.0, 0.0, 0.0, 0.0, 0.0};
        } else {
          if(nodeA.color != voxelAColor || nodeA.s.rotation != voxelARot) {
            std::string name = FLAGS_voxelFolder + "R" 
              + std::to_string(nodeA.s.rotation) + "/" + pointVoxelFileNames[nodeA.color];
            place::loadInVoxel(name, aPoint);

            name = FLAGS_voxelFolder + "R" 
              + std::to_string(nodeA.s.rotation) + "/" + freeVoxelFileNames[nodeA.color];
            place::loadInVoxel(name, aFree);
            voxelAColor = nodeA.color;
            voxelARot = nodeA.s.rotation;
          }

          if(nodeB.color != voxelBcolor || nodeB.s.rotation != voxelBRot) {
            std::string name = FLAGS_voxelFolder + "R"
              + std::to_string(nodeB.s.rotation) + "/" + pointVoxelFileNames[nodeB.color];
            place::loadInVoxel(name, bPoint);
            name = FLAGS_voxelFolder + "R" 
              + std::to_string(nodeB.s.rotation) + "/" + freeVoxelFileNames[nodeB.color];
            place::loadInVoxel(name, bFree);
            voxelBcolor = nodeB.color;
            voxelBRot = nodeB.s.rotation;
          }


          place::cube crossWRTA, crossWRTB;

          crossWRTA.X1 = XSection.X1 - aBox.X1;
          crossWRTA.X2 = XSection.X2 - aBox.X1;
          crossWRTA.Y1 = XSection.Y1 - aBox.Y1;
          crossWRTA.Y2 = XSection.Y2 - aBox.Y1;
          crossWRTA.Z1 = XSection.Z1 - aBox.Z1;
          crossWRTA.Z2 = XSection.Z2 - aBox.Z1;

          crossWRTB.X1 = XSection.X1 - bBox.X1;
          crossWRTB.X2 = XSection.X2 - bBox.X1;
          crossWRTB.Y1 = XSection.Y1 - bBox.Y1;
          crossWRTB.Y2 = XSection.Y2 - bBox.Y1;
          crossWRTB.Z1 = XSection.Z1 - bBox.Z1;
          crossWRTB.Z2 = XSection.Z2 - bBox.Z1;


          place::edgeWeight weight = place::compare3D(aPoint, bPoint, aFree, 
            bFree, crossWRTA, crossWRTB);

          if(false) {
            std::cout << weight << std::endl;
            cvNamedWindow("aPoint", CV_WINDOW_NORMAL);
            displayVoxelGrid(aPoint.v, "aPoint");

            cvNamedWindow("bPoint", CV_WINDOW_NORMAL);
            displayVoxelGrid(bPoint.v, "bPoint");

            cvNamedWindow("aFree", CV_WINDOW_NORMAL);
            displayVoxelGrid(aFree.v, "aFree");

            cvNamedWindow("bFree", CV_WINDOW_NORMAL);
            displayVoxelGrid(bFree.v, "bFree");

            displayVoxelGrid(aFree.v, bPoint.v, crossWRTA, crossWRTB);

          }
          


          adjacencyMatrix(j, i) = weight; 

          //adjacencyMatrix(j, i) = {0, 0, 0, 0, 0};
        }
      }
    }
  }
  std::cout << std::endl;
  if(FLAGS_save)
    place::saveGraph(adjacencyMatrix);
}

void place::loadInPlacementGraph(const std::string & imageName, 
  std::vector<place::node> & nodes, const int num) {
  const std::string placementName = FLAGS_preDone +
    imageName.substr(imageName.find("_")-3, 3)
    + "_placement_" + imageName.substr(imageName.find(".")-3, 3) + ".dat";
  std::ifstream in(placementName, std::ios::in | std::ios::binary);

  int numToLoad;
  in.read(reinterpret_cast<char *>(&numToLoad), sizeof(numToLoad));
  numToLoad = (numToLoad > 5) ? 5 : numToLoad;
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
      uchar * scanDst = trimmedScan.ptr<uchar>(i - minRow);
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

void place::displayGraph(const Eigen::MatrixXS & adjacencyMatrix, 
  const std::vector<place::node> & nodes,
  const std::vector<std::vector<Eigen::MatrixXb> > & scans, 
  const std::vector<std::vector<Eigen::Vector2i> > & zeroZeros) {


  const int rows = adjacencyMatrix.rows();
  const int cols = adjacencyMatrix.cols();

  for(int i = 46; i < cols; ++i) {
    for(int j = 0; j < rows; ++j) {

      const place::node & nodeA = nodes[i];
      const place::node & nodeB = nodes[j];

      if(nodeA.color == nodeB.color)
        continue;
      /*if(i > j)
        continue;*/
      if(adjacencyMatrix(j,i).w == 0)
        continue;

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
          if(l + yOffset < 0 || l + yOffset >= output.rows)
            continue;
          if(k + xOffset < 0 || k + xOffset >= output.cols)
            continue;

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
          if(l + yOffset < 0 || l + yOffset >= output.rows)
            continue;
          if(k + xOffset < 0 || k + xOffset >= output.cols)
            continue;

          if(bScan(l, k) != 0) {
            _output(l + yOffset, k + xOffset)[0]=255;
            _output(l + yOffset, k + xOffset)[1]=0;
            _output(l + yOffset, k + xOffset)[2]=0;
          }
        }
      }

      cvNamedWindow("Preview", CV_WINDOW_NORMAL);
      cv::imshow("Preview", output);
      if(!FLAGS_quiteMode) {
        std::cout << "Color A: " << nodeA.color << "  Color B: " << nodeB.color << std::endl;
        std::cout << adjacencyMatrix(j,i) << std::endl;

      }
      cv::waitKey(0);
      ~output;
    }
  }
}

void place::displayBest(const std::vector<const place::node *> & bestNodes,
  const std::vector<std::vector<Eigen::MatrixXb> > & scans, 
  const std::vector<std::vector<Eigen::Vector2i> > & zeroZeros) {
 
  std::cout << "Displaying TRW solution" << std::endl;
 
  for(auto & n : bestNodes) {
    cv::Mat output(fpColor.rows, fpColor.cols, CV_8UC3);
    fpColor.copyTo(output);
    cv::Mat_<cv::Vec3b> _output = output;
    const Eigen::MatrixXb & scan = scans[n->color][n->s.rotation];
    const Eigen::Vector2i zeroZero = zeroZeros[n->color][n->s.rotation];
    const int xOffset = n->s.x - zeroZero[0];
    const int yOffset = n->s.y - zeroZero[1];
    
    for(int i = 0; i < scan.cols(); ++i) {
      for(int j = 0; j < scan.rows(); ++j) {
        if(scan(j,i) != 0) {
          if(j + yOffset < 0 || j + yOffset >= output.rows)
            continue;
          if(i + xOffset < 0 || i + xOffset >= output.cols)
            continue;

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

place::edgeWeight place::compare3D(const place::voxel & aPoint,
  const place::voxel & bPoint,
  const place::voxel & aFree,
  const place::voxel & bFree, 
  const place::cube & aRect, const place::cube & bRect) {


  const int z = aRect.Z2 - aRect.Z1 + 1;
  const int Xrows = aRect.Y2 - aRect.Y1 + 1;
  const int Xcols = aRect.X2 - aRect.X1 + 1;

  double pointAgreement = 0.0, freeSpaceAgreementA = 0.0, 
    freeSpaceAgreementB = 0.0, freeSpaceCross = 0.0;

  double totalPointA = 0.0, totalPointB = 0.0,
    averageFreeSpace = 0.0;

  for(int i = 0; i < z; ++i) {
    const Eigen::MatrixXb & Ap = aPoint.v[i + aRect.Z1];
    const Eigen::MatrixXb & Bp = bPoint.v[i + bRect.Z1];
    const Eigen::MatrixXb & Af = aFree.v[i + aRect.Z1];
    const Eigen::MatrixXb & Bf = bFree.v[i + bRect.Z1];

    if((Ap.sum() == 0 && Af.sum() == 0) || (Bp.sum() == 0 && Bf.sum() == 0))
      continue;

    for(int k = 0; k < Xcols; ++k) {
      for(int l = 0; l < Xrows; ++l) {
        if((localGroup(Ap, l + aRect.Y1, k + aRect.X1) && 
          Bp(l + bRect.Y1, k + bRect.X1)) || (Ap(l + aRect.Y1, k + aRect.X1) && 
          localGroup(Bp, l + bRect.Y1, k + bRect.X1)))
          ++pointAgreement /*+= Ap(l + aRect.Y1, k + aRect.X1) + Bp(l + bRect.Y1, k + bRect.X1)*/;

        if(Ap(l + aRect.Y1, k + aRect.X1) && 
          Bf(l + bRect.Y1, k + bRect.X1))
          ++freeSpaceAgreementA/* += Bf(l + bRect.Y1, k + bRect.X1)*/;

        if(Bp(l + bRect.Y1, k + bRect.X1) &&
          Af(l + aRect.Y1, k + aRect.X1))
            ++freeSpaceAgreementB /*+= Af(l + aRect.Y1, k + aRect.X1)*/;

        if(Bf(l + bRect.Y1, k + bRect.X1) &&
          Af(l + aRect.Y1, k + aRect.X1))
          ++freeSpaceCross/* += Bf(l + bRect.Y1, k + bRect.X1) + Af(l + aRect.Y1, k + aRect.X1)*/;

        if(Bf(l + bRect.Y1, k + bRect.X1))
          ++averageFreeSpace;
        if(Af(l + aRect.Y1, k + aRect.X1))
          ++averageFreeSpace;

        if(Ap(l + aRect.Y1, k + aRect.X1))
          ++totalPointA;

        if(Bp(l + bRect.Y1, k + bRect.X1))
          ++totalPointB;
      }
    }
  }
  // averageFreeSpace /= 2.0;
  double averagePoint = (totalPointA + totalPointB)/2.0;

  if(averageFreeSpace == 0.0 || averagePoint == 0.0 || 
    totalPointA == 0.0 || totalPointB == 0.0)
    return {0, 0, 0, 0, 0};

  /*double weight = 2.0/(averagePoint/pointAgreement + averageFreeSpace/freeSpaceCross) -
     (freeSpaceAgreementB/totalPointB + freeSpaceAgreementA/totalPointA)/2.0;*/

  double weight = (pointAgreement/averagePoint + freeSpaceCross/averageFreeSpace)/2.0
    - (freeSpaceAgreementB/totalPointB + freeSpaceAgreementA/totalPointA)/2.0;


  return {pointAgreement/averagePoint, freeSpaceAgreementA/totalPointA, 
    freeSpaceAgreementB/totalPointB, freeSpaceCross/averageFreeSpace, weight};
}

void place::loadInVoxel(const std::string & name, 
  place::voxel & dst) {
  int x,y,z;
  std::ifstream in (name, std::ios::in | std::ios::binary);
  in.read(reinterpret_cast<char *>(&z), sizeof(z));
  in.read(reinterpret_cast<char *>(&y), sizeof(y));
  in.read(reinterpret_cast<char *>(&x), sizeof(x));

  dst.v.assign(z, Eigen::MatrixXb(y,x));

  for(int i = 0; i < z; ++i)
    in.read(dst.v[i].data(), dst.v[i].size());
  
  in.read(reinterpret_cast<char *>(&dst.c), sizeof(size_t));

  in.close();
}

void place::TRWSolver(const Eigen::MatrixXS & adjacencyMatrix,
  const std::vector<place::node> & nodes, 
  std::vector<const place::node * > & bestNodes) {

  
  typedef opengm::DiscreteSpace<> Space;
  typedef opengm::ExplicitFunction<double> Function;

  //Construct the Model from the Adjacency Matrix
  typedef opengm::GraphicalModel<
    double,
    opengm::Adder
  > Model;

  //Figure out how many different variables and labels there are
  std::vector<size_t> numberOfLabels;
  {
    size_t i = 0;
    const place::node * prevNode = &nodes[0];
    for(auto & n : nodes) {
      if(n.color == prevNode->color) {
        prevNode = &n;
        ++i;
      } else {
        numberOfLabels.push_back(i);
        i = 1;
        prevNode = &n;
      }
    }
    numberOfLabels.push_back(i);
  }
  const int numVars = numberOfLabels.size();

  //Construct the model
  Model gm (Space (numberOfLabels.begin(), numberOfLabels.end()));


  //Add urnary terms
  for(size_t i = 0, offset = 0; i < numVars; ++i) {
    const size_t shape [] = {numberOfLabels[i]};
    Function f(shape, shape + 1);
    for(int j = 0; j < numberOfLabels[i]; ++j) {
      const place::posInfo & currentScore = nodes[offset + j].s;
      double scanExplained =
        (currentScore.scanPixels - currentScore.scanFP)/(currentScore.scanPixels);
      double fpExplained = 
      (currentScore.fpPixels - currentScore.fpScan)/(currentScore.fpPixels);

      f(j) = (scanExplained + fpExplained)/2.0;
    }
    Model::FunctionIdentifier fid = gm.addFunction(f);
    const size_t factors [] = {i};
    gm.addFactor(fid, factors, factors + 1);

    offset += numberOfLabels[i];
  }

  //Add pairwise terms
  for(size_t i = 0, colOffset = 0, rowOffset = 0; i < numVars; ++i) {
    rowOffset += numberOfLabels[i];
    int rowOcp = rowOffset;
    for(size_t j = i + 1; j < numVars; ++j) {
      Eigen::MatrixXS currentMat = adjacencyMatrix.block(rowOcp, colOffset,
        numberOfLabels[j], numberOfLabels[i]);

      const size_t shape [] = {numberOfLabels[i], numberOfLabels[j]};
      Function f(shape, shape + 2);
      for(int a = 0; a < currentMat.cols(); ++a) {
        for(int b = 0; b < currentMat.rows(); ++b) {
          f(a,b) = currentMat(b,a).w;
        }
      }
      Model::FunctionIdentifier fid = gm.addFunction(f);
      const size_t factors [] = {i,j};
      gm.addFactor(fid, factors, factors + 2);

      rowOcp += numberOfLabels[j];
    }
    colOffset += numberOfLabels[i];
  }

  // set up the optimizer (TRW)
  typedef opengm::TRWSi_Parameter<Model> Parameter;
  typedef opengm::TRWSi<Model, opengm::Maximizer> Solver;
  
  Parameter parameter (30);
  Solver solver (gm, parameter);
  Solver::VerboseVisitorType verboseVisitor;
  solver.infer(verboseVisitor);

  std::vector<Model::LabelType> labeling (numVars);
  solver.arg(labeling);

  std::cout << "Labels : ";
  for(auto & l : labeling)
    std::cout << l << "_";
  std::cout << std::endl;

  bestNodes.reserve(numVars);
  for(int i = 0, offset = 0; i < numVars; ++i) {
    if(labeling[i] < numberOfLabels[i])
      bestNodes.push_back(&nodes[offset + labeling[i]]);
    offset += numberOfLabels[i];
  }
}

static void condenseStack(std::vector<GRBVar> & stacked,
  GRBModel & model) {
  if(stacked.size() == 2) {
    GRBVar first = stacked.back();
    stacked.pop_back();
    GRBVar second = stacked.back();
    stacked.pop_back();

    GRBVar newStack = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
    model.update();
    model.addQConstr(first * second,
      GRB_EQUAL, newStack);
    stacked.push_back(newStack);

  } else if (stacked.size() == 1) return;
  else {
    std::vector<GRBVar> firstHalf (stacked.begin(), 
      stacked.begin() + stacked.size()/2);
    std::vector<GRBVar> secondHalf(stacked.begin() + stacked.size()/2,
      stacked.end());

    condenseStack(firstHalf, model);
    condenseStack(secondHalf, model);
    stacked.clear();
    stacked.insert(stacked.end(), firstHalf.begin(), firstHalf.end());
    stacked.insert(stacked.end(), secondHalf.begin(), secondHalf.end());
  }
}

static void stackTerms(const std::vector<int> & toStack,
  const GRBVar * varList, GRBModel & model,
  std::map<std::pair<int,int>, GRBVar> & preStacked,
  std::vector<GRBVar> & stacked) {
  int i = 0;
  for(; i < toStack.size() - 1; i+=2) {
    std::pair<int, int> key (toStack[i], toStack[i+1]);
    auto it = preStacked.find(key);
    if(it == preStacked.end()) {
      GRBVar newStack = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
      model.update();
      model.addQConstr(varList[toStack[i]] * varList[toStack[i+1]],
        GRB_EQUAL, newStack);
      preStacked.emplace(key, newStack);
      stacked.push_back(newStack);
    } else {
      stacked.push_back(it->second);
    }
  }
  for(; i < toStack.size(); ++i) {
    if(stacked.size() > 1) {
      GRBVar newStack = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
      model.update();
      model.addQConstr(varList[toStack[i]] * stacked.back(),
        GRB_EQUAL, newStack);
      stacked.pop_back();
      stacked.push_back(newStack);
    } else {
      stacked.push_back(varList[toStack[i]]);
    }
  }
  while(stacked.size() > 2)
    condenseStack(stacked, model);
}

void place::MIPSolver(const Eigen::MatrixXS & adjacencyMatrix, 
  const std::map<std::vector<int>, double> & highOrder, const std::vector<place::node> & nodes,
  std::vector<const place::node *> & bestNodes) {

  std::vector<int> numberOfLabels;
  {
    int i = 0;
    const place::node * prevNode = &nodes[0];
    for(auto & n : nodes) {
      if(n.color == prevNode->color) {
        prevNode = &n;
        ++i;
      } else {
        numberOfLabels.push_back(i);
        i = 1;
        prevNode = &n;
      }
    }
    numberOfLabels.push_back(i);
  }
  const int numVars = numberOfLabels.size();
  const int numOpts = nodes.size();
  try {
    GRBEnv env = GRBEnv();
    env.set("TimeLimit", "300");

    GRBModel model = GRBModel(env);

    double * upperBound = new double [numOpts];
    char * type = new char [numOpts];
    for(int i = 0; i < numOpts; ++i) {
      upperBound[i] = 1.0;
      type[i] = GRB_BINARY;
    }
    
    GRBVar * varList = model.addVars(NULL, upperBound, NULL, type, NULL, numOpts);
    delete [] upperBound;
    delete [] type;
    // Integrate new variables

    model.update();

    
    GRBQuadExpr objective = 0.0;
    for(int i = 0; i < numOpts; ++i) {
      for(int j = i + 1; j < numOpts; ++j) {
        if(adjacencyMatrix(j,i).w == 0.0)
          continue;

        objective += adjacencyMatrix(j,i).w*varList[i]*varList[j];
      }
      const place::posInfo & currentScore = nodes[i].s;
      double scanExplained =
        (currentScore.scanPixels - currentScore.scanFP)/(currentScore.scanPixels);
      double fpExplained = 
      (currentScore.fpPixels - currentScore.fpScan)/(currentScore.fpPixels);

      objective += varList[i]*(fpExplained + scanExplained)/2.0;
    }

    for(int i = 0, offset = 0; i < numVars; ++i) {
      GRBLinExpr constr = 0.0;
      double * coeff = new double [numberOfLabels[i]];
      for(int a = 0; a < numberOfLabels[i]; ++ a)
        coeff[a] = 1.0;

      constr.addTerms(coeff, varList + offset, numberOfLabels[i]);
      model.addConstr(constr, GRB_EQUAL, 1.0);
      offset += numberOfLabels[i];
      delete [] coeff;
    }

    std::map<std::pair<int, int>, GRBVar> termCondense;
    for(auto & it : highOrder) {
      auto & incident = it.first;
      if(incident.size() == 2) {
        objective += varList[incident[0]]*varList[incident[1]]*it.second;
      } else if(incident.size() == 1) {
        objective += varList[incident[0]]*it.second;
      }else {
        std::vector<GRBVar> final;
        stackTerms(incident, varList, model, termCondense, final);
        objective += final[0]*final[1]*it.second;
      }
    }

    model.setObjective(objective, GRB_MAXIMIZE);
    model.optimize();

    for(int i = 0; i < numOpts; ++i) {
      if(varList[i].get(GRB_DoubleAttr_X) == 1.0) {
        bestNodes.push_back(&(nodes[i]));
        std::cout << i << "_";
      }
    }
    std::cout << std::endl;
  } catch(GRBException e) {
    std::cout << "Error code = " << e.getErrorCode() << std::endl;
    std::cout << e.getMessage() << std::endl;
  } catch(...) {
    std::cout << "Exception during optimization" << std::endl;
  }
}

bool place::reloadGraph(Eigen::MatrixXS & adjacencyMatrix) {
  const std::string graphName = FLAGS_preDoneV2 + "graph.dat";
  std::ifstream in (graphName, std::ios::in | std::ios::binary);

  if(!in.is_open())
    return false;

  int cols, rows;
  in.read(reinterpret_cast<char *>(&rows), sizeof(rows));
  in.read(reinterpret_cast<char *>(&cols), sizeof(cols));

  adjacencyMatrix = Eigen::MatrixXS(rows, cols);
  in.read(reinterpret_cast<char *>(adjacencyMatrix.data()),
    sizeof(place::edgeWeight)*adjacencyMatrix.size());

  in.close();

  return true;
}

void place::saveGraph(Eigen::MatrixXS & adjacencyMatrix) {
  const std::string graphName = FLAGS_preDoneV2 + "graph.dat";
  std::ofstream out (graphName, std::ios::out | std::ios::binary);

  if(!FLAGS_save)
    return;

  int rows = adjacencyMatrix.rows();
  int cols = adjacencyMatrix.cols();

  out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
  out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));

  out.write(reinterpret_cast<const char *>(adjacencyMatrix.data()),
    sizeof(place::edgeWeight)*adjacencyMatrix.size());

  out.close();
}

bool place::localGroup(const Eigen::MatrixXb & toCheck, const int yOffset, 
  const int xOffset) {
  for(int i = -2; i < 2; ++i) {
    for(int j = -2; j < 2; ++j) {
      if(yOffset + j < 0 || yOffset + j >= toCheck.rows())
        continue;
      if(xOffset + i < 0 || xOffset + i >= toCheck.cols())
        continue;
      if(toCheck(yOffset + j, xOffset + i))
        return true;
    }
  }

  return false;
}

static double harmonic(int stop, double r) {
  double val = 0.0;
  for(int i = 1; i <= stop; ++i) {
    double v = std::pow(static_cast<double>(i), r);
    val += 0.1/v;
  }
  return val;
}

void place::createHigherOrderTerms(const std::vector<std::vector<Eigen::MatrixXb> > & scans,
  const std::vector<std::vector<Eigen::Vector2i> > & zeroZeros,
  const std::vector<place::node> & nodes, std::map<std::vector<int>, double> &
    highOrder) {
  Eigen::ArrayXH hMap (floorPlan.rows, floorPlan.cols);
  for(int a = 0; a < nodes.size(); ++a) {
    const place::node & currentNode = nodes[a];
    const Eigen::MatrixXb & currentScan = scans[currentNode.color][currentNode.s.rotation];
    const Eigen::Vector2i & zeroZero = zeroZeros[currentNode.color][currentNode.s.rotation];
    const int xOffset = currentNode.s.x - zeroZero[0], 
      yOffset = currentNode.s.y - zeroZero[1];

    for(int j = 0; j < currentScan.rows(); ++j) {
      const uchar * src = floorPlan.ptr(j + yOffset);
      for(int i = 0; i < currentScan.cols(); ++i) {
        if(src[i + xOffset] != 255) {
          if(localGroup(currentScan, j, i)) {
            const place::posInfo & currentScore = currentNode.s;
            double scanExplained =
              (currentScore.scanPixels - currentScore.scanFP)/(currentScore.scanPixels);
            double fpExplained = 
            (currentScore.fpPixels - currentScore.fpScan)/(currentScore.fpPixels);


            hMap(j+yOffset, i + xOffset).incident.push_back(a);
            const double weight = (scanExplained + fpExplained)/2.0;
            hMap(j+yOffset, i + xOffset).weight += weight;
          }
        }
      }
    }
  }

  cv::Mat out (floorPlan.rows, floorPlan.cols, CV_8UC3);
  fpColor.copyTo(out);
  cv::Mat_<cv::Vec3b> _out = out;
  for(int i = 0; i < hMap.cols(); ++i) {
    for(int j = 0; j < hMap.rows(); ++j) {
      if(hMap(j,i).incident.size() != 0) {
        const int gray = 
          cv::saturate_cast<uchar>(255*hMap(j,i).incident.size()/5.0);
         int red, green, blue;
        if (gray < 128) {
          red = 0;
          blue = 2 * gray;
          green = 255 - blue;
        } else {
          blue = 0;
          red = 2 * (gray - 128);
          green = 255 - red;
        }
        _out(j,i)[0] = blue;
        _out(j,i)[1] = green;
        _out(j,i)[2] = red;
      }
    }
  }

  cvNamedWindow("Preview", CV_WINDOW_NORMAL);
  cv::imshow("Preview", out);
  cv::waitKey(0);

  place::hOrder * data = hMap.data();
  for(int i = 0; i < hMap.size(); ++i) {
    if((data + i)->incident.size() != 0) {
      const double scale = harmonic((data + i)->incident.size(), 1.5);
      (data + i)->weight /= (data + i)->incident.size();
      (data + i)->weight *= scale;
    }
  }

  for(int i = 0; i < hMap.size(); ++i) {
    std::vector<int> & key = (data + i)->incident;
    if(key.size() != 0 && (data + i)->weight > 0.0) {
      auto it = highOrder.find(key);
      if(it != highOrder.end())
        it->second += (data + i)->weight;
      else 
        highOrder.emplace(key, (data + i)->weight);
    }
  }

  double average = 0.0;
  for(auto & it : highOrder)
    average += it.second;
  average /= highOrder.size();
  double sigma = 0.0;
  for(auto & it : highOrder)
    sigma += (it.second - average)*(it.second -average);
  sigma /= (highOrder.size() - 1);
  sigma = sqrt(sigma);
  for(auto & it : highOrder)
    it.second = std::max(0.0,((it.second - average)/(sigma) + 2.0)/100.0);
}

/*void place::createWeightedFloorPlan (Eigen::SparseMatrix<double> & weightedFloorPlan) {
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
}*/