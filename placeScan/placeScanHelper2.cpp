#include "placeScan_placeScanHelper2.h"
#include "placeScan_panoramaMatcher.h"

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
#include "gurobi_c++.h"

const int minScans = 2;
const int maxScans = 20;

static void displayCollapsed(auto & collapsed, const std::string & windowName) {
  double average, sigma;
  average = sigma = 0;
  int count = 0;
  const double * dataPtr = collapsed.data();
  for (int i = 0; i < collapsed.size(); ++i) {
    if (*(dataPtr+ i)) {
      ++count;
      average+= *(dataPtr + i);
    }
  }

  average = average/count;

  for (int i = 0; i < collapsed.size(); ++i) {
    if (*(dataPtr + i) !=0)
      sigma += (*(dataPtr + i) - average)*(*(dataPtr + i)- average);
  }

  sigma = sigma/(count-1);
  sigma = sqrt(sigma);

  cv::Mat heatMap (collapsed.rows(), collapsed.cols(), CV_8UC3, cv::Scalar::all(255));
  for (int i = 0; i < heatMap.rows; ++i) {
    uchar * dst = heatMap.ptr<uchar>(i);
    for (int j = 0; j < heatMap.cols; ++j) {
      if (collapsed(i, j)){
        const int gray = cv::saturate_cast<uchar>(
          255.0 * (collapsed(i, j) - average)
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
  cvNamedWindow(windowName.data(), CV_WINDOW_NORMAL);
  cv::imshow(windowName, heatMap);
}

static void displayVoxelGrid(const auto & grid,
  const std::string & windowName) {
  Eigen::MatrixXd collapsed = Eigen::MatrixXd::Zero (grid[0].rows(), grid[0].cols());

  for (int k = 0; k < grid.size(); ++k)
    for (int i = 0; i < grid[0].cols(); ++i)
      for (int j = 0; j < grid[0].rows(); ++j)
        collapsed(j, i) += grid[k](j,i) ? 1 : 0;

  displayCollapsed(collapsed, windowName);
}

template<typename T>
static void displayVoxelGridS(const T & sparse,
  const std::string & windowName) {
  typedef typename T::value_type::InnerIterator InnerIT;
  Eigen::MatrixXd collapsed = Eigen::MatrixXd::Zero(sparse[0].rows(), sparse[0].cols());
  for (int k = 0; k < sparse.size(); ++k) {
    auto & current = sparse[k];
    for (int i = 0; i < current.outerSize(); ++i) {
      for (InnerIT it (current, i); it; ++it) {
        collapsed(it.row(), it.col()) += it.value();
      }
    }
  }
  displayCollapsed(collapsed, windowName);
}

static void displayCollapsed(const auto & collapsedA, const auto & collapsedB,
  const place::cube & aRect, const place::cube & bRect) {

  const int z = aRect.Z2 - aRect.Z1 + 1;
  const int Xrows = aRect.Y2 - aRect.Y1 + 1;
  const int Xcols = aRect.X2 - aRect.X1 + 1;

  double averageA = 0, sigmaA = 0, averageB = 0, sigmaB = 0;
  int countA = 0, countB = 0;
  const double * dataPtrA = collapsedA.data();
  const double * dataPtrB = collapsedB.data();
  for (int i = 0; i < collapsedA.size(); ++i) {
    if (*(dataPtrA+ i)) {
      ++countA;
      averageA += *(dataPtrA + i);
    }
    if (*(dataPtrB+ i)) {
      ++countB;
      averageB += *(dataPtrB + i);
    }
  }

  averageA = averageA/countA;
  averageB = averageB/countB;

  for (int i = 0; i < collapsedA.size(); ++i) {
    if (*(dataPtrA + i) !=0)
      sigmaA += (*(dataPtrA + i) - averageA)*(*(dataPtrA + i)- averageA);

    if (*(dataPtrB + i) !=0)
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

      double aVal = collapsedA(i, j);
      double bVal = collapsedB(i, j);

      if (aVal && bVal) {
        const int grayA = cv::saturate_cast<uchar>(
          255.0 * (aVal - averageA)
            / (1.0 * sigmaA));
        const int grayB = cv::saturate_cast<uchar>(
          255.0 * (bVal - averageB)
            / (1.0 * sigmaB));
        int red = 255, green = 0, blue = 255;
        dst[j*3] = blue;
        dst[j*3 +1] = green;
        dst[j*3 + 2] = red;
      } else if (aVal){
        const int gray = cv::saturate_cast<uchar>(
          255.0 * (aVal - averageA)
            / (1.0 * sigmaA));
        int red = 255, green = 0, blue = 0;
        dst[j*3] = blue;
        dst[j*3 +1] = green;
        dst[j*3 + 2] = red;
      } else if (bVal){
        const int gray = cv::saturate_cast<uchar>(
          255.0 * (bVal - averageB)
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

static void displayVoxelGrid(const auto & voxelA, const auto & voxelB,
  const place::cube & aRect, const place::cube & bRect) {

  const int z = aRect.Z2 - aRect.Z1 + 1;
  const int Xrows = aRect.Y2 - aRect.Y1 + 1;
  const int Xcols = aRect.X2 - aRect.X1 + 1;

  Eigen::MatrixXd collapsedA = Eigen::MatrixXd::Zero(Xrows, Xcols);
  Eigen::MatrixXd collapsedB = Eigen::MatrixXd::Zero(Xrows, Xcols);
  for (int k = 0; k < z; ++k) {
    for (int i = 0; i < Xcols; ++i) {
      for (int j = 0; j < Xrows; ++j) {
        Eigen::Vector3i APos (i + aRect.X1, j + aRect.Y1, z + aRect.Z1);
        Eigen::Vector3i BPos (i + bRect.X1, j + bRect.Y1, z + bRect.Z1);
        collapsedA(j,i) += voxelA[APos[2]](APos[1], APos[0]);
        collapsedB(j,i) += voxelB[BPos[2]](BPos[1], BPos[0]);
      }
    }
  }
  displayCollapsed(collapsedA, collapsedB, aRect, bRect);
}

template<typename TA, typename TB>
static void displayVoxelGridS(const TA & voxelA, const TB & voxelB,
  const place::cube & aRect, const place::cube & bRect) {

  typedef typename TA::value_type::InnerIterator InnerITA;
  typedef typename TB::value_type::InnerIterator InnerITB;

  const int z = aRect.Z2 - aRect.Z1 + 1;
  const int Xrows = aRect.Y2 - aRect.Y1 + 1;
  const int Xcols = aRect.X2 - aRect.X1 + 1;

  Eigen::MatrixXd collapsedA = Eigen::MatrixXd::Zero(Xrows, Xcols);
  Eigen::MatrixXd collapsedB = Eigen::MatrixXd::Zero(Xrows, Xcols);

  for (int k = 0; k < z; ++k) {
    auto & currentA = voxelA[k + aRect.Z1];
    auto & currentB = voxelB[k + bRect.Z1];

    for (int i = 0; i < currentA.outerSize(); ++i) {
      for (InnerITA it (currentA, i); it; ++it) {
        Eigen::Vector3i APos (it.col() - aRect.X1, it.row() - aRect.Y1, 0);
        if (APos[0] < 0 || APos[0] >= Xcols)
          continue;
        if (APos[1] < 0 || APos[1] >= Xrows)
          continue;
        collapsedA(APos[1], APos[0]) += it.value();
      }
    }

    for (int i = 0; i < currentB.outerSize(); ++i) {
      for (InnerITB it (currentB, i); it; ++it) {
        Eigen::Vector3i BPos (it.col() - bRect.X1, it.row() - bRect.Y1, 0);
        if (BPos[0] < 0 || BPos[0] >= Xcols)
          continue;
        if (BPos[1] < 0 || BPos[1] >= Xrows)
          continue;
        collapsedB(BPos[1], BPos[0]) += it.value();
      }
    }
  }
  displayCollapsed(collapsedA, collapsedB, aRect, bRect);

}

void place::loadInScansGraph(const std::vector<std::string> & pointFileNames,
  const std::vector<std::string> & freeFileNames,
  const std::vector<std::string> & zerosFileNames,
  std::vector<std::vector<Eigen::MatrixXb> > & scans,
  std::vector<std::vector<Eigen::MatrixXb> > & masks,
  std::vector<std::vector<Eigen::Vector2i> > & zeroZeros) {

  for (int i = 0; i < zeroZeros.size(); ++i) {

    const std::string scanName = pointFileNames[i];
    const std::string zerosFile = FLAGS_zerosFolder + zerosFileNames[i];
    const std::string maskName = freeFileNames[i];

    std::vector<cv::Mat> toTrimScans, toTrimMasks,
      trimmedScans, trimmedMasks, toTrimMasksD;
    place::loadInScansAndMasks(scanName, zerosFile,
    maskName, toTrimScans, toTrimMasks, zeroZeros[i]);

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
    for (auto & src : toTrimMasks) {
      cv::Mat dst;
      cv::dilate(src, dst, element);
      toTrimMasksD.push_back(dst);
    }

    place::trimScansAndMasks(toTrimScans, toTrimMasksD,
      trimmedScans, trimmedMasks, zeroZeros[i]);


    std::vector<Eigen::MatrixXb> tmpvec;
    for (auto & scan : trimmedScans) {
      Eigen::MatrixXb tmp = Eigen::MatrixXb::Zero(scan.rows, scan.cols);
      for (int j = 0; j < scan.rows; ++j) {
        const uchar * src = scan.ptr<uchar>(j);
        for (int k = 0; k < scan.cols; ++k) {
          if (src[k] != 255) {
            const double confidence = 1.0 - src[k]/255.0;
              if (confidence > 0.75)
                tmp(j,k) = static_cast<char>(1);
          }
        }
      }
      tmpvec.push_back(tmp);
    }
    scans.push_back(tmpvec);
    tmpvec.clear();

    for (auto & mask : trimmedMasks) {
      Eigen::MatrixXb tmp = Eigen::MatrixXb::Zero(mask.rows, mask.cols);
      for (int j = 0; j < mask.rows; ++j) {
        const uchar * src = mask.ptr<uchar>(j);
        for (int k = 0; k < mask.cols; ++k) {
          if (src[k] != 255)
            tmp(j,k) = static_cast<char>(1);
        }
      }
      tmpvec.push_back(tmp);
    }
    masks.push_back(tmpvec);
  }
}

static std::string encapPair(int x, int y) {
  return "(" + std::to_string(x) + "," + std::to_string(y) + ")";
}

static bool orderPairs(int x1, int y1, int x2, int y2) {
  return x1 < x2 || (x1 == x2 && y1 < y2);
}

void place::weightEdges(const std::vector<place::node> & nodes,
  const std::vector<std::vector<place::metaData> > & voxelInfo,
  const std::vector<std::string> & pointVoxelFileNames,
  const std::vector<std::string> & freeVoxelFileNames,
  const std::vector<std::vector<Eigen::Matrix3d> > & rotationMatricies,
  std::vector<place::Panorama> & panoramas,
  Eigen::MatrixXE & adjacencyMatrix) {

  typedef struct {
    int i, j;
    place::cube crossWRTA, crossWRTB;
  } later;

  if (!FLAGS_redo && place::reloadGraph(adjacencyMatrix))
    return;

  const int rows = adjacencyMatrix.rows();
  const int cols = adjacencyMatrix.cols();
  std::vector<later> tracker;
  //Iterator over the lower triangle of the adjaceny matrix
  for (int i = 0; i < cols ; ++i) {
    const place::node & nodeA = nodes[i];
    for (int j = i + 1; j < rows; ++j) {
      const place::node & nodeB = nodes[j];
      if (nodeA.color == nodeB.color) {
        continue;
      }

      auto & metaA = voxelInfo[nodeA.color][nodeA.s.rotation];
      auto & metaB = voxelInfo[nodeB.color][nodeB.s.rotation];

      place::cube aBox, bBox;
      aBox.X1 = nodeA.s.x*(metaA.vox/ metaA.s) - metaA.zZ[0];
      aBox.X2 = aBox.X1 + metaA.x - 1;
      aBox.Y1 = nodeA.s.y*(metaA.vox/ metaA.s) - metaA.zZ[1];
      aBox.Y2 = aBox.Y1 + metaA.y - 1;
      aBox.Z1 = 0 - metaA.zZ[2];
      aBox.Z2 = aBox.Z1 + metaA.z - 1;

      bBox.X1 = nodeB.s.x*(metaB.vox/ metaB.s) - metaB.zZ[0];
      bBox.X2 = bBox.X1 + metaB.x - 1;
      bBox.Y1 = nodeB.s.y*(metaB.vox/ metaB.s) - metaB.zZ[1];
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

      if (!(XSection.X1 > XSection.X2 ||
        XSection.Y1 > XSection.Y2 ||
        XSection.Z1 > XSection.Z2)) {

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

        tracker.push_back({i, j, crossWRTA, crossWRTB});
      }
    }
  }

  std::sort(tracker.begin(), tracker.end(),
    [&nodes](auto & a, auto & b) {
      int aAColor = nodes[a.i].color;
      int aARot = nodes[a.i].s.rotation;
      int aBColor = nodes[a.j].color;
      int aBRot = nodes[a.j].s.rotation;

      int bAColor = nodes[b.i].color;
      int bARot = nodes[b.i].s.rotation;
      int bBColor = nodes[b.j].color;
      int bBRot = nodes[b.j].s.rotation;

      return orderPairs(aAColor, aARot, bAColor, bARot)
        || (aAColor == bAColor && aARot == bARot &&
          orderPairs(aBColor, aBRot, bBColor, bBRot));
    });

  std::cout << tracker.size() << std::endl;

  omp_set_nested(1);
  #pragma omp target teams num_teams(1)\
   shared(tracker, adjacencyMatrix, nodes)
  {
    int voxelAColor = -1, voxelARot = -1;
    int voxelBcolor = -1, voxelBRot = -1;
    place::voxelGrid aPoint, bPoint;
    place::voxelGrid aFree, bFree;
    #pragma omp distribute
    for (int k = 0; k < tracker.size(); ++k) {
      const later & current = tracker[k];
      const int i = current.i;
      const int j = current.j;
      const place::node & nodeA = nodes[i];
      const place::node & nodeB = nodes[j];

     //  if (j != 228 /*224*/) continue;
     //  if (i != 226) continue;

      if (nodeA.color != voxelAColor || nodeA.s.rotation != voxelARot) {
        std::string name = FLAGS_voxelFolder + "R"
          + std::to_string(nodeA.s.rotation) + "/" + pointVoxelFileNames[nodeA.color];
        place::loadInVoxel(name, aPoint);

        name = FLAGS_voxelFolder + "R"
          + std::to_string(nodeA.s.rotation) + "/" + freeVoxelFileNames[nodeA.color];
        place::loadInVoxel(name, aFree);
        voxelAColor = nodeA.color;
        voxelARot = nodeA.s.rotation;
      }

      if (nodeB.color != voxelBcolor || nodeB.s.rotation != voxelBRot) {
        std::string name = FLAGS_voxelFolder + "R"
          + std::to_string(nodeB.s.rotation) + "/" + pointVoxelFileNames[nodeB.color];
        place::loadInVoxel(name, bPoint);

        name = FLAGS_voxelFolder + "R"
          + std::to_string(nodeB.s.rotation) + "/" + freeVoxelFileNames[nodeB.color];
        place::loadInVoxel(name, bFree);
        voxelBcolor = nodeB.color;
        voxelBRot = nodeB.s.rotation;
      }

      place::edge weight = place::compare3D(aPoint, bPoint, aFree,
        bFree, current.crossWRTA, current.crossWRTB);

      auto & RA = rotationMatricies[nodeA.color][nodeA.s.rotation];
      auto & RB = rotationMatricies[nodeB.color][nodeB.s.rotation];

      auto & panoA = panoramas[nodeA.color];
      auto & panoB = panoramas[nodeB.color];

      auto & metaA = voxelInfo[nodeA.color][nodeA.s.rotation];
      auto & metaB = voxelInfo[nodeB.color][nodeB.s.rotation];

      Eigen::Vector3d AZeroZero (nodeA.s.x, nodeA.s.y, 0),
        BZeroZero (nodeB.s.x, nodeB.s.y, 0);
      AZeroZero /= metaA.s;
      BZeroZero /= metaB.s;
      const Eigen::Vector3d aToB = AZeroZero - BZeroZero;
      const Eigen::Vector3d bToA = BZeroZero - AZeroZero;

      if (false) {
        displayVoxelGridS(aPoint.v, "aPoint");
        displayVoxelGridS(bPoint.v, "bPoint");
        displayVoxelGrid(aFree.v, "aFree");
        displayVoxelGrid(bFree.v,  "bFree");
        displayVoxelGridS(aPoint.v, bPoint.v,
          current.crossWRTA, current.crossWRTB);
      }
      weight.panoW = pano::compareNCC2(panoA,
        panoB, RA, RB, aToB, bToA);
      adjacencyMatrix(j, i) = weight;
    }
  }
  //Copy the lower tranalge into the upper triangle
  for (int i = 0; i < cols ; ++i)
    for (int j = i + 1; j < rows; ++j)
      adjacencyMatrix(i, j) = adjacencyMatrix(j, i);

  if (FLAGS_save)
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
  place::posInfo tmp;
  in.read(reinterpret_cast<char *>(&tmp), sizeof(tmp));
  in.seekg(sizeof(numToLoad));

  std::vector<place::posInfo> scoretmp;
  double lastScore = 1.0;
  const double initailScore = tmp.score;
  bool deltaExceeded = false;
  int final = 0;
  for (int i = 0; i < numToLoad; ++i) {
    in.read(reinterpret_cast<char *>(&tmp), sizeof(tmp));
    scoretmp.push_back(tmp);

    if (tmp.score - lastScore > maxDelta) deltaExceeded = true;
    if (tmp.score - initailScore > maxTotal) deltaExceeded = true;
    if (!final && i + 1 >= minScans && deltaExceeded)
      final = i;

    lastScore = tmp.score;
  }
  if(!final) final = numToLoad - 1;
  if (final > maxScans - 1) final = maxScans - 1;
  std::vector<place::node> nodestmp;
  for (auto & s : scoretmp)
    nodestmp.push_back({s, 0.0, 0.0, num});

  for (auto & n : nodestmp) {
    const place::posInfo & currentScore = n.s;
    double scanExplained =
      (currentScore.scanPixels - currentScore.scanFP)/(currentScore.scanPixels);
    double fpExplained =
    (currentScore.fpPixels - currentScore.fpScan)/(currentScore.fpPixels);
    const double w = (scanExplained + fpExplained)/2.0;
    n.w = w;
  }

  double average = 0.0;
  for (auto & n : nodestmp)
    average += n.w;
  average /= nodestmp.size();

  double sigma = 0.0;
  for (auto & n : nodestmp)
    sigma += (n.w - average)*(n.w - average);

  sigma /= nodestmp.size() - 1;
  sigma = sqrt(sigma);

  for (auto & n : nodestmp)
    n.nw = (n.w - average)/sigma;

  for (auto & n : nodestmp)
    n.w = n.nw;

  nodes.insert(nodes.end(), nodestmp.begin(),
    nodestmp.begin() + (final + 1));
}

void place::trimScansAndMasks(const std::vector<cv::Mat> & toTrimScans,
  const std::vector<cv::Mat> & toTrimMasks, std::vector<cv::Mat> & trimmedScans,
  std::vector<cv::Mat> & trimmedMasks, std::vector<Eigen::Vector2i> & zeroZero) {
  for (int i = 0; i < toTrimScans.size(); ++i) {
    const cv::Mat & currentScan = toTrimScans[i];
    const cv::Mat & currentMask = toTrimMasks[i];
    int minRow = currentScan.rows;
    int minCol = currentScan.cols;
    int maxRow = 0;
    int maxCol = 0;

    for (int i = 0; i < currentScan.rows; ++i) {
      const uchar * src = currentScan.ptr<uchar>(i);
      for (int j = 0; j < currentScan.cols; ++j) {
        if (src[j]!=255) {
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

void place::displayGraph(const Eigen::MatrixXE & adjacencyMatrix,
  const std::vector<place::node> & nodes,
  const std::vector<std::vector<Eigen::MatrixXb> > & scans,
  const std::vector<std::vector<Eigen::Vector2i> > & zeroZeros) {

  const int rows = adjacencyMatrix.rows();
  const int cols = adjacencyMatrix.cols();
  int numBreaks = 0;
  for (int i = 0; i < cols; ++i) {
    const place::node & nodeA = nodes[i];
    if (nodeA.color != 40) continue;
    for (int j = 0; j < rows; ++j) {
      const place::node & nodeB = nodes[j];

      /*if (i > j)
        continue;*/
      if (adjacencyMatrix(j, i).w == 0)
        continue;
      std::cout << "(" << i << ", " << j << ")" << std::endl;

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
        for (int l = 0; l < aScan.rows(); ++l) {
          if (l + yOffset < 0 || l + yOffset >= output.rows)
            continue;
          if (k + xOffset < 0 || k + xOffset >= output.cols)
            continue;

          if (aScan(l, k) != 0) {
            _output(l + yOffset, k + xOffset)[0]=0;
            _output(l + yOffset, k + xOffset)[1]=0;
            _output(l + yOffset, k + xOffset)[2]=255;
          }
        }
      }

      yOffset = nodeB.s.y - zeroZeroB[1];
      xOffset = nodeB.s.x - zeroZeroB[0];
      for (int k = 0; k < bScan.cols(); ++k) {
        for (int l = 0; l < bScan.rows(); ++l) {
          if (l + yOffset < 0 || l + yOffset >= output.rows)
            continue;
          if (k + xOffset < 0 || k + xOffset >= output.cols)
            continue;

          if (bScan(l, k) != 0) {
            _output(l + yOffset, k + xOffset)[0]=255;
            _output(l + yOffset, k + xOffset)[1]=0;
            _output(l + yOffset, k + xOffset)[2]=0;
          }
        }
      }

      cvNamedWindow("Preview", CV_WINDOW_NORMAL);
      cv::imshow("Preview", output);
      if (!FLAGS_quiteMode) {
        std::cout << "Color A: " << nodeA.color << "  Color B: " << nodeB.color << std::endl;
        std::cout << adjacencyMatrix(j,i) << std::endl;
        std::cout << "urnary: " << nodeA.w << "   " << nodeB.w << std::endl;
      }
      cv::waitKey(0);
      ~output;
    }
  }
}

void place::displayBest(const std::vector<const place::node *> & bestNodes,
  const std::vector<std::vector<Eigen::MatrixXb> > & scans,
  const std::vector<std::vector<Eigen::Vector2i> > & zeroZeros) {

  std::cout << "Displaying solution" << std::endl;

  for (auto & n : bestNodes) {
    std::cout << n->color << std::endl;
    cv::Mat output(fpColor.rows, fpColor.cols, CV_8UC3);
    fpColor.copyTo(output);
    cv::Mat_<cv::Vec3b> _output = output;
    const Eigen::MatrixXb & scan = scans[n->color][n->s.rotation];
    const Eigen::Vector2i zeroZero = zeroZeros[n->color][n->s.rotation];
    const int xOffset = n->s.x - zeroZero[0];
    const int yOffset = n->s.y - zeroZero[1];

    for (int i = 0; i < scan.cols(); ++i) {
      for (int j = 0; j < scan.rows(); ++j) {
        if (scan(j,i) != 0) {
          if (j + yOffset < 0 || j + yOffset >= output.rows)
            continue;
          if (i + xOffset < 0 || i + xOffset >= output.cols)
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

static inline double getAngle (const Eigen::Vector3i & A,
  const Eigen::Vector3i & aZero, const Eigen::Vector3i & B,
  const Eigen::Vector3i & bZero) {
  const Eigen::Vector3i rayA = A - aZero;
  const Eigen::Vector3i rayB = B - bZero;
  return acos (static_cast<double>(rayA.dot(rayB)) /
    (static_cast<double>(rayA.norm()) * rayB.norm()));
}

#pragma omp declare reduction \
  (merge : std::vector<Eigen::Vector3d> : omp_out.insert(omp_out.end(),\
   omp_in.begin(), omp_in.end()))

place::edge place::compare3D(const place::voxelGrid & aPoint,
  const place::voxelGrid & bPoint,
  const place::voxelGrid & aFree,
  const place::voxelGrid & bFree,
  const place::cube & aRect, const place::cube & bRect) {

  const int z = aRect.Z2 - aRect.Z1 + 1;
  const int Xrows = aRect.Y2 - aRect.Y1 + 1;
  const int Xcols = aRect.X2 - aRect.X1 + 1;

  double pointAgreement = 0.0, freeSpaceAgreementA = 0.0,
    freeSpaceAgreementB = 0.0, freeSpaceCross = 0.0;

  double totalPointA = aPoint.c,
    totalPointB = bPoint.c,
    averageFreeSpace = (aFree.c + bFree.c)/2.0;

  #pragma omp parallel for shared(aPoint, bPoint, aFree, bFree, aRect, bRect) \
    reduction(+: pointAgreement, freeSpaceAgreementA, freeSpaceAgreementB) \
    reduction(+: freeSpaceCross, totalPointA, totalPointB) \
    reduction(+: averageFreeSpace)
  for (int k = 0; k < z; ++k) {
    auto & Ap = aPoint.v[k + aRect.Z1];
    auto & Bp = bPoint.v[k + bRect.Z1];
    auto & Af = aFree.v[k + aRect.Z1];
    auto & Bf = bFree.v[k + bRect.Z1];

    if ((Ap.sum() == 0 && Af.sum() == 0) || (Bp.sum() == 0 && Bf.sum() == 0))
      continue;
    for (int i = 0; i < Xcols; ++i) {
      for (int j = 0; j < Xrows; ++j) {
        const Eigen::Vector3i APos (i + aRect.X1, j + aRect.Y1, aRect.Z1 + k);
        const Eigen::Vector3i BPos (i + bRect.X1, j + bRect.Y1, bRect.Z1 + k);

        if ((localGroup(Ap, APos[1], APos[0], 2) &&
          Bp(BPos[1], BPos[0])) || (Ap(APos[1], APos[0]) &&
          localGroup(Bp, BPos[1], BPos[0], 2)))
          ++pointAgreement /*+= Ap(APos[1], APos[0]) + Bp(BPos[1], BPos[0])*/;

        if (Ap(APos[1], APos[0]) &&
          Bf(BPos[1], BPos[0]))
          ++freeSpaceAgreementA/* += Bf(BPos[1], BPos[0])*/;

        if (Bp(BPos[1], BPos[0]) &&
          Af(APos[1], APos[0]))
            ++freeSpaceAgreementB /*+= Af(APos[1], APos[0])*/;

        if (Bf(BPos[1], BPos[0]) &&
          Af(APos[1], APos[0]))
          ++freeSpaceCross/* += Bf(BPos[1], BPos[0]) + Af(APos[1], APos[0])*/;

        if (Bf(BPos[1], BPos[0]))
          ++averageFreeSpace;

        if (Af(APos[1], APos[0]))
          ++averageFreeSpace;

        if (Ap(APos[1], APos[0]))
          ++totalPointA;

        if (Bp(BPos[1], BPos[0]))
          ++totalPointB;
      }
    }
  }

  averageFreeSpace /= 2.0;
  totalPointA /= 2.0;
  totalPointB /= 2.0;
  double averagePoint = (totalPointA + totalPointB)/2.0;
  if (averageFreeSpace == 0.0 || averagePoint == 0.0 ||
    totalPointA == 0.0 || totalPointB == 0.0) {
    return place::edge();
  }

  double weight = pointAgreement/averagePoint
    - (freeSpaceAgreementB/totalPointB + freeSpaceAgreementA/totalPointA);

  return place::edge (pointAgreement/averagePoint,
    freeSpaceAgreementA/totalPointA, freeSpaceAgreementB/totalPointB,
    freeSpaceCross/averageFreeSpace, weight, 0);
}

inline void place::loadInVoxel(const std::string & name,
  place::voxelGrid & dst) {

  std::ifstream in (name, std::ios::in | std::ios::binary);
  dst.loadFromFile(in);
  in.close();
}

void place::TRWSolver(const Eigen::MatrixXE & adjacencyMatrix,
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
    for (auto & n : nodes) {
      if (n.color == prevNode->color) {
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
  for (size_t i = 0, offset = 0; i < numVars; ++i) {
    const size_t shape [] = {numberOfLabels[i]};
    Function f(shape, shape + 1);
    for (int j = 0; j < numberOfLabels[i]; ++j) {
      f(j) = nodes[offset + j].w;
    }
    // f(shape[0] - 1) = 0;
    Model::FunctionIdentifier fid = gm.addFunction(f);
    const size_t factors [] = {i};
    gm.addFactor(fid, factors, factors + 1);

    offset += numberOfLabels[i];
  }

  //Add pairwise terms
  for (size_t i = 0, colOffset = 0, rowOffset = 0; i < numVars; ++i) {
    rowOffset += numberOfLabels[i];
    int rowOcp = rowOffset;
    for (size_t j = i + 1; j < numVars; ++j) {
      Eigen::MatrixXE currentMat = adjacencyMatrix.block(rowOcp, colOffset,
        numberOfLabels[j], numberOfLabels[i]);

      const size_t shape [] = {numberOfLabels[i], numberOfLabels[j]};
      Function f(shape, shape + 2);
      for (int a = 0; a < currentMat.cols(); ++a) {
        for (int b = 0; b < currentMat.rows(); ++b) {
          f(a,b) = currentMat(b,a).w /*+ currentMat(b,a).panoW*/;
        }
      }
      /* for (int a = 0; a < shape[1]; ++a) {
        f(shape[0] - 1, a) = 0;
      }
      for (int a = 0; a < shape[0]; ++a) {
        f(a, shape[1] - 1) = 0;
      } */

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

  Parameter parameter (1000);
  Solver solver (gm, parameter);
  Solver::VerboseVisitorType verboseVisitor;
  solver.infer(verboseVisitor);

  std::vector<Model::LabelType> labeling (numVars);
  solver.arg(labeling);

  std::cout << "Labels : ";
  for (auto & l : labeling)
    std::cout << l << "_";
  std::cout << std::endl;

  bestNodes.reserve(numVars);
  for (int i = 0, offset = 0; i < numVars; ++i) {
    if (labeling[i] < numberOfLabels[i])
      bestNodes.push_back(&nodes[offset + labeling[i]]);
    offset += numberOfLabels[i];
  }
}

static void condenseStack(std::vector<GRBVar> & stacked,
  GRBModel & model) {
  if (stacked.size() == 2) {
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
  std::map<std::pair<int,int>, GRBVar > & preStacked,
  std::vector<GRBVar> & stacked) {
  int i = 0;
  for (; i < toStack.size() - 1; i+=2) {
    std::pair<int, int> key (toStack[i], toStack[i+1]);
    auto it = preStacked.find(key);
    if (it == preStacked.end()) {
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
  for (; i < toStack.size(); ++i) {
    if (stacked.size() > 1) {
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
  while (stacked.size() > 2)
    condenseStack(stacked, model);
}

void place::MIPSolver(const Eigen::MatrixXE & adjacencyMatrix,
  const std::unordered_map<std::vector<int>, double> & highOrder, const std::vector<place::node> & nodes,
  std::vector<const place::node *> & bestNodes) {

  std::vector<int> numberOfLabels;
  {
    int i = 0;
    const place::node * prevNode = &nodes[0];
    for (auto & n : nodes) {
      if (n.color == prevNode->color) {
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
    env.set("TimeLimit", "600");

    GRBModel model = GRBModel(env);

    double * upperBound = new double [numOpts];
    char * type = new char [numOpts];
    for (int i = 0; i < numOpts; ++i) {
      upperBound[i] = 1.0;
      type[i] = GRB_BINARY;
    }

    GRBVar * varList = model.addVars(NULL, upperBound, NULL, type, NULL, numOpts);
    GRBVar * inverseVarList = model.addVars(NULL, upperBound, NULL, type, NULL, numOpts);
    delete [] upperBound;
    delete [] type;
    // Integrate new variables
    model.update();
    for (int i = 0; i < numOpts; ++i) {
      model.addConstr(varList[i] + inverseVarList[i], GRB_EQUAL, 1.0);
    }

    GRBQuadExpr objective = 0.0;
    for (int i = 0; i < numOpts; ++i) {
      for (int j = i + 1; j < numOpts; ++j) {
        if (adjacencyMatrix(j,i).w == 0.0)
          continue;

        objective += (adjacencyMatrix(j,i).w + adjacencyMatrix(j,i).shotW)*varList[i]*varList[j];
      }
      const place::posInfo & currentScore = nodes[i].s;
      double scanExplained =
        (currentScore.scanPixels - currentScore.scanFP)/(currentScore.scanPixels);
      double fpExplained =
      (currentScore.fpPixels - currentScore.fpScan)/(currentScore.fpPixels);

      objective += varList[i]*(fpExplained + scanExplained)/2.0;
    }

    for (int i = 0, offset = 0; i < numVars; ++i) {
      GRBLinExpr constr = 0.0;
      double * coeff = new double [numberOfLabels[i]];
      for (int a = 0; a < numberOfLabels[i]; ++ a)
        coeff[a] = 1.0;

      constr.addTerms(coeff, varList + offset, numberOfLabels[i]);
      model.addConstr(constr, GRB_LESS_EQUAL, 1.0);
      offset += numberOfLabels[i];
      delete [] coeff;
    }


    /*for (auto & it : highOrder) {
      auto & incident = it.first;
      for (auto & i : incident)
        objective += varList[i]*it.second;
    }
*/
    std::map<std::pair<int, int>, GRBVar > termCondense;
    for (auto & it : highOrder) {
      auto & incident = it.first;
      /*if (incident.size() == 2) {
        objective -= inverseVarList[incident[0]]*inverseVarList[incident[1]]*it.second;
      } else if (incident.size() == 1) {

      }else*/ if(incident.size() > 3) {
        std::vector<GRBVar> final;
        stackTerms(incident, inverseVarList, model, termCondense, final);
        objective -= final[0]*final[1]*it.second;
      }
    }
    model.update();
    model.setObjective(objective, GRB_MAXIMIZE);
    model.optimize();

    for (int i = 0, offset = 0, k = 0; i < numOpts; ++i) {
      if (varList[i].get(GRB_DoubleAttr_X) == 1.0) {
        bestNodes.push_back(&(nodes[i]));
        std::cout << i - offset << "_";
      }
      if (numberOfLabels[k] == i + 1 - offset)
        offset += numberOfLabels[k++];
    }
    std::cout << std::endl;
    std::cout << "Labeling found for " << bestNodes.size() << " out of " << numVars << " options" << std::endl;
  } catch(GRBException e) {
    std::cout << "Error code = " << e.getErrorCode() << std::endl;
    std::cout << e.getMessage() << std::endl;
  } catch(...) {
    std::cout << "Exception during optimization" << std::endl;
  }
}

bool place::reloadGraph(Eigen::MatrixXE & adjacencyMatrix) {
  const std::string graphName = FLAGS_preDoneV2 + "graph.dat";
  std::ifstream in (graphName, std::ios::in | std::ios::binary);

  if (!in.is_open())
    return false;

  int cols, rows;
  in.read(reinterpret_cast<char *>(&rows), sizeof(rows));
  in.read(reinterpret_cast<char *>(&cols), sizeof(cols));

  adjacencyMatrix = Eigen::MatrixXE(rows, cols);
  in.read(reinterpret_cast<char *>(adjacencyMatrix.data()),
    sizeof(place::edge)*adjacencyMatrix.size());

  in.close();

  return true;
}

void place::saveGraph(Eigen::MatrixXE & adjacencyMatrix) {
  const std::string graphName = FLAGS_preDoneV2 + "graph.dat";
  std::ofstream out (graphName, std::ios::out | std::ios::binary);

  if (!FLAGS_save)
    return;

  int rows = adjacencyMatrix.rows();
  int cols = adjacencyMatrix.cols();

  out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
  out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));

  out.write(reinterpret_cast<const char *>(adjacencyMatrix.data()),
    sizeof(place::edge)*adjacencyMatrix.size());

  out.close();
}

bool place::localGroup(auto & toCheck, const int yOffset,
  const int xOffset, const int range) {
  for (int i = -range; i <= range; ++i) {
    for (int j = -range; j <= range; ++j) {
      if (yOffset + j < 0 || yOffset + j >= toCheck.rows())
        continue;
      if (xOffset + i < 0 || xOffset + i >= toCheck.cols())
        continue;
      if (toCheck(yOffset + j, xOffset + i))
        return true;
    }
  }

  return false;
}

static double harmonic(int stop, double r) {
  double val = 0.0;
  for (int i = 1; i <= stop; ++i) {
    double v = std::pow(static_cast<double>(i), r);
    val += 0.1/v;
  }
  return val;
}

void place::createHigherOrderTerms(const std::vector<std::vector<Eigen::MatrixXb> > & scans,
  const std::vector<std::vector<Eigen::Vector2i> > & zeroZeros,
  const std::vector<place::node> & nodes, std::unordered_map<std::vector<int>, double> &
    highOrder) {

  std::vector<int> numberOfLabels;
  {
    int i = 0;
    const place::node * prevNode = &nodes[0];
    for (auto & n : nodes) {
      if (n.color == prevNode->color) {
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

  Eigen::ArrayXH hMap (floorPlan.rows, floorPlan.cols);
  for (int a = 0, offset = 0; a < numberOfLabels.size(); ++a) {
    for (int b = 0; b < std::min(numberOfLabels[a], 3); ++b) {
      auto & currentNode = nodes[b + offset];
      auto & currentScan = scans[currentNode.color][currentNode.s.rotation];
      auto & zeroZero = zeroZeros[currentNode.color][currentNode.s.rotation];
      const int xOffset = currentNode.s.x - zeroZero[0],
        yOffset = currentNode.s.y - zeroZero[1];

      for (int j = 0; j < currentScan.rows(); ++j) {
        if (j + yOffset < 0 || j + yOffset >= floorPlan.rows)
          continue;
        const uchar * src = floorPlan.ptr<uchar>(j + yOffset);
        for (int i = 0; i < currentScan.cols(); ++i) {
          if (i + xOffset < 0 || i + xOffset >= floorPlan.cols)
            continue;
          if (src[i + xOffset] != 255) {
            if (localGroup(currentScan, j, i, 2)) {
              hMap(j+yOffset, i + xOffset).incident.push_back(b + offset);
              hMap(j+yOffset, i + xOffset).weight += currentNode.w;
            }
          }
        }
      }
    }
    offset += numberOfLabels[a];
  }

  place::hOrder * data = hMap.data();
  for (int i = 0; i < hMap.size(); ++i) {
    if ((data + i)->incident.size() != 0) {
      // const double scale = harmonic((data + i)->incident.size(), 0.0);
      (data + i)->weight /= (data + i)->incident.size();
      // (data + i)->weight *= scale;
    }
  }

  for (int i = 0; i < hMap.size(); ++i) {
    std::vector<int> & key = (data + i)->incident;
    if (key.size() != 0 && (data + i)->weight > 0.0) {
      auto it = highOrder.find(key);
      if (it != highOrder.end())
        it->second += (data + i)->weight;
      else
        highOrder.emplace(key, (data + i)->weight);
    }
  }

  double average = 0.0, aveTerms = 0.0;
  for (auto & it : highOrder) {
    average += it.second;
    aveTerms += it.first.size();
  }
  average /= highOrder.size();
  aveTerms /= highOrder.size();

  double sigma = 0.0, sigTerms = 0.0;
  for (auto & it : highOrder) {
    sigma += (it.second - average)*(it.second -average);
    sigTerms += (it.first.size() - aveTerms) * (it.first.size() - aveTerms);
  }
  sigma /= (highOrder.size() - 1);
  sigma = sqrt(sigma);

  sigTerms /= (highOrder.size() - 1);
  sigTerms = sqrt(sigTerms);

  std::cout << "average: " << average << "   sigma: " << sigma << std::endl;

  for (auto & it : highOrder) {
    it.second = std::max(0.0,(((it.second - average)/(sigma) + 1.0)/2.0));
    const double significance = (it.first.size() - aveTerms)/sigTerms;
    if(significance < 10000)
      highOrder.erase(it.first);
  }
}

void place::displayHighOrder(const std::unordered_map<std::vector<int>, double> highOrder,
  const std::vector<place::node> & nodes,
  const std::vector<std::vector<Eigen::MatrixXb> > & scans,
  const std::vector<std::vector<Eigen::Vector2i> > & zeroZeros) {

  for (auto & it : highOrder) {
    auto & key = it.first;
    cv::Mat output (fpColor.rows, fpColor.cols, CV_8UC3);
    fpColor.copyTo(output);
    cv::Mat_<cv::Vec3b> _output = output;
    for (auto & i : key) {
      const place::node & nodeA = nodes[i];

      auto & aScan = scans[nodeA.color][nodeA.s.rotation];

      auto & zeroZeroA = zeroZeros[nodeA.color][nodeA.s.rotation];

      int yOffset = nodeA.s.y - zeroZeroA[1];
      int xOffset = nodeA.s.x - zeroZeroA[0];
      for (int k = 0; k < aScan.cols(); ++k) {
        for (int l = 0; l < aScan.rows(); ++l) {
          if (l + yOffset < 0 || l + yOffset >= output.rows)
            continue;
          if (k + xOffset < 0 || k + xOffset >= output.cols)
            continue;

          if (aScan(l, k) != 0) {
            _output(l + yOffset, k + xOffset)[0]=0;
            _output(l + yOffset, k + xOffset)[1]=0;
            _output(l + yOffset, k + xOffset)[2]=255;
          }
        }
      }
    }
    cvNamedWindow("Preview", CV_WINDOW_NORMAL);
    cv::imshow("Preview", output);
    if (!FLAGS_quiteMode) {
      std::cout << it.second << std::endl;
      for (auto & i : key)
        std::cout << i << "_";
      std::cout << std::endl;
    }
    cv::waitKey(0);
    ~output;
  }
}

void place::normalizeWeights(Eigen::MatrixXE & adjacencyMatrix,
  std::vector<place::node> & nodes) {

  std::vector<int> numberOfLabels;
  {
    int i = 0;
    const place::node * prevNode = &nodes[0];
    for (auto & n : nodes) {
      if (n.color == prevNode->color) {
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

  for (int a = 0, rowOffset = 0; a < numVars; ++a) {
    double averageW = 0.0, averageS = 0.0, averageP = 0;
    int countW = 0, countS = 0, countP = 0;
    for (int j = 0; j < numberOfLabels[a]; ++j) {
      for (int i = 0; i < adjacencyMatrix.cols(); ++i) {
        const double weight = adjacencyMatrix(j + rowOffset, i).w;
        const double shot = adjacencyMatrix(j + rowOffset, i).shotW;
        const double pano = adjacencyMatrix(j + rowOffset, i).panoW;
        if(weight) {
          averageW += weight;
          ++countW;
        }
        if (shot) {
          averageS += shot;
          ++countS;
        }
        if (pano) {
          averageP += pano;
          ++countP;
        }
      }
    }

    averageW /= countW;
    averageS /= countS;
    averageP /= countP;

    double sigmaW = 0, sigmaS = 0, sigmaP = 0;
    for (int j = 0; j < numberOfLabels[a]; ++j) {
      for (int i = 0; i < adjacencyMatrix.cols(); ++i) {
        const double weight = adjacencyMatrix(j + rowOffset, i).w;
        const double shot = adjacencyMatrix(j + rowOffset, i).shotW;
        const double pano = adjacencyMatrix(j + rowOffset, i).panoW;
        if(weight)
          sigmaW += (weight - averageW)*(weight - averageW);

        if (shot)
          sigmaS += (shot - averageS)*(shot - averageS);

        if (pano)
          sigmaP += (pano - averageP)*(pano - averageP);
      }
    }
    sigmaW /= countW;
    sigmaW = sqrt(sigmaW);

    sigmaS /= countS;
    sigmaS = sqrt(sigmaS);

    sigmaP /= countP;
    sigmaP = sqrt(sigmaP);

    for (int j = 0; j < numberOfLabels[a]; ++j) {
      for (int i = 0; i < adjacencyMatrix.cols(); ++i) {
        const double weight = adjacencyMatrix(j + rowOffset, i).w;
        const double shot = adjacencyMatrix(j + rowOffset, i).shotW;
        const double pano = adjacencyMatrix(j + rowOffset, i).panoW;
        if (weight && countW > 1 && sigmaW)
          adjacencyMatrix(j + rowOffset, i).w = (weight - averageW)/sigmaW;

        if (shot && countS > 1 && sigmaS)
          adjacencyMatrix(j + rowOffset, i).shotW = (shot - averageS)/sigmaS;

        if (pano && countP > 1 && sigmaP)
          adjacencyMatrix(j + rowOffset, i).panoW = (pano - averageP)/sigmaP;

      }
    }
    rowOffset += numberOfLabels[a];
  }
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


  for (auto & vec : scoreInfo) {
    const std::string scanName = FLAGS_dmFolder + pointFileNames[vec[0].scanNum];
    const std::string rotationFile = FLAGS_rotFolder + rotationFileNames[vec[0].scanNum];
    const std::string zerosFile = FLAGS_zerosFolder + zerosFileNames[vec[0].scanNum];

    std::vector<cv::Mat> rotatedScans, toTrim;
    std::vector<Eigen::Vector2i> zeroZero;
    place::loadInScans(scanName, rotationFile, zerosFile, toTrim, zeroZero);
    place::trimScans(toTrim, rotatedScans, zeroZero);
    std::vector<Eigen::SparseMatrix<double> > rSSparse;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));

    for (auto & scan : rotatedScans) {
      cv::Mat dst;
      cv::erode(scan, dst, element);
      rSSparse.push_back(scanToSparse(dst));
    }
    toTrim.clear();
    rotatedScans.clear();

    for (auto & m : vec) {
      const Eigen::SparseMatrix<double> & currentScan = rSSparse[m.s.rotation];
      const int xOffset = m.s.x - zeroZero[m.s.rotation][0];
      const int yOffset = m.s.y - zeroZero[m.s.rotation][1];

      for (int i = 0; i < currentScan.outerSize(); ++i) {
        for (Eigen::SparseMatrix<double>::InnerIterator it (currentScan, i); it; ++it) {
          if (yOffset + it.row() >= weightedFp.rows() || yOffset + it.row() < 0)
            continue;
          if (xOffset + it.col() >= weightedFp.cols() || xOffset + it.col() < 0)
            continue;

          if (weightedFp(yOffset + it.row(), xOffset + it.col()) != 0)
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
  if (!FLAGS_previewOut)
    return;
  double maxV = 0;

  for (int i = 0; i < weightedFloorPlan.outerSize(); ++i) {
    for (Eigen::SparseMatrix<double>::InnerIterator it (weightedFloorPlan, i); it; ++it ) {
      maxV = std::max(maxV, it.value());
    }
  }

  cv::Mat out (weightedFloorPlan.rows(), weightedFloorPlan.cols(), CV_8UC3, cv::Scalar::all(255));
  cv::Mat_<cv::Vec3b> _out = out;

  for (int i = 0; i < weightedFloorPlan.outerSize(); ++i) {
    for (Eigen::SparseMatrix<double>::InnerIterator it (weightedFloorPlan, i); it; ++it ) {
      if (it.value() > 0) {
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