#include "placeScan_multiLabeling.h"
#include "placeScan_panoramaMatcher.h"
#include "placeScan_placeScanHelper2.h"

#include <fstream>
#include <iostream>

#include <boost/progress.hpp>
#include <boost/timer/timer.hpp>

#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <opengm/functions/explicit_function.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/discretespace.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/inference/trws/trws_trws.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/maximizer.hxx>

#include <dirent.h>
#include <math.h>
#include <omp.h>

DEFINE_int32(graphColor, -1, "Color of the graph to display");

template <typename T>
static void displayCollapsed(T &collapsed, const std::string &windowName) {
  double average, sigma;
  const auto dataPtr = collapsed.data();
  std::tie(average, sigma) =
      place::aveAndStdev(dataPtr, dataPtr + collapsed.size(),
                         [](auto &v) { return v; }, [](auto &v) { return v; });

  cv::Mat heatMap(collapsed.rows(), collapsed.cols(), CV_8UC3,
                  cv::Scalar::all(255));
  for (int i = 0; i < heatMap.rows; ++i) {
    uchar *dst = heatMap.ptr<uchar>(i);
    for (int j = 0; j < heatMap.cols; ++j) {
      if (collapsed(i, j)) {
        const int gray = cv::saturate_cast<uchar>(
            255.0 * (collapsed(i, j) - average) / (1.0 * sigma));
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
        dst[j * 3] = blue;
        dst[j * 3 + 1] = green;
        dst[j * 3 + 2] = red;
      }
    }
  }
  cvNamedWindow(windowName.data(), CV_WINDOW_NORMAL);
  cv::imshow(windowName, heatMap);
}

template <typename T>
static void displayVoxelGrid(const T &grid, const std::string &windowName) {
  Eigen::MatrixXd collapsed =
      Eigen::MatrixXd::Zero(grid[0].rows(), grid[0].cols());

  for (int k = 0; k < grid.size(); ++k)
    for (int i = 0; i < grid[0].cols(); ++i)
      for (int j = 0; j < grid[0].rows(); ++j)
        collapsed(j, i) += grid[k](j, i) ? 1 : 0;

  displayCollapsed(collapsed, windowName);
}

template <typename T>
static void displayVoxelGridS(const T &sparse, const std::string &windowName) {
  typedef typename T::value_type::InnerIterator InnerIT;
  Eigen::MatrixXd collapsed =
      Eigen::MatrixXd::Zero(sparse[0].rows(), sparse[0].cols());
  for (int k = 0; k < sparse.size(); ++k) {
    auto &current = sparse[k];
    for (int i = 0; i < current.outerSize(); ++i) {
      for (InnerIT it(current, i); it; ++it) {
        collapsed(it.row(), it.col()) += it.value();
      }
    }
  }
  displayCollapsed(collapsed, windowName);
}
template <typename T, typename S>
static void displayCollapsed(const T &collapsedA, const S &collapsedB,
                             const place::cube &aRect,
                             const place::cube &bRect) {
  const int Xrows = aRect.Y2 - aRect.Y1 + 1;
  const int Xcols = aRect.X2 - aRect.X1 + 1;

  double averageA, sigmaA, averageB, sigmaB;
  const auto dataPtrA = collapsedA.data();
  std::tie(averageA, sigmaA) =
      place::aveAndStdev(dataPtrA, dataPtrA + collapsedA.size(),
                         [](auto &v) { return v; }, [](auto &v) { return v; });

  const auto dataPtrB = collapsedB.data();
  std::tie(averageB, sigmaB) =
      place::aveAndStdev(dataPtrB, dataPtrB + collapsedB.size(),
                         [](auto &v) { return v; }, [](auto &v) { return v; });

  cv::Mat heatMap(Xrows, Xcols, CV_8UC3, cv::Scalar::all(255));
  for (int i = 0; i < heatMap.rows; ++i) {
    uchar *dst = heatMap.ptr<uchar>(i);
    for (int j = 0; j < heatMap.cols; ++j) {
      double aVal = collapsedA(i, j);
      double bVal = collapsedB(i, j);

      if (aVal && bVal) {
        int red = 255, green = 0, blue = 255;
        dst[j * 3] = blue;
        dst[j * 3 + 1] = green;
        dst[j * 3 + 2] = red;
      } else if (aVal) {
        int red = 255, green = 0, blue = 0;
        dst[j * 3] = blue;
        dst[j * 3 + 1] = green;
        dst[j * 3 + 2] = red;
      } else if (bVal) {
        int red = 0, green = 0, blue = 255;
        dst[j * 3] = blue;
        dst[j * 3 + 1] = green;
        dst[j * 3 + 2] = red;
      }
    }
  }

  cvNamedWindow("Preview", CV_WINDOW_NORMAL);
  cv::imshow("Preview", heatMap);
  cv::waitKey(0);
}

template <typename T, typename S>
static void displayVoxelGrid(const T &voxelA, const S &voxelB,
                             const place::cube &aRect,
                             const place::cube &bRect) {
  const int z = aRect.Z2 - aRect.Z1 + 1;
  const int Xrows = aRect.Y2 - aRect.Y1 + 1;
  const int Xcols = aRect.X2 - aRect.X1 + 1;

  Eigen::MatrixXd collapsedA = Eigen::MatrixXd::Zero(Xrows, Xcols);
  Eigen::MatrixXd collapsedB = Eigen::MatrixXd::Zero(Xrows, Xcols);
  for (int k = 0; k < z; ++k) {
    for (int i = 0; i < Xcols; ++i) {
      for (int j = 0; j < Xrows; ++j) {
        Eigen::Vector3i APos(i + aRect.X1, j + aRect.Y1, z + aRect.Z1);
        Eigen::Vector3i BPos(i + bRect.X1, j + bRect.Y1, z + bRect.Z1);
        collapsedA(j, i) += voxelA[APos[2]](APos[1], APos[0]);
        collapsedB(j, i) += voxelB[BPos[2]](BPos[1], BPos[0]);
      }
    }
  }
  displayCollapsed(collapsedA, collapsedB, aRect, bRect);
}

template <typename TA, typename TB>
static void displayVoxelGridS(const TA &voxelA, const TB &voxelB,
                              const place::cube &aRect,
                              const place::cube &bRect) {
  typedef typename TA::value_type::InnerIterator InnerITA;
  typedef typename TB::value_type::InnerIterator InnerITB;

  const int z = aRect.Z2 - aRect.Z1 + 1;
  const int Xrows = aRect.Y2 - aRect.Y1 + 1;
  const int Xcols = aRect.X2 - aRect.X1 + 1;

  Eigen::MatrixXd collapsedA = Eigen::MatrixXd::Zero(Xrows, Xcols);
  Eigen::MatrixXd collapsedB = Eigen::MatrixXd::Zero(Xrows, Xcols);

  for (int k = 0; k < z; ++k) {
    auto &currentA = voxelA[k + aRect.Z1];
    auto &currentB = voxelB[k + bRect.Z1];

    for (int i = 0; i < currentA.outerSize(); ++i) {
      for (InnerITA it(currentA, i); it; ++it) {
        Eigen::Vector3i APos(it.col() - aRect.X1, it.row() - aRect.Y1, 0);
        if (APos[0] < 0 || APos[0] >= Xcols)
          continue;
        if (APos[1] < 0 || APos[1] >= Xrows)
          continue;
        collapsedA(APos[1], APos[0]) += it.value();
      }
    }

    for (int i = 0; i < currentB.outerSize(); ++i) {
      for (InnerITB it(currentB, i); it; ++it) {
        Eigen::Vector3i BPos(it.col() - bRect.X1, it.row() - bRect.Y1, 0);
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

void place::loadInScansGraph(
    const std::vector<std::string> &pointFileNames,
    const std::vector<std::string> &freeFileNames,
    const std::vector<std::string> &zerosFileNames,
    std::vector<std::vector<Eigen::MatrixXb>> &scans,
    std::vector<std::vector<Eigen::MatrixXb>> &masks,
    std::vector<std::vector<Eigen::Vector2i>> &zeroZeros) {
  for (int i = 0; i < zeroZeros.size(); ++i) {
    const std::string scanName = pointFileNames[i];
    const std::string zerosFile = FLAGS_zerosFolder + zerosFileNames[i];
    const std::string maskName = freeFileNames[i];

    std::vector<cv::Mat> toTrimScans, toTrimMasks, trimmedScans, trimmedMasks,
        toTrimMasksD;
    place::loadInScansAndMasks(scanName, zerosFile, maskName, toTrimScans,
                               toTrimMasks, zeroZeros[i]);

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    for (auto &src : toTrimMasks) {
      cv::Mat dst;
      cv::dilate(src, dst, element);
      toTrimMasksD.push_back(dst);
    }

    place::trimScansAndMasks(toTrimScans, toTrimMasksD, trimmedScans,
                             trimmedMasks, zeroZeros[i]);

    std::vector<Eigen::MatrixXb> tmpvec;
    for (auto &scan : trimmedScans) {
      Eigen::MatrixXb tmp = Eigen::MatrixXb::Zero(scan.rows, scan.cols);
      for (int j = 0; j < scan.rows; ++j) {
        const uchar *src = scan.ptr<uchar>(j);
        for (int k = 0; k < scan.cols; ++k) {
          if (src[k] != 255) {
            const double confidence = 1.0 - src[k] / 255.0;
            if (confidence > 0.75)
              tmp(j, k) = static_cast<char>(1);
          }
        }
      }
      tmpvec.push_back(tmp);
    }
    scans.push_back(tmpvec);
    tmpvec.clear();

    for (auto &mask : trimmedMasks) {
      Eigen::MatrixXb tmp = Eigen::MatrixXb::Zero(mask.rows, mask.cols);
      for (int j = 0; j < mask.rows; ++j) {
        const uchar *src = mask.ptr<uchar>(j);
        for (int k = 0; k < mask.cols; ++k) {
          if (src[k] != 255)
            tmp(j, k) = static_cast<char>(1);
        }
      }
      tmpvec.push_back(tmp);
    }
    masks.push_back(tmpvec);
  }
}

namespace std {
template <> struct hash<std::pair<int, int>> {
  std::hash<int> h;
  std::size_t operator()(const std::pair<int, int> &k) const {
    size_t seed = h(k.first);
    seed ^= h(k.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};
} // std

void place::weightEdges(
    const std::vector<place::node> &nodes,
    const std::vector<std::vector<place::MetaData>> &voxelInfo,
    const std::vector<std::string> &pointVoxelFileNames,
    const std::vector<std::string> &freeVoxelFileNames,
    const std::vector<std::vector<Eigen::Matrix3d>> &rotationMatricies,
    std::vector<place::Panorama> &panoramas, Eigen::MatrixXE &adjacencyMatrix) {
  adjacencyMatrix = Eigen::MatrixXE(nodes.size(), nodes.size());

  struct later {
    int i, j;
    place::cube crossWRTA, crossWRTB;
  };

#pragma omp declare reduction(                                                 \
    merge : std::vector <                                                      \
    later > : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

  const int rows = adjacencyMatrix.rows();
  const int cols = adjacencyMatrix.cols();
  const double scale = buildingScale.getScale();
  std::vector<later> tracker;
// Iterator over the lower triangle of the adjaceny matrix
#pragma omp parallel for reduction(merge : tracker)
  for (int i = 0; i < cols; ++i) {
    const place::node &nodeA = nodes[i];
    for (int j = i + 1; j < rows; ++j) {
      const place::node &nodeB = nodes[j];
      if (nodeA.color == nodeB.color) {
        continue;
      }

      auto &metaA = voxelInfo[nodeA.color][nodeA.rotation];
      auto &metaB = voxelInfo[nodeB.color][nodeB.rotation];

      place::cube aBox, bBox;
      aBox.X1 = nodeA.x * (metaA.vox / scale) - metaA.zZ[0];
      aBox.X2 = aBox.X1 + metaA.x - 1;
      aBox.Y1 = nodeA.y * (metaA.vox / scale) - metaA.zZ[1];
      aBox.Y2 = aBox.Y1 + metaA.y - 1;
      aBox.Z1 = 0 - metaA.zZ[2];
      aBox.Z2 = aBox.Z1 + metaA.z - 1;

      bBox.X1 = nodeB.x * (metaB.vox / scale) - metaB.zZ[0];
      bBox.X2 = bBox.X1 + metaB.x - 1;
      bBox.Y1 = nodeB.y * (metaB.vox / scale) - metaB.zZ[1];
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

      if (!(XSection.X1 > XSection.X2 || XSection.Y1 > XSection.Y2 ||
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

  std::unordered_map<std::pair<int, int>, place::VoxelGrid> pointGrids,
      freeGrids;
  for (auto &c : tracker) {
    auto &nodeA = nodes[c.i];
    auto &nodeB = nodes[c.j];

    auto a = std::make_pair(nodeA.color, nodeA.rotation);

    auto loaded = pointGrids.find(a);
    if (loaded == pointGrids.cend()) {
      std::string name = FLAGS_voxelFolder + "R" +
                         std::to_string(nodeA.rotation) + "/" +
                         pointVoxelFileNames[nodeA.color];
      pointGrids.emplace(a, loadInVoxel(name));

      name = FLAGS_voxelFolder + "R" + std::to_string(nodeA.rotation) + "/" +
             freeVoxelFileNames[nodeA.color];
      freeGrids.emplace(a, loadInVoxel(name));
    }

    auto b = std::make_pair(nodeB.color, nodeB.rotation);
    loaded = pointGrids.find(b);
    if (loaded == pointGrids.cend()) {
      std::string name = FLAGS_voxelFolder + "R" +
                         std::to_string(nodeB.rotation) + "/" +
                         pointVoxelFileNames[nodeB.color];
      pointGrids.emplace(b, loadInVoxel(name));

      name = FLAGS_voxelFolder + "R" + std::to_string(nodeB.rotation) + "/" +
             freeVoxelFileNames[nodeB.color];
      freeGrids.emplace(b, loadInVoxel(name));
    }
  }

  std::cout << tracker.size() << std::endl;
  boost::progress_display *show_progress =
      new boost::progress_display(tracker.size());
  boost::timer::auto_cpu_timer *timer = new boost::timer::auto_cpu_timer();

  for (int k = 0; k < tracker.size(); ++k) {
    const later &current = tracker[k];
    const int i = current.i;
    const int j = current.j;
    const place::node &nodeA = nodes[i];
    const place::node &nodeB = nodes[j];

    if (FLAGS_debugMode && (j != 331 || i != 321))
      continue;

    auto &aPoint =
        pointGrids.find(std::make_pair(nodeA.color, nodeA.rotation))->second;
    auto &aFree =
        freeGrids.find(std::make_pair(nodeA.color, nodeA.rotation))->second;
    auto &bPoint =
        pointGrids.find(std::make_pair(nodeB.color, nodeB.rotation))->second;
    auto &bFree =
        freeGrids.find(std::make_pair(nodeB.color, nodeB.rotation))->second;

    place::edge weight = place::compare3D(aPoint, bPoint, aFree, bFree,
                                          current.crossWRTA, current.crossWRTB);

    auto &RA = rotationMatricies[nodeA.color][nodeA.rotation];
    auto &RB = rotationMatricies[nodeB.color][nodeB.rotation];

    auto &panoA = panoramas[nodeA.color];
    auto &panoB = panoramas[nodeB.color];

    Eigen::Vector3d AZeroZero(nodeA.x, nodeA.y, 0),
        BZeroZero(nodeB.x, nodeB.y, 0);
    AZeroZero /= scale;
    BZeroZero /= scale;
    constexpr double maxDistance = 1.0;
    weight.distance = (AZeroZero - BZeroZero).norm();

    const Eigen::Vector3d aToB = AZeroZero - BZeroZero;
    const Eigen::Vector3d bToA = BZeroZero - AZeroZero;

    pano::compareNCC2(panoA, panoB, RA, RB, aToB, bToA, weight);
#if 0
    if (FLAGS_debugMode) {
      std::cout << "(" << i << "," << j << ")" << std::endl;
      std::cout << weight << std::endl;
      displayVoxelGridS(aPoint.v, "aPoint");
      displayVoxelGridS(bPoint.v, "bPoint");
      displayVoxelGrid(aFree.v, "aFree");
      displayVoxelGrid(bFree.v, "bFree");
      displayVoxelGridS(aPoint.v, bPoint.v, current.crossWRTA,
                        current.crossWRTB);
    }
#endif

    adjacencyMatrix(j, i) = weight;
    ++(*show_progress);
  }

  delete show_progress;
  delete timer;
  // Copy the lower tranalge into the upper
  // triangle
  // adjacencyMatrix.triangluarView<Upper>()
  // = adjacencyMatrix.transpose();
  for (int i = 0; i < cols; ++i)
    for (int j = i + 1; j < rows; ++j)
      adjacencyMatrix(i, j) = adjacencyMatrix(j, i);
}

void place::loadInPlacementGraph(const std::string &imageName,
                                 std::vector<place::node> &nodes,
                                 const int num) {
  const std::string placementName =
      FLAGS_outputV1 + imageName.substr(imageName.find("_") - 3, 3) +
      "_placement_" + imageName.substr(imageName.find(".") - 3, 3) + ".dat";
  std::ifstream in(placementName, std::ios::in | std::ios::binary);

  int numToLoad;
  in.read(reinterpret_cast<char *>(&numToLoad), sizeof(numToLoad));
  numToLoad = std::min(numToLoad, 200);

  std::vector<place::posInfo> scoretmp(numToLoad);
  in.read(reinterpret_cast<char *>(scoretmp.data()),
          sizeof(place::posInfo) * numToLoad);

  std::vector<place::node> nodestmp;
  for (auto &s : scoretmp)
    nodestmp.push_back({s, 0.0, 0.0, num, 0});

  for (auto &n : nodestmp) {
    double scanExplained = (n.scanPixels - n.scanFP) / (n.scanPixels);
    double fpExplained = (n.fpPixels - n.fpScan) / (n.fpPixels);
    const double w = (scanExplained + fpExplained);
    n.w = w;
  }

  double average, sigma;
  std::tie(average, sigma) =
      place::aveAndStdev(nodestmp.begin(), nodestmp.end(),
                         [](const place::node &n) { return n.w; });

  for (auto &n : nodestmp)
    n.nw = (n.w - average) / sigma;

  nodes.insert(nodes.end(), nodestmp.begin(), nodestmp.end());
}

void place::trimScansAndMasks(const std::vector<cv::Mat> &toTrimScans,
                              const std::vector<cv::Mat> &toTrimMasks,
                              std::vector<cv::Mat> &trimmedScans,
                              std::vector<cv::Mat> &trimmedMasks,
                              std::vector<Eigen::Vector2i> &zeroZero) {
  for (int i = 0; i < toTrimScans.size(); ++i) {
    const cv::Mat &currentScan = toTrimScans[i];
    const cv::Mat &currentMask = toTrimMasks[i];
    int minRow = currentScan.rows;
    int minCol = currentScan.cols;
    int maxRow = 0;
    int maxCol = 0;

    for (int i = 0; i < currentScan.rows; ++i) {
      const uchar *src = currentScan.ptr<uchar>(i);
      for (int j = 0; j < currentScan.cols; ++j) {
        if (src[j] != 255) {
          minRow = std::min(i, minRow);
          minCol = std::min(j, minCol);
          maxRow = std::max(i, maxRow);
          maxCol = std::max(j, maxCol);
        }
      }
    }

    cv::Mat trimmedMask(maxRow - minRow + 1, maxCol - minCol + 1, CV_8UC1);
    cv::Mat trimmedScan(maxRow - minRow + 1, maxCol - minCol + 1, CV_8UC1);
    for (int i = minRow; i < maxRow + 1; ++i) {
      const uchar *scanSrc = currentScan.ptr<uchar>(i);
      const uchar *maskSrc = currentMask.ptr<uchar>(i);
      uchar *scanDst = trimmedScan.ptr<uchar>(i - minRow);
      uchar *maskDst = trimmedMask.ptr<uchar>(i - minRow);
      for (int j = minCol; j < maxCol + 1; ++j) {
        scanDst[j - minCol] = scanSrc[j];
        maskDst[j - minCol] = maskSrc[j];
      }
    }
    zeroZero[i][0] -= minCol;
    zeroZero[i][1] -= minRow;
    trimmedScans.push_back(trimmedScan);
    trimmedMasks.push_back(trimmedMask);
  }
}

void place::displayGraph(
    const Eigen::MatrixXE &adjacencyMatrix,
    const std::vector<place::node> &nodes,
    const std::vector<std::vector<Eigen::MatrixXb>> &scans,
    const std::vector<std::vector<Eigen::Vector2i>> &zeroZeros) {
  const int rows = adjacencyMatrix.rows();
  const int cols = adjacencyMatrix.cols();

  std::vector<std::pair<int, int>> usablePairs;

  for (int i = 0; i < cols; ++i) {
    for (int j = 0; j < rows; ++j) {
      const place::node &nodeA = nodes[i];
      const place::node &nodeB = nodes[j];
      if (adjacencyMatrix(j, i).w == 0 && adjacencyMatrix(j, i).panoW == 0)
        continue;

      usablePairs.emplace_back(i, j);
    }
  }

  bool nextI = false;
  int currentI = 0;
  for (int k = 0; k < usablePairs.size();) {
    int i, j;
    std::tie(i, j) = usablePairs[k];

    const place::node &nodeA = nodes[i];
    const place::node &nodeB = nodes[j];

    if (nextI) {
      if (i == currentI) {
        ++k;
        continue;
      } else
        nextI = false;
    }

    if (FLAGS_graphColor != -1 &&
        (nodeA.color != FLAGS_graphColor && nodeB.color != FLAGS_graphColor)) {
      ++k;
      continue;
    }
    /*if (i > j)
      continue;*/

    std::cout << "(" << i << ", " << j << ")" << std::endl;

    const Eigen::MatrixXb &aScan = scans[nodeA.color][nodeA.rotation];
    const Eigen::MatrixXb &bScan = scans[nodeB.color][nodeB.rotation];

    const Eigen::Vector2i &zeroZeroA = zeroZeros[nodeA.color][nodeA.rotation];
    const Eigen::Vector2i &zeroZeroB = zeroZeros[nodeB.color][nodeB.rotation];

    cv::Mat output(fpColor.rows, fpColor.cols, CV_8UC3);
    fpColor.copyTo(output);

    cv::Mat_<cv::Vec3b> _output = output;

    int yOffset = nodeA.y - zeroZeroA[1];
    int xOffset = nodeA.x - zeroZeroA[0];
    for (int k = 0; k < aScan.cols(); ++k) {
      for (int l = 0; l < aScan.rows(); ++l) {
        if (l + yOffset < 0 || l + yOffset >= output.rows)
          continue;
        if (k + xOffset < 0 || k + xOffset >= output.cols)
          continue;

        if (aScan(l, k) != 0) {
          _output(l + yOffset, k + xOffset)[0] = 0;
          _output(l + yOffset, k + xOffset)[1] = 0;
          _output(l + yOffset, k + xOffset)[2] = 255;
        }
      }
    }

    yOffset = nodeB.y - zeroZeroB[1];
    xOffset = nodeB.x - zeroZeroB[0];
    for (int k = 0; k < bScan.cols(); ++k) {
      for (int l = 0; l < bScan.rows(); ++l) {
        if (l + yOffset < 0 || l + yOffset >= output.rows)
          continue;
        if (k + xOffset < 0 || k + xOffset >= output.cols)
          continue;

        if (bScan(l, k) != 0) {
          _output(l + yOffset, k + xOffset)[0] = 255;
          _output(l + yOffset, k + xOffset)[1] = 0;
          _output(l + yOffset, k + xOffset)[2] = 0;
        }
      }
    }

    for (int l = -10; l <= 10; ++l) {
      for (int k = -10; k <= 10; ++k) {
        _output(l + nodeA.y, k + nodeA.x)[0] = 0;
        _output(l + nodeA.y, k + nodeA.x)[1] = 255;
        _output(l + nodeA.y, k + nodeA.x)[2] = 0;

        _output(l + nodeB.y, k + nodeB.x)[0] = 0;
        _output(l + nodeB.y, k + nodeB.x)[1] = 255;
        _output(l + nodeB.y, k + nodeB.x)[2] = 0;
      }
    }

    cvNamedWindow("Preview", CV_WINDOW_NORMAL);

    cv::imshow("Preview", output);

    std::cout << "Color A: " << nodeA.color << "  Color B: " << nodeB.color
              << std::endl
              << adjacencyMatrix(j, i) << std::endl
              << "urnary: " << nodeA.nw << "  " << nodeA.hWeight << "   "
              << nodeB.nw << "  " << nodeB.hWeight << std::endl;

    int kc = cv::waitKey(0);
    if (kc == 27) {
      nextI = true;
      currentI = i;
    } else if (kc == 8) {
      k = k == 0 ? 0 : k - 1;
    } else
      ++k;
  }
}

void place::displayBest(
    const std::vector<place::SelectedNode> &bestNodes,
    const std::vector<std::vector<Eigen::MatrixXb>> &scans,
    const std::vector<std::vector<Eigen::MatrixXb>> &masks,
    const std::vector<std::vector<Eigen::Vector2i>> &zeroZeros) {
  std::cout << "Displaying solution" << std::endl;

  cv::Mat all = fpColor.clone();
  cv::Mat seen(fpColor.size(), CV_8UC1, cv::Scalar::all(0));

  cv::Mat_<cv::Vec3b> _all = all;
  cv::Mat_<uchar> _seen = seen;

  for (auto &n : bestNodes) {
    auto &scan = scans[n.color][n.rotation];
    auto &mask = masks[n.color][n.rotation];
    auto &zeroZero = zeroZeros[n.color][n.rotation];
    const int xOffset = n.x - zeroZero[0];
    const int yOffset = n.y - zeroZero[1];
    auto color = randomColor();

    for (int i = 0; i < scan.cols(); ++i) {
      if (i + xOffset < 0 || i + xOffset >= all.cols)
        continue;

      for (int j = 0; j < scan.rows(); ++j) {
        if (j + yOffset < 0 || j + yOffset >= all.rows)
          continue;
        if (scan(j, i) != 0) {
          _all(yOffset + j, xOffset + i) = color;
#if 0
          if (!n.locked) {
            _all(yOffset + j, xOffset + i)[0] = 0;
            _all(yOffset + j, xOffset + i)[1] = 0;
            _all(yOffset + j, xOffset + i)[2] = 255;
          } else {
            _all(yOffset + j, xOffset + i)[0] = 255;
            _all(yOffset + j, xOffset + i)[1] = 0;
            _all(yOffset + j, xOffset + i)[2] = 0;
          }
#endif
        }
        if (mask(j, i) != 0)
          _seen(yOffset + j, xOffset + i) = 255;
      }
    }

    for (int j = -10; j < 10; ++j) {
      uchar *dst2 = all.ptr<uchar>(j + n.y);
      for (int i = -10; i < 10; ++i) {
        const int col = 3 * (i + n.x);

        dst2[col + 0] = 0;
        dst2[col + 1] = 255;
        dst2[col + 2] = 0;
      }
    }
  }

  for (int i = 0; i < bestNodes.size();) {
    auto &n = bestNodes[i];
    std::cout << n << std::endl;
    cv::Mat output = fpColor.clone();
    if (n.agreement != -1000) {

      cv::Mat_<cv::Vec3b> _output = output;
      auto &scan = scans[n.color][n.rotation];
      auto &zeroZero = zeroZeros[n.color][n.rotation];
      const int xOffset = n.x - zeroZero[0];
      const int yOffset = n.y - zeroZero[1];

      for (int i = 0; i < scan.cols(); ++i) {
        if (i + xOffset < 0 || i + xOffset >= output.cols)
          continue;

        for (int j = 0; j < scan.rows(); ++j) {
          if (j + yOffset < 0 || j + yOffset >= output.rows)
            continue;
          if (scan(j, i) != 0) {
            if (!n.locked) {
              _output(yOffset + j, xOffset + i)[0] = 0;
              _output(yOffset + j, xOffset + i)[1] = 0;
              _output(yOffset + j, xOffset + i)[2] = 255;
            } else {
              _output(yOffset + j, xOffset + i)[0] = 255;
              _output(yOffset + j, xOffset + i)[1] = 0;
              _output(yOffset + j, xOffset + i)[2] = 0;
            }
          }
        }
      }

      for (int j = -10; j < 10; ++j) {
        uchar *dst = output.ptr<uchar>(j + n.y);
        for (int i = -10; i < 10; ++i) {
          const int col = 3 * (i + n.x);
          dst[col + 0] = 0;
          dst[col + 1] = 255;
          dst[col + 2] = 0;
        }
      }
    }

    cvNamedWindow("Preview", CV_WINDOW_NORMAL);
    cv::imshow("Preview", output);
    const int kc = cv::waitKey(0);

    if (kc == 27)
      break;
    else if (kc == 8)
      i = i > 0 ? i - 1 : i;
    else
      ++i;
  }

  cvNamedWindow("All", CV_WINDOW_NORMAL);
  cv::imshow("All", all);

  cvNamedWindow("Seen", CV_WINDOW_NORMAL);
  cv::imshow("Seen", seen);
  cv::waitKey(0);
}

const double sigCutoff = 0.05, scoreCutoff = 0.05;

place::edge
place::compare3D(const place::VoxelGrid &aPoint, const place::VoxelGrid &bPoint,
                 const place::VoxelGrid &aFree, const place::VoxelGrid &bFree,
                 const place::cube &aRect, const place::cube &bRect) {
  const int z = aRect.Z2 - aRect.Z1 + 1;
  const int Xrows = aRect.Y2 - aRect.Y1 + 1;
  const int Xcols = aRect.X2 - aRect.X1 + 1;

  double pointAgreement = 0.0, freeSpaceAgreementA = 0.0,
         freeSpaceAgreementB = 0.0;

  double totalPointA = 0, totalPointB = 0, totalCount = 0;

#pragma omp parallel for shared(                                               \
    aPoint, bPoint, aFree, bFree, aRect, bRect)                                \
        reduction(+ : pointAgreement, freeSpaceAgreementA,                     \
                  freeSpaceAgreementB, totalPointA, totalPointB, totalCount)
  for (int k = 0; k < z; ++k) {
    auto &Ap = aPoint.v[k + aRect.Z1];
    auto &Bp = bPoint.v[k + bRect.Z1];
    auto &Af = aFree.v[k + aRect.Z1];
    auto &Bf = bFree.v[k + bRect.Z1];

    if ((Ap.sum() == 0 && Af.sum() == 0) || (Bp.sum() == 0 && Bf.sum() == 0))
      continue;
    for (int i = 0; i < Xcols; ++i) {
      for (int j = 0; j < Xrows; ++j) {
        const Eigen::Vector3i APos(i + aRect.X1, j + aRect.Y1, aRect.Z1 + k);
        const Eigen::Vector3i BPos(i + bRect.X1, j + bRect.Y1, bRect.Z1 + k);

        if ((localGroup(Ap, APos[1], APos[0], 2) && Bp(BPos[1], BPos[0])) ||
            (Ap(APos[1], APos[0]) && localGroup(Bp, BPos[1], BPos[0], 2)))
          ++pointAgreement;

        if (Ap(APos[1], APos[0]) && Bf(BPos[1], BPos[0]))
          ++freeSpaceAgreementA;

        if (Bp(BPos[1], BPos[0]) && Af(APos[1], APos[0]))
          ++freeSpaceAgreementB;

        if (Ap(APos[1], APos[0])) {
          ++totalPointA;
          ++totalCount;
        }

        if (Bp(BPos[1], BPos[0])) {
          ++totalPointB;
          ++totalCount;
        }

        if (Af(APos[1], APos[0]))
          ++totalCount;
        if (Bf(BPos[1], BPos[0]))
          ++totalCount;
      }
    }
  }

  double averagePoint = (totalPointA + totalPointB) / 2.0;
  if (averagePoint == 0.0 || totalPointA == 0.0 || totalPointB == 0.0) {
    return place::edge();
  }

  const double weight =
      pointAgreement / averagePoint -
      (freeSpaceAgreementB / totalPointB + freeSpaceAgreementA / totalPointA);
  constexpr double precent = 0.5;
  totalCount /= 4.0;
  const double expectedCount = (aPoint.c + bPoint.c + aFree.c + bFree.c) / 4.0;
  const double significance =
      sigmoidWeight(totalCount, expectedCount * precent);

  return significance < sigCutoff || std::abs(weight) < scoreCutoff
             ? place::edge()
             : place::edge(pointAgreement / averagePoint,
                           freeSpaceAgreementA / totalPointA,
                           freeSpaceAgreementB / totalPointB, 0, weight,
                           significance);
}

inline place::VoxelGrid place::loadInVoxel(const std::string &name) {
  place::VoxelGrid tmp;
  place::loadInVoxel(name, tmp);
  return tmp;
}

inline void place::loadInVoxel(const std::string &name, place::VoxelGrid &dst) {
  std::ifstream in(name, std::ios::in | std::ios::binary);
  dst.loadFromFile(in);
  in.close();
}

typedef opengm::DiscreteSpace<> Space;
typedef opengm::ExplicitFunction<double> Function;
typedef opengm::GraphicalModel<double, opengm::Adder> Model;

static Model createModel(const Eigen::MatrixXE &adjacencyMatrix,
                         const std::vector<place::node> &nodes,
                         std::vector<size_t> &numberOfLabels,
                         size_t *&labelSize) {
  {
    size_t i = 0;
    const place::node *prevNode = &nodes[0];
    for (auto &n : nodes) {
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
  std::cout << "Number of labels: ";
  for (auto &i : numberOfLabels)
    std::cout << i << "_";
  std::cout << std::endl;

  // Construct the model
  labelSize = new size_t[numberOfLabels.size()];
  for (int i = 0; i < numberOfLabels.size(); ++i) {
    // The "+1" adds a special "no label" label
    labelSize[i] = numberOfLabels[i] + 1;
  }
  return Model(Space(labelSize, labelSize + numberOfLabels.size()));
}

static void populateModel(const Eigen::MatrixXE &adjacencyMatrix,
                          const std::vector<place::node> &nodes,
                          const std::vector<size_t> &numberOfLabels,
                          const size_t *labelSize, Model &gm) {
  const int numVars = numberOfLabels.size();
  // Add urnary terms
  for (size_t i = 0, offset = 0; i < numVars; ++i) {
    const size_t shape[] = {labelSize[i]};
    Function f(shape, shape + 1);
    for (int j = 0; j < numberOfLabels[i]; ++j) {
      const double weight = nodes[offset + j].getWeight();

      if (!Eigen::numext::isfinite(weight))
        std::cout << nodes[offset + j].color << std::endl;

      constexpr double urnaryWeight = 0.5;
      f(j) = urnaryWeight * (Eigen::numext::isfinite(weight) ? weight : 0);
    }
    f(shape[0] - 1) = -5;
    Model::FunctionIdentifier fid = gm.addFunction(f);
    const size_t factors[] = {i};
    gm.addFactor(fid, factors, factors + 1);

    offset += numberOfLabels[i];
  }

  // Add pairwise terms
  for (size_t i = 0, colOffset = 0, startRow = 0; i < numVars; ++i) {
    startRow += numberOfLabels[i];
    int currentRow = startRow;
    for (size_t j = i + 1; j < numVars; ++j) {

      const size_t shape[] = {labelSize[i], labelSize[j]};
      Function f(shape, shape + 2);
      for (int a = 0; a < numberOfLabels[i]; ++a) {
        for (int b = 0; b < numberOfLabels[j]; ++b) {
          const int row = nodes[currentRow + b].id;
          const int col = nodes[colOffset + a].id;

          f(a, b) = adjacencyMatrix(row, col).getWeight();
        }
      }

      for (int a = 0; a < shape[1]; ++a) {
        f(shape[0] - 1, a) = 0;
      }
      for (int a = 0; a < shape[0]; ++a) {
        f(a, shape[1] - 1) = 0;
      }

      Model::FunctionIdentifier fid = gm.addFunction(f);
      const size_t factors[] = {i, j};
      gm.addFactor(fid, factors, factors + 2);

      currentRow += numberOfLabels[j];
    }
    colOffset += numberOfLabels[i];
  }
}

void place::TRWSolver(const Eigen::MatrixXE &adjacencyMatrix,
                      const std::vector<place::node> &nodes,
                      std::vector<place::SelectedNode> &bestNodes) {
  std::cout << "Solving" << std::endl;
  // Figure out how many different variables and
  // labels per variable
  std::vector<size_t> numberOfLabels;
  size_t *labelSize;
  Model gm = createModel(adjacencyMatrix, nodes, numberOfLabels, labelSize);
  populateModel(adjacencyMatrix, nodes, numberOfLabels, labelSize, gm);

  constexpr size_t numIters = 60000;

  /*typedef opengm::TrbpUpdateRules<Model, opengm::Maximizer> UpdateRules;
  typedef opengm::MessagePassing<Model, opengm::Maximizer, UpdateRules,
                                 opengm::MaxDistance>
      Solver;
  constexpr double convergenceBound = 1e-7;
  constexpr double damping = 0.0;
  Solver::Parameter parameter(numIters, convergenceBound, damping);*/

  // set up the optimizer (TRW)
  typedef opengm::TRWSi_Parameter<Model> Parameter;
  typedef opengm::TRWSi<Model, opengm::Maximizer> Solver;
  Parameter parameter(numIters);
  Solver solver(gm, parameter);

  // Set up MQPBO
  /*typedef opengm::MQPBO<Model, opengm::Maximizer> Solver;
  Solver::Parameter parameter;
  parameter.useKovtunsMethod_ = false;
  parameter.rounds_ = 100;
  parameter.strongPersistency_ = true;
  parameter.permutationType_ = Solver::MINMARG;
  Solver solver(gm, parameter);*/

  Solver::EmptyVisitorType visitor;

  boost::timer::auto_cpu_timer *timer = new boost::timer::auto_cpu_timer;
  solver.infer(visitor);
  delete timer;

  std::cout << std::endl
            << solver.name() << " found a solution with a value of "
            << solver.value() << " bound " << solver.bound() << std::endl
            << std::endl;

  const int numVars = numberOfLabels.size();
  std::vector<Model::LabelType> labeling(numVars);
  solver.arg(labeling);

  std::cout << "Labels : ";
  for (auto &l : labeling)
    std::cout << l << "_";
  std::cout << std::endl;
  bestNodes.reserve(numVars);
  for (int i = 0, offset = 0; i < numVars; ++i) {
    const int index = offset + labeling[i];
    if (labeling[i] >= numberOfLabels[i]) {
      bestNodes.emplace_back(nodes[index - 1], -1000, labeling[i], false,
                             numberOfLabels[i]);
    } else {
      double agreement = 0;
      int count = 0;
      const int col = nodes[index].id;
      for (int j = 0, rowOffset = 0; j < numVars; ++j) {
        const int row = nodes[rowOffset + labeling[j]].id;
        const double w = adjacencyMatrix(row, col).getWeight();
        if (w != 0) {
          agreement += w;
          ++count;
        }
        rowOffset += numberOfLabels[j];
      }
      // agreement += 0.5 * nodes[index].w;
      agreement /= count ? count : 1;
      bestNodes.emplace_back(nodes[index], agreement, labeling[i], true,
                             numberOfLabels[i]);
    }
    offset += numberOfLabels[i];
  }

  delete[] labelSize;
  assert(bestNodes.size() == numVars);
}

bool place::reloadGraph(Eigen::MatrixXE &adjacencyMatrix, int level) {
  const std::string graphName =
      FLAGS_outputV2 + "graph" + std::to_string(level) + ".dat";
  std::ifstream in(graphName, std::ios::in | std::ios::binary);

  if (!in.is_open())
    return false;

  int cols, rows;
  in.read(reinterpret_cast<char *>(&rows), sizeof(rows));
  in.read(reinterpret_cast<char *>(&cols), sizeof(cols));

  adjacencyMatrix = Eigen::MatrixXE(rows, cols);
  in.read(reinterpret_cast<char *>(adjacencyMatrix.data()),
          sizeof(place::edge) * adjacencyMatrix.size());

  in.close();

  return true;
}

void place::saveGraph(Eigen::MatrixXE &adjacencyMatrix, int level) {
  const std::string graphName =
      FLAGS_outputV2 + "graph" + std::to_string(level) + ".dat";
  std::ofstream out(graphName, std::ios::out | std::ios::binary);

  if (!FLAGS_save)
    return;

  int rows = adjacencyMatrix.rows();
  int cols = adjacencyMatrix.cols();

  out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
  out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));

  out.write(reinterpret_cast<const char *>(adjacencyMatrix.data()),
            sizeof(place::edge) * adjacencyMatrix.size());

  out.close();
}

static double clampedNorm(double val, double center, double average,
                          double sigma) {
  constexpr double clampVal = 2.5;

  const double norm = (val - average) / sigma;
  return val > center ? std::max(0.0, std::min(clampVal, norm))
                      : std::max(-clampVal / 2.0, std::min(0.0, norm));
}

void place::normalizeWeights(Eigen::MatrixXE &adjacencyMatrix,
                             std::vector<place::node> &nodes) {
  std::vector<int> numberOfLabels;
  {
    int i = 0;
    const place::node *prevNode = &nodes[0];
    for (auto &n : nodes) {
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
  // numberOfLabels.resize(1);
  // numberOfLabels[1] = adjacencyMatrix.rows();
  const int numVars = numberOfLabels.size();

  for (int a = 0, rowOffset = 0; a < numVars; ++a) {
    double averageW = 0.0, averageP = 0;
    int countW = 0, countP = 0;
    for (int j = 0; j < numberOfLabels[a]; ++j) {
      for (int i = 0; i < adjacencyMatrix.cols(); ++i) {
        const double weight = adjacencyMatrix(j + rowOffset, i).w;
        const double pano = adjacencyMatrix(j + rowOffset, i).panoW;
        if (weight) {
          averageW += weight;
          ++countW;
        }
        if (pano) {
          averageP += pano;
          ++countP;
        }
      }
    }

    averageW /= countW;
    averageP /= countP;

    double sigmaW = 0, sigmaP = 0;
    for (int j = 0; j < numberOfLabels[a]; ++j) {
      for (int i = 0; i < adjacencyMatrix.cols(); ++i) {
        const double weight = adjacencyMatrix(j + rowOffset, i).w;
        const double pano = adjacencyMatrix(j + rowOffset, i).panoW;
        if (weight)
          sigmaW += (weight - averageW) * (weight - averageW);

        if (pano)
          sigmaP += (pano - averageP) * (pano - averageP);
      }
    }
    sigmaW /= countW - 1;
    sigmaW = sqrt(sigmaW);

    sigmaP /= countP - 1;
    sigmaP = sqrt(sigmaP);

    constexpr double geoCenter = 0.1, panoCenter = 0.3;
    averageW = std::max(geoCenter, averageW);
    averageP = std::max(panoCenter, averageP);

    for (int j = 0; j < numberOfLabels[a]; ++j) {
      for (int i = 0; i < adjacencyMatrix.cols(); ++i) {
        const double weight = adjacencyMatrix(j + rowOffset, i).w;
        const double pano = adjacencyMatrix(j + rowOffset, i).panoW;
        if (weight && countW > 1 && Eigen::numext::isfinite(sigmaW))
          adjacencyMatrix(j + rowOffset, i).w =
              clampedNorm(weight, geoCenter, averageW, sigmaW);

        if (pano && countP > 1 && Eigen::numext::isfinite(sigmaP))
          adjacencyMatrix(j + rowOffset, i).panoW =
              clampedNorm(pano, panoCenter, averageP, sigmaP);
      }
    }
    rowOffset += numberOfLabels[a];
  }
}
