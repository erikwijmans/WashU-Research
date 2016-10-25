#include "highOrder.h"
#include "placeScan_doorDetector.h"
#include "placeScan_multiLabeling.h"
#include "placeScan_placeScan.h"
#include "placeScan_placeScanHelper2.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_set>

#include <boost/progress.hpp>
#include <boost/timer/timer.hpp>

#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <math.h>
#include <omp.h>
#include <signal.h>
#include <time.h>

DEFINE_bool(errosion, true,
            "This is used for scale finding only, don't touch!");
DEFINE_bool(displayGraph, false, "Displays the graph");

static constexpr int errosionKernelSize = 5;
static_assert(errosionKernelSize % 2 == 1,
              "Errosion kernel size needs to be odd");
static constexpr int searchKernelSize = errosionKernelSize + 2;

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  prependDataPath();

  if (!FLAGS_V1 && !FLAGS_V2)
    FLAGS_V1 = FLAGS_V2 = true;

  if (FLAGS_debugMode) {
    FLAGS_save = false;
    FLAGS_redo = true;
    FLAGS_quietMode = false;
  }

  if (FLAGS_threads)
    omp_set_num_threads(FLAGS_threads);

  cv::Mat inFP = cv::imread(FLAGS_floorPlan, 0);
  if (!inFP.data) {
    std::cout << "Error reading floorPlan" << std::endl;
    exit(1);
  }

  place::removeMinimumConnectedComponents(inFP);
  cv::imwrite(FLAGS_dataPath + "/cleanedFloorPlan.png", inFP);

  const int newRows = inFP.rows * 1.1, newCols = inFP.cols * 1.1;
  const int dY = (newRows - inFP.rows) / 2, dX = (newCols - inFP.cols) / 2;
  floorPlan = cv::Mat(newRows, newCols, CV_8UC1, cv::Scalar::all(255));
  for (int i = 0; i < inFP.rows; ++i) {
    const uchar *src = inFP.ptr<uchar>(i);
    uchar *dst = floorPlan.ptr<uchar>(i + dY);
    for (int j = 0; j < inFP.cols; ++j) {
      dst[j + dX] = src[j];
    }
  }

  fpColor =
      cv::Mat(floorPlan.rows, floorPlan.cols, CV_8UC3, cv::Scalar::all(255));
  for (int i = 0; i < fpColor.rows; ++i) {
    uchar *dst = fpColor.ptr<uchar>(i);
    const uchar *src = floorPlan.ptr<uchar>(i);
    for (int j = 0; j < fpColor.cols; ++j) {
      if (src[j] != 255) {
        dst[j * 3] = 128;
        dst[j * 3 + 1] = 128;
        dst[j * 3 + 2] = 128;
      }
    }
  }

// cv::Mat fpWeights = place::getDirections();

#if 0
  if (FLAGS_debugMode) {
    cv::Mat image = cv::imread(FLAGS_dmFolder + "R2/DUC_point_032.png", 0);
    if (!image.data) {
      std::cout << "Could not load image" << std::endl;
      return 1;
    }
    std::vector<Eigen::Vector2i> tmp(4);
    std::vector<cv::Mat> toTrim = {image}, trimmed;
    place::trimScans(toTrim, trimmed, tmp);
    image = trimmed[0];

    const int xOffset = 3900;
    const int yOffset = 508;
    std::cout << xOffset << "  " << yOffset << std::endl;
    for (int i = 0; i < image.rows; ++i) {
      uchar *src = image.ptr<uchar>(i);
      uchar *dst = fpColor.ptr<uchar>(i + yOffset);
      for (int j = 0; j < image.cols; ++j) {
        if (src[j] != 255) {
          dst[3 * (j + xOffset) + 0] = 0;
          dst[3 * (j + xOffset) + 1] = 0;
          dst[3 * (j + xOffset) + 2] = 255;
        }
      }
    }
    cvNamedWindow("Preview", CV_WINDOW_NORMAL);
    cv::imshow("Preview", fpColor);
    std::cout << cv::waitKey(0) << std::endl;

    return 0;
  }
#endif

  std::vector<std::string> pointFileNames, zerosFileNames, freeFileNames,
      doorsNames;

  place::parseFolders(pointFileNames, zerosFileNames, &freeFileNames);
  parseFolder(FLAGS_doorsFolder + "floorplan", doorsNames);

  if (FLAGS_startNumber != -1)
    FLAGS_startIndex = numberToIndex(pointFileNames, FLAGS_startNumber);
  if (FLAGS_numScans == -1)
    FLAGS_numScans = pointFileNames.size() - FLAGS_startIndex;

  if (FLAGS_V1) {
    boost::progress_display *show_progress = nullptr;
    boost::timer::auto_cpu_timer timer;
    if (FLAGS_quietMode)
      show_progress = new boost::progress_display(FLAGS_numScans);

    std::vector<Eigen::SparseMatrix<double>> fpPyramid, erodedFpPyramid;
    std::vector<Eigen::MatrixXb> fpMasks;
    place::DoorDetector d;

    place::createFPPyramids(floorPlan, fpPyramid, erodedFpPyramid, fpMasks);
    d.run(fpPyramid, erodedFpPyramid, fpMasks);

    for (int i = FLAGS_startIndex;
         i < std::min(FLAGS_startIndex + FLAGS_numScans,
                      (int)pointFileNames.size());
         ++i) {
      const std::string scanName = pointFileNames[i];
      const std::string zerosFile = FLAGS_zerosFolder + zerosFileNames[i];
      const std::string maskName = freeFileNames[i];
      const std::string doorName =
          FLAGS_doorsFolder + "floorplan/" + doorsNames[i];

      if (FLAGS_redo ||
          !place::reshowPlacement(scanName, zerosFile, doorName, d,
                                  FLAGS_outputV1)) {

        place::createFPPyramids(floorPlan, fpPyramid, erodedFpPyramid, fpMasks);
        d.run(fpPyramid, erodedFpPyramid, fpMasks);
        place::analyzePlacement(fpPyramid, erodedFpPyramid, fpMasks, scanName,
                                zerosFile, maskName, doorName, d);
      }
      if (show_progress)
        ++(*show_progress);
    }
    if (show_progress)
      delete show_progress;
  }

  if (FLAGS_V2) {
    multi::Labeler labeler;
    labeler.weightEdges();
    if (FLAGS_displayGraph)
      labeler.displayGraph();
    labeler.solveTRW();
    labeler.saveFinal(0);

    if (!FLAGS_redo || FLAGS_previewOut)
      labeler.displaySolution();
    // labeler.solveMIP();
    // labeler.saveFinal(1);

    // labeler.displaySolution();
  }
  return 0;
}

void place::analyzePlacement(
    const std::vector<Eigen::SparseMatrix<double>> &fpPyramid,
    const std::vector<Eigen::SparseMatrix<double>> &erodedFpPyramid,
    const std::vector<Eigen::MatrixXb> &fpMasks, const std::string &scanName,
    const std::string &zerosFile, const std::string &maskName,
    const std::string &doorName, const place::DoorDetector &d) {
  boost::timer::auto_cpu_timer *timer = nullptr;

  if (!FLAGS_quietMode) {
    std::cout << scanName << std::endl;
    std::cout << maskName << std::endl;
    timer = new boost::timer::auto_cpu_timer;
  }

  cv::Mat element = cv::getStructuringElement(
      cv::MORPH_RECT, cv::Size(errosionKernelSize, errosionKernelSize));
  std::vector<cv::Mat> rotatedScans, masks;
  std::vector<Eigen::Vector2i> zeroZero;
  place::loadInScansAndMasks(scanName, zerosFile, maskName, rotatedScans, masks,
                             zeroZero);

  std::vector<Eigen::SparseMatrix<double>> rSSparse, eScanSparse, eMasksSpare;

  for (auto &scan : rotatedScans) {
    if (FLAGS_errosion) {
      cv::Mat dst;
      cv::erode(scan, dst, element);
      eScanSparse.push_back(scanToSparse(dst));
    } else
      eScanSparse.push_back(scanToSparse(scan));

    rSSparse.push_back(scanToSparse(scan));
  }

  for (auto &mask : masks)
    eMasksSpare.push_back(scanToSparse(mask));

  rotatedScans.clear();
  masks.clear();

  std::vector<std::vector<Eigen::SparseMatrix<double>>> eMaskPyramid(
      {eMasksSpare});
  createPyramid(eMaskPyramid, FLAGS_numLevels);
  eMasksSpare.clear();

  std::vector<std::vector<Eigen::SparseMatrix<double>>> rSSparsePyramid(
      {rSSparse});
  createPyramid(rSSparsePyramid, FLAGS_numLevels);
  rSSparse.clear();

  std::vector<std::vector<Eigen::SparseMatrix<double>>> erodedSparsePyramid(
      {eScanSparse});
  createPyramid(erodedSparsePyramid, FLAGS_numLevels);
  eScanSparse.clear();

  std::vector<std::vector<Eigen::SparseMatrix<double>>>
      erodedSparsePyramidTrimmed, rSSparsePyramidTrimmed, eMaskPyramidTrimmed;
  trimScanPryamids(rSSparsePyramid, rSSparsePyramidTrimmed, erodedSparsePyramid,
                   erodedSparsePyramidTrimmed, eMaskPyramid,
                   eMaskPyramidTrimmed, zeroZero);
  rSSparsePyramid.clear();
  erodedSparsePyramid.clear();
  eMaskPyramid.clear();

  std::vector<std::vector<Eigen::MatrixXb>> eMaskPyramidTrimmedNS;
  for (auto &level : eMaskPyramidTrimmed) {
    std::vector<Eigen::MatrixXb> tmp;
    for (auto &mask : level) {
      Eigen::MatrixXb tmpMat = Eigen::MatrixXb::Zero(mask.rows(), mask.cols());
      for (int i = 0; i < mask.outerSize(); ++i) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(mask, i); it; ++it) {
          if (it.value() != 0)
            tmpMat(it.row(), it.col()) = static_cast<char>(1);
        }
      }
      tmp.push_back(tmpMat);
    }
    eMaskPyramidTrimmedNS.push_back(tmp);
  }
  eMaskPyramidTrimmed.clear();

  std::vector<Eigen::VectorXd> numPixelsUnderMask;
  findNumPixelsUnderMask(rSSparsePyramidTrimmed, eMaskPyramidTrimmedNS,
                         numPixelsUnderMask);

  if (FLAGS_debugMode || FLAGS_visulization)
    displayScanAndMask(rSSparsePyramidTrimmed, eMaskPyramidTrimmedNS);

  std::vector<std::vector<std::vector<place::Door>>> doors(
      {loadInDoors(doorName, zeroZero)});

  for (int i = 0; i < FLAGS_numLevels; ++i) {
    doors.emplace_back(std::vector<std::vector<place::Door>>(doors[i].begin(),
                                                             doors[i].end()));
    for (auto &ds : doors[i + 1]) {
      for (auto &d : ds) {
        d.corner /= 2;
        d.w /= 2;
      }
    }
  }

  constexpr double numRects = 1024 * 1.5;

#if 0
  if (FLAGS_debugMode) {
    for (int k = 0; k >= 0; --k) {
      std::vector<Eigen::Vector3i> tmpPoints;
      std::vector<place::posInfo> trueScores;

      Eigen::Vector3i tmp(3900, 500, 2);
      tmp[0] /= pow(2, k);
      tmp[1] /= pow(2, k);

      std::cout << tmp << std::endl << std::endl;

      for (int i = -30; i <= 30; ++i) {
        for (int j = -30; j <= 30; ++j) {
          tmpPoints.push_back(Eigen::Vector3i(tmp[0] + i, tmp[1] + j, tmp[2]));
        }
      }

      findPlacement(fpPyramid[k], rSSparsePyramidTrimmed[k], erodedFpPyramid[k],
                    erodedSparsePyramidTrimmed[k], eMaskPyramidTrimmedNS[k],
                    numPixelsUnderMask[k], fpMasks[k], tmpPoints,
                    d.getResponse(k), doors[k], trueScores);

      std::vector<const place::posInfo *> tmpMin;

      const int scanRows = std::min({rSSparsePyramidTrimmed[k][0].rows(),
                                     rSSparsePyramidTrimmed[k][1].rows(),
                                     rSSparsePyramidTrimmed[k][2].rows(),
                                     rSSparsePyramidTrimmed[k][3].rows()});
      const int scanCols = std::min({rSSparsePyramidTrimmed[k][0].cols(),
                                     rSSparsePyramidTrimmed[k][1].cols(),
                                     rSSparsePyramidTrimmed[k][2].cols(),
                                     rSSparsePyramidTrimmed[k][3].cols()});

      const double exclusion =
          (scanRows + scanCols) / (2.0 * std::sqrt(numRects));

      const int cols = fpPyramid[k].cols();
      const int rows = fpPyramid[k].rows();
      place::ExclusionMap maps(exclusion, rows, cols);

      findLocalMinima(trueScores, 1.0, maps, tmpMin);

      bool found = false;
      for (auto &min : tmpMin)
        if (min->x >= (tmp[0] - 1) && min->x <= (tmp[0] + 1) &&
            (min->y >= tmp[1] - 1) && min->y <= (tmp[1] + 1) &&
            min->rotation == tmp[2])
          found = true;

      if (!found)
        std::cout << "Lost at level: " << k << std::endl;

      std::sort(tmpMin.begin(), tmpMin.end(),
                [](const place::posInfo *a, const place::posInfo *b) {
                  return (a->score < b->score);
                });

      place::displayOutput(fpPyramid[k], rSSparsePyramidTrimmed[k],
                           d.getResponse(k), doors[k], tmpMin);
    }
  }
#endif

  std::vector<place::posInfo> scores;
  std::vector<const posInfo *> minima;
  std::vector<Eigen::Vector3i> pointsToAnalyze;
  /*
  * Initializ pointsToAnalyze with every point
  */
  pointsToAnalyze.reserve(NUM_ROTS * fpPyramid[FLAGS_numLevels].cols() *
                          fpPyramid[FLAGS_numLevels].rows());
  for (int k = 0; k < NUM_ROTS; ++k) {
    const int xStop = fpPyramid[FLAGS_numLevels].cols() -
                      rSSparsePyramidTrimmed[FLAGS_numLevels][k].cols();

    const int yStop = fpPyramid[FLAGS_numLevels].rows() -
                      rSSparsePyramidTrimmed[FLAGS_numLevels][k].rows();

    for (int i = 0; i < xStop; ++i)
      for (int j = 0; j < yStop; ++j)
        pointsToAnalyze.push_back(Eigen::Vector3i(i, j, k));
  }
  pointsToAnalyze.shrink_to_fit();

  /*
  * Main work loop.  This takes care of doing all the method calls
  * needed to make pryamiding work.  Each method will take of making sure
  * the container passed to it for it's output is cleared
  */
  for (int k = FLAGS_numLevels; k >= 0; --k) {
    findPlacement(fpPyramid[k], rSSparsePyramidTrimmed[k], erodedFpPyramid[k],
                  erodedSparsePyramidTrimmed[k], eMaskPyramidTrimmedNS[k],
                  numPixelsUnderMask[k], fpMasks[k], pointsToAnalyze,
                  d.getResponse(k), doors[k], scores);
    if (scores.size() == 0)
      return;

    const int scanRows = std::min({rSSparsePyramidTrimmed[k][0].rows(),
                                   rSSparsePyramidTrimmed[k][1].rows(),
                                   rSSparsePyramidTrimmed[k][2].rows(),
                                   rSSparsePyramidTrimmed[k][3].rows()});
    const int scanCols = std::min({rSSparsePyramidTrimmed[k][0].cols(),
                                   rSSparsePyramidTrimmed[k][1].cols(),
                                   rSSparsePyramidTrimmed[k][2].cols(),
                                   rSSparsePyramidTrimmed[k][3].cols()});

    const double exclusion =
        (scanRows + scanCols) / (2.0 * std::sqrt(numRects));

    const int cols = fpPyramid[k].cols();
    const int rows = fpPyramid[k].rows();
    place::ExclusionMap maps(exclusion, rows, cols);

    if (k == 0)
      findLocalMinima(scores, -0.5, maps, minima);
    else
      findLocalMinima(scores, 1.0, maps, minima);

    findPointsToAnalyzeV2(minima, pointsToAnalyze);

#if 0
    if (FLAGS_debugMode) {
      /*Eigen::Vector3i tmp(3197, 1241, 2);
      tmp[0] /= pow(2, k);
      tmp[1] /= pow(2, k);

      constexpr int offset = 1;
      std::vector<const place::posInfo *> trueMin;
      bool found = false;
      for (auto &min : minima) {
        if (min->x >= tmp[0] - offset && min->x <= tmp[0] + offset &&
            min->y >= tmp[1] - offset && min->y <= tmp[1] + offset &&
            min->rotation == tmp[2]) {
          trueMin.push_back(min);
          found = true;
        }
      }

      if (!found)
        std::cout << "Lost at level: " << k << std::endl;*/

      std::vector<const place::posInfo *> trueMin(minima.begin(), minima.end());
      std::sort(trueMin.begin(), trueMin.end(),
                [](const place::posInfo *a, const place::posInfo *b) {
                  return (a->score < b->score);
                });

      place::displayOutput(fpPyramid[k], rSSparsePyramidTrimmed[k],
                           d.getResponse(k), doors[k], trueMin);
    }
#endif
  }

  std::sort(minima.begin(), minima.end(),
            [](const place::posInfo *a, const place::posInfo *b) {
              return (a->score < b->score);
            });

  if (timer)
    delete timer;

  if (FLAGS_save) {
    const std::string placementName =
        FLAGS_outputV1 + scanName.substr(scanName.find("_") - 3, 3) +
        "_placement_" + scanName.substr(scanName.find(".") - 3, 3) + ".txt";
    savePlacement(minima, placementName, zeroZero);
  }

  if (FLAGS_visulization || FLAGS_previewOut)
    place::displayOutput(fpPyramid[0], rSSparsePyramidTrimmed[0],
                         d.getResponse(0), doors[0], minima);
}

void place::findLocalMinima(const std::vector<place::posInfo> &scores,
                            const float bias, place::ExclusionMap &maps,
                            std::vector<const place::posInfo *> &minima) {
  for (auto &s : scores)
    maps[s.rotation](s.y, s.x) = &s;

  constexpr int minMinia = 100;
  constexpr double FF = 1.00, hardFactor = 1.0 / 40.0;

  const double hardY = maps.exclusionSize * hardFactor,
               hardX = maps.exclusionSize * hardFactor;

  double averageScore, sigScore;
  std::tie(averageScore, sigScore) =
      place::aveAndStdev(scores.begin(), scores.end(),
                         [](const place::posInfo &s) { return s.score; });

  if (!FLAGS_quietMode) {
    std::cout << "Average         Sigma" << std::endl;
    std::cout << averageScore << "         " << sigScore << std::endl;
  }

  for (int k = 0; k < 100; ++k) {
    minima.clear();
    const double cutOff = averageScore - (bias * sigScore) + k * 0.01;

    for (auto &s : scores) {
      const int rot = s.rotation;
      if (s.score < cutOff) {

        bool isMin = true;
        for (int j = -maps.exclusionSize / 2;
             j <= maps.exclusionSize / 2 && isMin; ++j) {
          if (s.y + j < 0 || s.y + j >= maps.rows)
            continue;

          for (int i = -maps.exclusionSize / 2;
               i <= maps.exclusionSize / 2 && isMin; ++i) {
            if (s.x + i < 0 || s.x + i >= maps.cols)
              continue;

            if (maps[rot](s.y + j, s.x + i) &&
                maps[rot](s.y + j, s.x + i)->score *
                        ((std::abs(i) < hardFactor && std::abs(j) < hardFactor)
                             ? 1.0
                             : FF) <
                    s.score)
              isMin = false;
          }
        }

        if (isMin)
          minima.emplace_back(&s);
      }
    }

    if (minima.size() >= minMinia)
      break;
  }

  std::unordered_set<place::posInfo> duplicates;

  minima.erase(std::remove_if(minima.begin(), minima.end(),
                              [&duplicates](auto *p) {
                                return !duplicates.insert(*p).second;
                              }),
               minima.end());
}

void place::trimScanPryamids(
    const std::vector<std::vector<Eigen::SparseMatrix<double>>>
        &rSSparsePyramid,
    std::vector<std::vector<Eigen::SparseMatrix<double>>>
        &rSSparsePyramidTrimmed,
    const std::vector<std::vector<Eigen::SparseMatrix<double>>>
        &erodedSparsePyramid,
    std::vector<std::vector<Eigen::SparseMatrix<double>>>
        &erodedSparsePyramidTrimmed,
    const std::vector<std::vector<Eigen::SparseMatrix<double>>> &eMaskPyramid,
    std::vector<std::vector<Eigen::SparseMatrix<double>>> &eMaskPyramidTrimmed,
    std::vector<Eigen::Vector2i> &zeroZero) {

  std::vector<Eigen::Triplet<double>> tripletList;
  for (int level = 0; level < rSSparsePyramid.size(); ++level) {
    std::vector<Eigen::SparseMatrix<double>> levelTrimmed, erodedLevelTrimed,
        maskLevelTrimmed;
    for (int i = 0; i < rSSparsePyramid[level].size(); ++i) {
      auto *scan = &rSSparsePyramid[level][i];
      auto *erodedScan = &erodedSparsePyramid[level][i];
      auto *mask = &eMaskPyramid[level][i];

      Eigen::SparseMatrix<double> scanThreshHolded(scan->rows(), scan->cols());
      Eigen::SparseMatrix<double> eScanThreshHolded(erodedScan->rows(),
                                                    erodedScan->cols());

      for (int k = 0; k < scan->outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(*scan, k); it; ++it)
          if (it.value() > 0.75)
            tripletList.push_back(
                Eigen::Triplet<double>(it.row(), it.col(), it.value()));

      scanThreshHolded.setFromTriplets(tripletList.begin(), tripletList.end());
      tripletList.clear();

      for (int k = 0; k < erodedScan->outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(*erodedScan, k); it;
             ++it)
          if (it.value() > 0.75)
            tripletList.push_back(
                Eigen::Triplet<double>(it.row(), it.col(), it.value()));

      eScanThreshHolded.setFromTriplets(tripletList.begin(), tripletList.end());
      tripletList.clear();

      int minRow, minCol, maxRow, maxCol;
      maxRow = maxCol = 0;
      minRow = scanThreshHolded.rows();
      minCol = scanThreshHolded.cols();

      for (int k = 0; k < scanThreshHolded.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(scanThreshHolded, k);
             it; ++it) {
          if (it.value() != 0) {
            maxRow = std::max(maxRow, it.row());
            minRow = std::min(minRow, it.row());

            maxCol = std::max(maxCol, it.col());
            minCol = std::min(minCol, it.col());
          }
        }
      }

      Eigen::SparseMatrix<double> trimmed = scanThreshHolded.block(
          minRow, minCol, maxRow - minRow + 1, maxCol - minCol + 1);
      trimmed.prune(1.0);
      Eigen::SparseMatrix<double> erodedTrimmed = eScanThreshHolded.block(
          minRow, minCol, maxRow - minRow + 1, maxCol - minCol + 1);
      erodedTrimmed.prune(1.0);
      Eigen::SparseMatrix<double> maskTrimmed =
          mask->block(minRow, minCol, maxRow - minRow + 1, maxCol - minCol + 1);
      maskTrimmed.prune(1.0);

      if (level == 0) {
        zeroZero[i][0] -= minCol;
        zeroZero[i][1] -= minRow;
      }

      levelTrimmed.push_back(trimmed);
      erodedLevelTrimed.push_back(erodedTrimmed);
      maskLevelTrimmed.push_back(maskTrimmed);
    }

    rSSparsePyramidTrimmed.push_back(levelTrimmed);
    erodedSparsePyramidTrimmed.push_back(erodedLevelTrimed);
    eMaskPyramidTrimmed.push_back(maskLevelTrimmed);
  }

  if (FLAGS_visulization) {
    for (int i = 0; i < rSSparsePyramidTrimmed.size(); ++i) {
      for (int j = 0; j < rSSparsePyramidTrimmed[i].size(); ++j) {
        cvNamedWindow("Image", CV_WINDOW_NORMAL);
        cv::imshow("Image", sparseToImage(rSSparsePyramidTrimmed[i][j]));

        cvNamedWindow("Mask", CV_WINDOW_NORMAL);
        cv::imshow("Mask", sparseToImage(eMaskPyramidTrimmed[i][j]));

        cv::waitKey(0);
      }
    }
  }
}

static std::tuple<double, double>
computeDoorIntersections(const Eigen::MatrixXb &fpDoors,
                         const std::vector<std::vector<place::Door>> &pcDoors,
                         const Eigen::Vector3i &point) {
  auto &doors = pcDoors[point[2]];

  double unexplained = 0, total = 0;
  for (auto &d : doors) {
    total += std::ceil(d.w);
    unexplained += std::ceil(d.w);

    for (int x = 0; x < std::ceil(d.w); ++x) {
      Eigen::Vector3d index =
          (d.corner + x * d.xAxis + point.cast<double>()).unaryExpr([](auto v) {
            return std::round(v);
          });
      char val;
      if ((val = place::localGroup(fpDoors, index[1], index[0], 2)))
        unexplained -= val / 2.0;
    }
  }

  return std::make_tuple(unexplained, std::max(1.0, total));
}

void place::findPlacement(
    const Eigen::SparseMatrix<double> &fp,
    const std::vector<Eigen::SparseMatrix<double>> &scans,
    const Eigen::SparseMatrix<double> &fpE,
    const std::vector<Eigen::SparseMatrix<double>> &scansE,
    const std::vector<Eigen::MatrixXb> &masks,
    const Eigen::VectorXd &numPixelsUnderMask, const Eigen::MatrixXb &fpMask,
    const std::vector<Eigen::Vector3i> &points, const Eigen::MatrixXb &fpDoors,
    const std::vector<std::vector<place::Door>> &pcDoors,
    std::vector<place::posInfo> &scores) {
  if (!FLAGS_quietMode)
    std::cout << "Start: " << points.size() << std::endl;

  scores.resize(points.size());
  for (auto &s : scores)
    s.score = -1;

#pragma omp parallel for schedule(static) shared(scores)
  for (int i = 0; i < points.size(); ++i) {
    const Eigen::Vector3i &point = points[i];
    const int scanIndex = point[2];
    const int xStop = fp.cols() - scans[scanIndex].cols();
    const int yStop = fp.rows() - scans[scanIndex].rows();

    if (point[0] < 0 || point[0] >= xStop)
      continue;
    if (point[1] < 0 || point[1] >= yStop)
      continue;

    auto &currentScan = scans[scanIndex];
    auto &currentScanE = scansE[scanIndex];
    auto &currentMask = masks[scanIndex];

    int numPixelsInside = 0;
    int numPixelsInMask = 0;
    for (int i = 0; i < currentScan.cols(); ++i) {
      for (int j = 0; j < currentScan.rows(); ++j) {
        if (fpMask(j + point[1], i + point[0]) && currentMask(j, i))
          ++numPixelsInside;
        if (currentMask(j, i))
          ++numPixelsInMask;
      }
    }

    if (numPixelsInside < 0.7 * numPixelsInMask)
      continue;

    Eigen::SparseMatrix<double> currentFP =
        fp.block(point[1], point[0], currentScan.rows(), currentScan.cols());
    currentFP.prune(1.0);

    double numFPPixelsUM = 0.0;
    for (int i = 0; i < currentFP.outerSize(); ++i)
      for (Eigen::SparseMatrix<double>::InnerIterator it(currentFP, i); it;
           ++it)
        if (currentMask(it.row(), it.col()))
          numFPPixelsUM += it.value();

    if (numFPPixelsUM < 0.6 * numPixelsUnderMask[scanIndex])
      continue;

    double scanFPsetDiff = 0;
    double fpScanSetDiff = 0;

    Eigen::SparseMatrix<double> currentFPE =
        fpE.block(point[1], point[0], currentScan.rows(), currentScan.cols());
    currentFPE.prune(1.0);

    Eigen::SparseMatrix<double> diff = currentScan - currentFPE;
    for (int i = 0; i < diff.outerSize(); ++i)
      for (Eigen::SparseMatrix<double>::InnerIterator it(diff, i); it; ++it)
        if (it.value() > 0.0 && currentMask(it.row(), it.col()) != 0)
          scanFPsetDiff += it.value();

    diff = currentFP - currentScanE;
    for (int i = 0; i < diff.outerSize(); ++i)
      for (Eigen::SparseMatrix<double>::InnerIterator it(diff, i); it; ++it)
        if (it.value() > 0.0 && currentMask(it.row(), it.col()) != 0)
          fpScanSetDiff += it.value();

    double doorUxp, doorCount;
    std::tie(doorUxp, doorCount) =
        computeDoorIntersections(fpDoors, pcDoors, point);
    const double doorScore = doorUxp / doorCount;
    const double scanScore = scanFPsetDiff / numPixelsUnderMask[scanIndex];
    const double fpScore = fpScanSetDiff / numFPPixelsUM;
    const double score = (1.5 * scanScore + fpScore + doorScore) / 3.5;

    if (!Eigen::numext::isfinite(score))
      continue;

    posInfo tmp;
    tmp.x = point[0];
    tmp.y = point[1];
    tmp.rotation = scanIndex;
    tmp.score = score;
    tmp.scanFP = scanFPsetDiff;
    tmp.fpScan = fpScanSetDiff;
    tmp.scanPixels = numPixelsUnderMask[scanIndex];
    tmp.fpPixels = numFPPixelsUM;
    tmp.doorUxp = doorUxp;
    tmp.doorCount = doorCount;
    scores[i] = tmp;
  }

  scores.erase(std::remove_if(scores.begin(), scores.end(),
                              [](const place::posInfo &s) {
                                return std::abs(s.score + 1) < 1e-12;
                              }),
               scores.end());

  if (!FLAGS_quietMode)
    std::cout << "Done: " << scores.size() << std::endl;
}

template <int w> class Perimeter {
public:
  constexpr Perimeter() : coords() {
    constexpr int range = w / 2;
    int x = -range;
    int y = -range;
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < w; ++j) {
        coords[i * w + j][0] = x;
        coords[i * w + j][1] = y;
        switch (i) {
        case 0:
          ++x;
          break;
        case 1:
          ++y;
          break;
        case 2:
          --x;
          break;
        case 3:
          --y;
          break;
        }
      }
    }
  };
  constexpr int operator()(int x, int y) const { return coords[x][y]; };

private:
  int coords[w * 4][2];
};

void place::findPointsToAnalyzeV2(
    const std::vector<const place::posInfo *> &minima,
    std::vector<Eigen::Vector3i> &pointsToAnalyze) {
  constexpr Perimeter<searchKernelSize> perimeter;
  constexpr int range = searchKernelSize / 2;

  static std::random_device seed;
  static std::mt19937_64 gen(seed());
  static std::uniform_int_distribution<int> dist(0, 1);

  pointsToAnalyze.clear();
  pointsToAnalyze.reserve(minima.size() * (searchKernelSize * searchKernelSize -
                                           2 * searchKernelSize + 4));

  for (auto min : minima) {
    const int x = 2 * min->x;
    const int y = 2 * min->y;
    const int rotIndex = min->rotation;
    for (int i = -(range - 1); i <= (range - 1); ++i)
      for (int j = -(range - 1); j <= (range - 1); ++j)
        pointsToAnalyze.push_back(Eigen::Vector3i(x + i, y + j, rotIndex));

    for (int i = 0; i + 1 < searchKernelSize * 4; i += 2)
      if (dist(gen))
        pointsToAnalyze.push_back(Eigen::Vector3i(
            x + perimeter(i, 0), y + perimeter(i, 1), rotIndex));
      else
        pointsToAnalyze.push_back(Eigen::Vector3i(
            x + perimeter(i + 1, 0), y + perimeter(i + 1, 1), rotIndex));
  }
  pointsToAnalyze.shrink_to_fit();
}

void place::createFPPyramids(
    const cv::Mat &floorPlan,
    std::vector<Eigen::SparseMatrix<double>> &fpPyramid,
    std::vector<Eigen::SparseMatrix<double>> &erodedFpPyramid,
    std::vector<Eigen::MatrixXb> &fpMasks, bool reload) {

  /*for (int j = 0; j < floorPlan.rows; ++j) {
    const uchar *w = fpWeights.ptr<uchar>(j);
    uchar *dst = floorPlan.ptr<uchar>(j);
    for (int i = 0; i < floorPlan.cols; ++i) {
      if (dst[i] != 255)
        dst[i] = std::max(w[i], dst[i]);
    }
  }*/

  static bool loaded = false;
  if (loaded & !reload)
    return;

  cv::Mat element = cv::getStructuringElement(
      cv::MORPH_RECT, cv::Size(errosionKernelSize, errosionKernelSize));
  cv::Mat fpEroded(floorPlan.size(), CV_8UC1);
  if (FLAGS_errosion)
    cv::erode(floorPlan, fpEroded, element);
  else
    floorPlan.copyTo(fpEroded);

  Eigen::SparseMatrix<double> fpSparse, erodedFpSparse;
  place::scanToSparse(floorPlan, fpSparse);
  place::scanToSparse(fpEroded, erodedFpSparse);

  Eigen::SparseMatrix<double> fpTresh(fpSparse.rows(), fpSparse.cols()),
      erodedFpTresh(fpSparse.rows(), fpSparse.cols());
  std::vector<Eigen::Triplet<double>> fpTrip, erodedTrip;

  for (int i = 0; i < fpSparse.outerSize(); ++i)
    for (Eigen::SparseMatrix<double>::InnerIterator it(fpSparse, i); it; ++it)
      if (it.value() > 0.4)
        fpTrip.push_back(
            Eigen::Triplet<double>(it.row(), it.col(), it.value()));

  for (int i = 0; i < erodedFpSparse.outerSize(); ++i)
    for (Eigen::SparseMatrix<double>::InnerIterator it(erodedFpSparse, i); it;
         ++it)
      if (it.value() > 0.4)
        erodedTrip.push_back(
            Eigen::Triplet<double>(it.row(), it.col(), it.value()));

  fpTresh.setFromTriplets(fpTrip.begin(), fpTrip.end());
  erodedFpTresh.setFromTriplets(erodedTrip.begin(), erodedTrip.end());

  fpPyramid.push_back(fpTresh);
  createPyramid(fpPyramid, FLAGS_numLevels);

  erodedFpPyramid.push_back(erodedFpTresh);
  createPyramid(erodedFpPyramid, FLAGS_numLevels);

  for (auto &level : fpPyramid) {
    Eigen::MatrixXd levelNS = Eigen::MatrixXd(level);
    Eigen::MatrixXb mask =
        Eigen::MatrixXb::Zero(levelNS.rows(), levelNS.cols());

    for (int i = 0; i < levelNS.cols(); ++i) {
      int minRow = levelNS.rows();
      int maxRow = 0;
      for (int j = 0; j < levelNS.rows(); ++j) {
        if (levelNS(j, i)) {
          minRow = std::min(minRow, j);
          maxRow = std::max(maxRow, j);
        }
      }
      for (int j = minRow; j < maxRow + 1; ++j)
        mask(j, i) = 1;
    }

    for (int j = 0; j < levelNS.rows(); ++j) {
      int minCol = levelNS.cols();
      int maxCol = 0;
      for (int i = 0; i < levelNS.cols(); ++i) {
        if (levelNS(j, i)) {
          minCol = std::min(i, minCol);
          maxCol = std::max(i, maxCol);
        }
      }
      for (int i = 0; i < levelNS.cols(); ++i)
        if (i < minCol || i > maxCol)
          mask(j, i) = 0;
    }
    fpMasks.push_back(mask);

    if (FLAGS_visulization) {
      cv::Mat dst(mask.rows(), mask.cols(), CV_8UC1, cv::Scalar::all(255));
      for (int i = 0; i < mask.cols(); ++i) {
        for (int j = 0; j < mask.rows(); ++j) {
          if (mask(j, i))
            dst.at<uchar>(j, i) = 0;
        }
      }

      cvNamedWindow("Preview", CV_WINDOW_NORMAL);
      cv::imshow("Preview", dst);
      cv::waitKey(0);
    }
  }

  loaded = true;
}

void place::findNumPixelsUnderMask(
    const std::vector<std::vector<Eigen::SparseMatrix<double>>>
        &rSSparsePyramidTrimmed,
    const std::vector<std::vector<Eigen::MatrixXb>> &eMaskPyramidTrimmedNS,
    std::vector<Eigen::VectorXd> &numPixelsUnderMask) {

  for (int i = 0; i < rSSparsePyramidTrimmed.size(); ++i) {
    Eigen::VectorXd numPixelsLevel =
        Eigen::VectorXd::Zero(rSSparsePyramidTrimmed[i].size());
    for (int j = 0; j < rSSparsePyramidTrimmed[i].size(); ++j) {
      const Eigen::MatrixXb &currentMask = eMaskPyramidTrimmedNS[i][j];
      const Eigen::SparseMatrix<double> &currentScan =
          rSSparsePyramidTrimmed[i][j];

      for (int a = 0; a < currentScan.outerSize(); ++a) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(currentScan, a); it;
             ++it) {
          if (currentMask(it.row(), it.col()) != 0)
            numPixelsLevel[j] += it.value();
        }
      }
    }
    numPixelsUnderMask.push_back(numPixelsLevel);
  }
}
