#include "highOrder.h"
#include "placeScan_multiLabeling.h"
#include "placeScan_placeScan.h"

#include <math.h>
#include <omp.h>
#include <signal.h>
#include <time.h>

#include <algorithm>
#include <boost/progress.hpp>
#include <boost/timer/timer.hpp>
#include <fstream>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

DEFINE_bool(errosion, true,
            "This is used for scale finding only, don't touch!");

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
  place::getDirections();
  if (FLAGS_debugMode && false) {
    cv::Mat image = cv::imread(FLAGS_dmFolder + "R2/DUC_point_047.png", 0);
    std::vector<Eigen::Vector2i> tmp(4);
    std::vector<cv::Mat> toTrim = {image}, trimmed;
    place::trimScans(toTrim, trimmed, tmp);
    image = trimmed[0];

    const int xOffset = 2106 - image.cols / 2.0;
    const int yOffset = 1083 - image.rows / 2.0;
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
    cv::waitKey(0);
    return 0;
  }

  std::vector<std::string> pointFileNames, zerosFileNames, freeFileNames;

  place::parseFolders(pointFileNames, zerosFileNames, &freeFileNames);

  if (FLAGS_startNumber != -1)
    FLAGS_startIndex = numberToIndex(pointFileNames, FLAGS_startNumber);
  if (FLAGS_numScans == -1)
    FLAGS_numScans = pointFileNames.size() - FLAGS_startIndex;

  if (FLAGS_V1) {
    boost::progress_display *show_progress = nullptr;
    if (FLAGS_quietMode)
      show_progress = new boost::progress_display(FLAGS_numScans);
    std::vector<Eigen::SparseMatrix<double>> fpPyramid, erodedFpPyramid;
    std::vector<Eigen::MatrixXb> fpMasks;
    for (int i = FLAGS_startIndex;
         i < std::min(FLAGS_startIndex + FLAGS_numScans,
                      (int)pointFileNames.size());
         ++i) {
      const std::string scanName = pointFileNames[i];
      const std::string zerosFile = FLAGS_zerosFolder + zerosFileNames[i];
      const std::string maskName = freeFileNames[i];

      if (FLAGS_redo ||
          !place::reshowPlacement(scanName, zerosFile, FLAGS_outputV1)) {

        place::createFPPyramids(floorPlan, fpPyramid, erodedFpPyramid, fpMasks);
        place::analyzePlacement(fpPyramid, erodedFpPyramid, fpMasks, scanName,
                                zerosFile, maskName);
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
    labeler.displayGraph();
    labeler.solveTRW();
    labeler.saveFinal(0);
    labeler.displaySolution();
    // labeler.solveMIP();
    // labeler.saveFinal(1);
    while (false)
      labeler.displaySolution();
  }
  return 0;
}

void place::analyzePlacement(
    const std::vector<Eigen::SparseMatrix<double>> &fpPyramid,
    const std::vector<Eigen::SparseMatrix<double>> &erodedFpPyramid,
    const std::vector<Eigen::MatrixXb> &fpMasks, const std::string &scanName,
    const std::string &zerosFile, const std::string &maskName) {
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

  std::vector<std::vector<Eigen::SparseMatrix<double>>> eMaskPyramid{
      eMasksSpare};
  createPyramid(eMaskPyramid);
  eMasksSpare.clear();

  std::vector<std::vector<Eigen::SparseMatrix<double>>> rSSparsePyramid{
      rSSparse};
  createPyramid(rSSparsePyramid);
  rSSparse.clear();

  std::vector<std::vector<Eigen::SparseMatrix<double>>> erodedSparsePyramid{
      eScanSparse};
  createPyramid(erodedSparsePyramid);
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

  std::vector<Eigen::Vector4d> numPixelsUnderMask;
  findNumPixelsUnderMask(rSSparsePyramidTrimmed, eMaskPyramidTrimmedNS,
                         numPixelsUnderMask);

  if (FLAGS_debugMode || FLAGS_visulization)
    displayScanAndMask(rSSparsePyramidTrimmed, eMaskPyramidTrimmedNS);

  std::vector<Eigen::Vector3i> pointsToAnalyze;
  for (int k = 0; k < NUM_ROTS; ++k) {
    const int xStop = fpPyramid[FLAGS_numLevels].cols() -
                      rSSparsePyramidTrimmed[FLAGS_numLevels][k].cols();

    const int yStop = fpPyramid[FLAGS_numLevels].rows() -
                      rSSparsePyramidTrimmed[FLAGS_numLevels][k].rows();

    pointsToAnalyze.reserve(xStop * yStop + pointsToAnalyze.size());

    for (int i = 0; i < xStop; ++i)
      for (int j = 0; j < yStop; ++j)
        pointsToAnalyze.push_back(Eigen::Vector3i(i, j, k));
  }

  if (FLAGS_debugMode)
    loadInTruePlacement(scanName, zeroZero);

  std::vector<place::posInfo> scores;
  std::vector<const posInfo *> minima;

  constexpr double numRects = 25;
  const double divisor = std::sqrt(numRects);
  if (true && FLAGS_debugMode) {
    for (int k = FLAGS_numLevels; k >= 0; --k) {
      std::vector<Eigen::Vector3i> tmpPoints;
      std::vector<place::posInfo> trueScores;

      Eigen::Vector3i tmp(1878, 781, 2);
      tmp[0] /= pow(2, k);
      tmp[1] /= pow(2, k);

      for (int i = -20; i <= 20; ++i) {
        for (int j = -20; j <= 20; ++j) {
          tmpPoints.push_back(Eigen::Vector3i(tmp[0] + i, tmp[1] + j, tmp[2]));
        }
      }

      findPlacement(fpPyramid[k], rSSparsePyramidTrimmed[k], erodedFpPyramid[k],
                    erodedSparsePyramidTrimmed[k], eMaskPyramidTrimmedNS[k],
                    numPixelsUnderMask[k], fpMasks[k], tmpPoints, trueScores);

      std::vector<const place::posInfo *> tmpMin;

      const double exclusionX =
          std::min({rSSparsePyramidTrimmed[k][0].cols(),
                    rSSparsePyramidTrimmed[k][1].cols(),
                    rSSparsePyramidTrimmed[k][2].cols(),
                    rSSparsePyramidTrimmed[k][3].cols()}) /
          divisor;
      const double exclusionY =
          std::min({rSSparsePyramidTrimmed[k][0].rows(),
                    rSSparsePyramidTrimmed[k][1].rows(),
                    rSSparsePyramidTrimmed[k][2].rows(),
                    rSSparsePyramidTrimmed[k][3].rows()}) /
          divisor;
      const int newRows = std::ceil(fpPyramid[k].rows() / exclusionY);
      const int newCols = std::ceil(fpPyramid[k].cols() / exclusionX);
      place::exclusionMap maps(exclusionX, exclusionY, newRows, newCols);

      /*findLocalMinima(trueScores, 0.2, maps);
      findGlobalMinima(trueScores, maps, tmpMin);*/

      for (auto &t : trueScores)
        tmpMin.push_back(&t);

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

      place::displayOutput(fpPyramid[k], rSSparsePyramidTrimmed[k], tmpMin);
    }
  }

  /* Main work loop.  This takes care of doing all the method calls
  * needed to make pryamiding work.  Each method will take of making sure
  * the container passed to it for it's output is cleared */
  for (int k = FLAGS_numLevels; k >= 0; --k) {
    findPlacement(fpPyramid[k], rSSparsePyramidTrimmed[k], erodedFpPyramid[k],
                  erodedSparsePyramidTrimmed[k], eMaskPyramidTrimmedNS[k],
                  numPixelsUnderMask[k], fpMasks[k], pointsToAnalyze, scores);
    if (scores.size() == 0)
      return;

    const double exclusionX = std::min({rSSparsePyramidTrimmed[k][0].cols(),
                                        rSSparsePyramidTrimmed[k][1].cols(),
                                        rSSparsePyramidTrimmed[k][2].cols(),
                                        rSSparsePyramidTrimmed[k][3].cols()}) /
                              divisor;
    const double exclusionY = std::min({rSSparsePyramidTrimmed[k][0].rows(),
                                        rSSparsePyramidTrimmed[k][1].rows(),
                                        rSSparsePyramidTrimmed[k][2].rows(),
                                        rSSparsePyramidTrimmed[k][3].rows()}) /
                              divisor;
    const int newRows = std::ceil(fpPyramid[k].rows() / exclusionY);
    const int newCols = std::ceil(fpPyramid[k].cols() / exclusionX);
    place::exclusionMap maps(exclusionX, exclusionY, newRows, newCols);

    if (k == 0)
      findLocalMinima(scores, -0.5, maps);
    else
      findLocalMinima(scores, 0.5, maps);
    findGlobalMinima(scores, maps, minima);
    findPointsToAnalyzeV2(minima, pointsToAnalyze);

    if (false && FLAGS_debugMode) {
      Eigen::Vector3i tmp(110, 2919, 3);
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
        std::cout << "Lost at level: " << k << std::endl;

      std::sort(trueMin.begin(), trueMin.end(),
                [](const place::posInfo *a, const place::posInfo *b) {
                  return (a->score < b->score);
                });

      place::displayOutput(fpPyramid[k], rSSparsePyramidTrimmed[k], trueMin);
    }
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
    place::displayOutput(fpPyramid[0], rSSparsePyramidTrimmed[0], minima);
}

void place::findLocalMinima(const std::vector<place::posInfo> &scores,
                            const float bias, place::exclusionMap &maps) {
  double averageScore, sigScore;
  std::tie(averageScore, sigScore) =
      place::aveAndStdev(scores.begin(), scores.end(),
                         [](const place::posInfo &s) { return s.score; });

  if (!FLAGS_quietMode) {
    std::cout << "Average         Sigma" << std::endl;
    std::cout << averageScore << "         " << sigScore << std::endl;
  }

  const double cutOff = averageScore - bias * sigScore;
  for (int i = 0; i < scores.size(); ++i) {
    double lHS = i == 0 ? -1e10 : scores[i].score - scores[i - 1].score;
    double rHS =
        i == scores.size() - 1 ? 1e10 : scores[i + 1].score - scores[i].score;
    if (((lHS <= 0 && rHS > 0) || (lHS < 0 && rHS >= 0)) &&
        scores[i].score < cutOff) {
      const int mapRow = std::floor(scores[i].y / maps.exclusionY);
      const int mapCol = std::floor(scores[i].x / maps.exclusionX);
      const int rot = scores[i].rotation;

      if (!maps.maps[rot][mapCol * maps.rows + mapRow] ||
          scores[i].score < maps.maps[rot][mapCol * maps.rows + mapRow]->score)
        maps.maps[rot][mapCol * maps.rows + mapRow] = &scores[i];
    }
  }
}

/*Takes care of freeing maps */
void place::findGlobalMinima(const std::vector<place::posInfo> &scores,
                             place::exclusionMap &maps,
                             std::vector<const place::posInfo *> &minima) {
  double minScore = 5e30;
  for (auto &info : scores) {
    minScore = std::min(info.score, minScore);
  }

  if (!FLAGS_quietMode)
    std::cout << "Min score: " << minScore << std::endl;

  const int mapSize = maps.rows * maps.cols;

  for (int i = 0; i < scores.size(); ++i) {
    if (scores[i].score <= 1.05 * minScore) {
      const int mapRow = std::floor(scores[i].y / maps.exclusionY);
      const int mapCol = std::floor(scores[i].x / maps.exclusionX);
      const int rot = scores[i].rotation;

      if (!maps.maps[rot][mapCol * maps.rows + mapRow] ||
          scores[i].score < maps.maps[rot][mapCol * maps.rows + mapRow]->score)
        maps.maps[rot][mapCol * maps.rows + mapRow] = &scores[i];
    }
  }

  minima.clear();
  for (int i = 0; i < NUM_ROTS; ++i) {
    for (int j = 0; j < mapSize; ++j) {
      if (maps.maps[i][j])
        minima.push_back(maps.maps[i][j]);
    }
  }
}

template <typename MatType>
void place::createPyramid(std::vector<MatType> &pyramid) {
  typedef typename MatType::Scalar Scalar;
  typedef Eigen::Triplet<Scalar> TripType;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatNS;

  std::vector<TripType> tripletList;

  for (int i = 0; i < FLAGS_numLevels; ++i) {
    auto &currentLevel = pyramid[i];
    MatType newLevel(ceil(currentLevel.rows() / 2) + 1,
                     ceil(currentLevel.cols() / 2) + 1);

    MatNS currentLevelNS = MatNS(currentLevel);

    int j;
    for (j = 0; j < (currentLevel.rows() - 1); j += 2) {
      int k;
      for (k = 0; k < (currentLevel.cols() - 1); k += 2) {
        double maxV =
            std::max({currentLevelNS(j, k), currentLevelNS(j, k + 1),
                      currentLevelNS(j + 1, k), currentLevelNS(j + 1, k + 1)});
        tripletList.push_back(TripType(floor(j / 2), floor(k / 2), maxV));
      }
      for (; k < currentLevel.cols(); ++k) {
        tripletList.push_back(
            TripType(floor(j / 2), floor(k / 2), currentLevelNS(j, k)));
      }
    }

    for (; j < currentLevel.rows(); ++j) {
      int k;
      for (k = 0; k < (currentLevel.cols() - 1); k += 2) {
        double maxV = std::max(currentLevelNS(j, k), currentLevelNS(j, k + 1));
        tripletList.push_back(TripType(floor(j / 2), floor(k / 2), maxV));
      }
      for (; k < currentLevel.cols(); ++k) {
        tripletList.push_back(
            TripType(floor(j / 2), floor(k / 2), currentLevelNS(j, k)));
      }
    }
    newLevel.setFromTriplets(tripletList.begin(), tripletList.end());
    pyramid.push_back(newLevel);
    tripletList.clear();
  }

  if (FLAGS_visulization) {
    for (auto &level : pyramid) {
      cvNamedWindow("Preview", CV_WINDOW_NORMAL);
      cv::imshow("Preview", place::sparseToImage(level));
      cv::waitKey(0);
    }
  }
}

template <typename MatType>
void place::createPyramid(std::vector<std::vector<MatType>> &pyramid) {
  typedef typename MatType::Scalar Scalar;
  typedef Eigen::Triplet<Scalar> TripType;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatNS;

  std::vector<TripType> tripletList;

  for (int i = 0; i < FLAGS_numLevels; ++i) {
    std::vector<MatType> newLevel;
    for (auto &scan : pyramid[i]) {
      MatType newScan(ceil(scan.rows() / 2) + 1, ceil(scan.cols() / 2) + 1);
      MatNS scanNS = MatNS(scan);

      int j;
      for (j = 0; j < (scan.rows() - 1); j += 2) {
        int k;
        for (k = 0; k < (scan.cols() - 1); k += 2) {
          double maxV = std::max({scanNS(j, k), scanNS(j, k + 1),
                                  scanNS(j + 1, k), scanNS(j + 1, k + 1)});
          tripletList.push_back(TripType(floor(j / 2), floor(k / 2), maxV));
        }
        for (; k < scan.cols(); ++k) {
          tripletList.push_back(
              TripType(floor(j / 2), floor(k / 2), scanNS(j, k)));
        }
      }

      for (; j < scan.rows(); ++j) {
        int k;
        for (k = 0; k < (scan.cols() - 1); k += 2) {
          double maxV = std::max(scanNS(j, k), scanNS(j, k + 1));
          tripletList.push_back(TripType(floor(j / 2), floor(k / 2), maxV));
        }
        for (; k < scan.cols(); ++k) {
          tripletList.push_back(
              TripType(floor(j / 2), floor(k / 2), scanNS(j, k)));
        }
      }

      newScan.setFromTriplets(tripletList.begin(), tripletList.end());
      newLevel.push_back(newScan);
      tripletList.clear();
    }
    pyramid.push_back(newLevel);
  }

  if (FLAGS_visulization) {
    for (auto &level : pyramid) {
      for (auto &scan : level) {
        cvNamedWindow("Preview", CV_WINDOW_NORMAL);
        cv::imshow("Preview", place::sparseToImage(scan));
        cv::waitKey(0);
      }
    }
  }
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

void place::findPlacement(
    const Eigen::SparseMatrix<double> &fp,
    const std::vector<Eigen::SparseMatrix<double>> &scans,
    const Eigen::SparseMatrix<double> &fpE,
    const std::vector<Eigen::SparseMatrix<double>> &scansE,
    const std::vector<Eigen::MatrixXb> &masks,
    const Eigen::Vector4d &numPixelsUnderMask, const Eigen::MatrixXb &fpMask,
    const std::vector<Eigen::Vector3i> &points,
    std::vector<place::posInfo> &scores) {
  if (!FLAGS_quietMode)
    std::cout << "Start: " << points.size() << std::endl;

  scores.clear();
  scores.reserve(points.size());

#pragma omp parallel shared(scores)
  {
    std::vector<posInfo> privateScores;
#pragma omp for nowait schedule(static)
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
      for (int i = 0; i < diff.outerSize(); ++i) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(diff, i); it; ++it)
          if (it.value() > 0.0 && currentMask(it.row(), it.col()) != 0)
            scanFPsetDiff += it.value();
      }

      diff = currentFP - currentScanE;
      for (int i = 0; i < diff.outerSize(); ++i)
        for (Eigen::SparseMatrix<double>::InnerIterator it(diff, i); it; ++it)
          if (it.value() > 0.0 && currentMask(it.row(), it.col()) != 0)
            fpScanSetDiff += it.value();

      const double score =
          (1.5 * scanFPsetDiff / numPixelsUnderMask[scanIndex] +
           fpScanSetDiff / (numFPPixelsUM)) /
          2.5;

      posInfo tmp;
      tmp.x = point[0];
      tmp.y = point[1];
      tmp.rotation = scanIndex;
      tmp.score = score;
      tmp.scanFP = scanFPsetDiff;
      tmp.fpScan = fpScanSetDiff;
      tmp.scanPixels = numPixelsUnderMask[scanIndex];
      tmp.fpPixels = numFPPixelsUM;
      privateScores.push_back(tmp);
    }

#pragma omp for schedule(static) ordered
    for (int i = 0; i < omp_get_num_threads(); ++i) {
#pragma omp ordered
      scores.insert(scores.end(), privateScores.begin(), privateScores.end());
    }
  }
  scores.shrink_to_fit();
  if (!FLAGS_quietMode)
    std::cout << "Done: " << scores.size() << std::endl;
}

struct Perimeter {
  constexpr Perimeter() : coords() {
    int x = 0;
    int y = 0;
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < searchKernelSize; ++j) {
        coords[i * searchKernelSize + j][0] = x;
        coords[i * searchKernelSize + j][1] = y;
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
  }
  int coords[searchKernelSize * 4][2];
};

void place::findPointsToAnalyzeV2(
    const std::vector<const place::posInfo *> &minima,
    std::vector<Eigen::Vector3i> &pointsToAnalyze) {
  static constexpr auto perimeter = Perimeter();
  constexpr int range = searchKernelSize / 2;
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
      if (rand() % 2)
        pointsToAnalyze.push_back(Eigen::Vector3i(
            x + perimeter.coords[i][0], y + perimeter.coords[i][1], rotIndex));
      else
        pointsToAnalyze.push_back(
            Eigen::Vector3i(x + perimeter.coords[i + 1][0],
                            y + perimeter.coords[i + 1][1], rotIndex));
  }
  pointsToAnalyze.shrink_to_fit();
}

void place::createFPPyramids(
    const cv::Mat &floorPlan,
    std::vector<Eigen::SparseMatrix<double>> &fpPyramid,
    std::vector<Eigen::SparseMatrix<double>> &erodedFpPyramid,
    std::vector<Eigen::MatrixXb> &fpMasks) {

  static bool loaded = false;
  if (loaded)
    return;

  cv::Mat element = cv::getStructuringElement(
      cv::MORPH_RECT, cv::Size(errosionKernelSize, errosionKernelSize));
  cv::Mat fpEroded(floorPlan.rows, floorPlan.cols, CV_8UC1);
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
  createPyramid(fpPyramid);

  erodedFpPyramid.push_back(erodedFpTresh);
  createPyramid(erodedFpPyramid);

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
    std::vector<Eigen::Vector4d> &numPixelsUnderMask) {
  for (int i = 0; i < rSSparsePyramidTrimmed.size(); ++i) {
    Eigen::Vector4d numPixelsLevel = Eigen::Vector4d::Zero();
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
