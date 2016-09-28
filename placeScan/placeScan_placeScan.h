#ifndef PLACESCAN_PLACE_SCAN_H_
#define PLACESCAN_PLACE_SCAN_H_

#include "placeScan_placeScanHelper.h"
#include "placeScan_placeScanHelper2.h"

#include <scan_typedefs.hpp>

namespace place {
void analyzePlacement(
    const std::vector<Eigen::SparseMatrix<double>> &fpPyramid,
    const std::vector<Eigen::SparseMatrix<double>> &erodedFpPyramid,
    const std::vector<Eigen::MatrixXb> &fpMasks, const std::string &scanName,
    const std::string &zerosFile, const std::string &maskName);

void findLocalMinima(const std::vector<place::posInfo> &scores,
                     const float bias, place::ExclusionMap &maps,
                     std::vector<const place::posInfo *> &minima);

void trimScanPryamids(
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
    std::vector<Eigen::Vector2i> &zeroZero);

void findPlacement(const Eigen::SparseMatrix<double> &fp,
                   const std::vector<Eigen::SparseMatrix<double>> &scans,
                   const Eigen::SparseMatrix<double> &fpE,
                   const std::vector<Eigen::SparseMatrix<double>> &scansE,
                   const std::vector<Eigen::MatrixXb> &masks,
                   const Eigen::VectorXd &numPixelsUnderMask,
                   const Eigen::MatrixXb &fpMask,
                   const std::vector<Eigen::Vector3i> &points,
                   std::vector<place::posInfo> &scores);

void findPointsToAnalyze(const std::vector<posInfo> &scores,
                         const std::vector<int> &localMinima,
                         std::vector<Eigen::Vector3i> &pointsToAnalyze);

void findPointsToAnalyzeV2(const std::vector<const place::posInfo *> &minima,
                           std::vector<Eigen::Vector3i> &pointsToAnalyze);

Eigen::MatrixXd distanceTransform(const Eigen::SparseMatrix<double> &image);

void createFPPyramids(const cv::Mat &floorPlan,
                      std::vector<Eigen::SparseMatrix<double>> &fpPyramid,
                      std::vector<Eigen::SparseMatrix<double>> &erodedFpPyramid,
                      std::vector<Eigen::MatrixXb> &fpMasks,
                      bool reload = false);

void findNumPixelsUnderMask(
    const std::vector<std::vector<Eigen::SparseMatrix<double>>>
        &rSSparsePyramidTrimmed,
    const std::vector<std::vector<Eigen::MatrixXb>> &eMaskPyramidTrimmedNS,
    std::vector<Eigen::VectorXd> &numPixelsUnderMask);

/*void blurMinima(const std::vector<posInfo> & scores,
  const Eigen::Vector4i & rows, const Eigen::Vector4i & cols,
  std::vector<Eigen::MatrixXd> & scoreMatricies);*/

void analyzePlacementWeighted(
    const std::vector<Eigen::SparseMatrix<double>> &fpPyramid,
    const std::vector<Eigen::SparseMatrix<double>> &erodedFpPyramid,
    const std::string &scanName, const std::string &zerosFile,
    const std::string &maskName);

void createFPPyramidsWeighted(
    const Eigen::SparseMatrix<double> &weightedFloorPlan,
    std::vector<Eigen::SparseMatrix<double>> &fpPyramid,
    std::vector<Eigen::SparseMatrix<double>> &erodedFpPyramid);

template <typename MatType>
void createPyramid(std::vector<MatType> &pyramid,
                   int levels = FLAGS_numLevels) {
  typedef typename MatType::Scalar Scalar;
  typedef Eigen::Triplet<Scalar> TripType;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatNS;

  std::vector<TripType> tripletList;

  for (int i = 0; i < levels; ++i) {
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
void createPyramid(std::vector<std::vector<MatType>> &pyramid,
                   int levels = FLAGS_numLevels) {
  typedef typename MatType::Scalar Scalar;
  typedef Eigen::Triplet<Scalar> TripType;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatNS;

  std::vector<TripType> tripletList;

  for (int i = 0; i < levels; ++i) {
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

} // namespace place

#endif