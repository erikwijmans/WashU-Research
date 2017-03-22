#pragma once
#ifndef PLACESCAN_PLACE_SCAN_HELPER_HPP_
#define PLACESCAN_PLACE_SCAN_HELPER_HPP_

#include "placeScan_doorDetector.h"

#include <scan_gflags.h>

#include <string>

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/StdVector>
#include <etw_utils.hpp>
#include <gflags/gflags.h>
#include <iostream>
#include <opencv2/core.hpp>

extern const double maxDelta, maxTotal;

extern cv::Mat fpColor, floorPlan;
extern std::vector<Eigen::Vector3i> truePlacement;

namespace place {

void parseFolders(std::vector<std::string> &pointFileNames,
                  std::vector<std::string> &zerosFileNames,
                  std::vector<std::string> *freeFileNames);

void loadInScans(const std::string &scanName, const std::string &zerosFile,
                 std::vector<cv::Mat> &rotatedScans,
                 std::vector<Eigen::Vector2i> &zeroZero);

void loadInScans(const std::string &scanName,
                 std::vector<cv::Mat> &rotatedScans);

void loadInScansAndMasks(const std::string &scanName,
                         const std::string &zerosFile,
                         const std::string &maskName,
                         std::vector<cv::Mat> &rotatedScans,
                         std::vector<cv::Mat> &masks,
                         std::vector<Eigen::Vector2i> &zeroZero);

void trimScans(const std::vector<cv::Mat> &toTrim,
               std::vector<cv::Mat> &trimmedScans,
               std::vector<Eigen::Vector2i> &zeroZero);

void savePlacement(const std::vector<const place::posInfo *> &minima,
                   const std::string &outName,
                   const std::vector<Eigen::Vector2i> &zeroZero);

bool reshowPlacement(const std::string &scanName, const std::string &zerosFile,
                     const std::string &doorName, const place::DoorDetector &d,
                     const std::string &preDone);

void displayOutput(
    const std::vector<Eigen::SparseMatrix<double>> &rSSparseTrimmed,
    const std::vector<const place::posInfo *> &minima);

void loadInTruePlacement(const std::string &scanName,
                         const std::vector<Eigen::Vector2i> &zeroZero);

void displayTruePlacement(const std::vector<Eigen::SparseMatrix<double>> &,
                          const std::vector<posInfo> &scores,
                          const std::vector<Eigen::Vector2i> &zeroZero);

void scanToSparse(const cv::Mat &scan, Eigen::SparseMatrix<double> &sparse);

Eigen::SparseMatrix<double> scanToSparse(const cv::Mat &scan);

void displayScanAndMask(
    const std::vector<std::vector<Eigen::SparseMatrix<double>>>
        &rSSparsePyramidTrimmed,
    const std::vector<std::vector<Eigen::MatrixXb>> &eMaskPyramidTrimmedNS);

void erodeSparse(const Eigen::SparseMatrix<double> &src,
                 Eigen::SparseMatrix<double> &dst);

void displayOutput(
    const Eigen::SparseMatrix<double> &fp,
    const std::vector<Eigen::SparseMatrix<double>> &rSSparseTrimmed,
    const Eigen::MatrixXb &fpDoors,
    const std::vector<std::vector<place::Door>> &pcDoors,
    const std::vector<const place::posInfo *> &minima);

void removeMinimumConnectedComponents(cv::Mat &image);

std::vector<std::vector<place::Door>>
loadInDoors(const std::string &name,
            const std::vector<Eigen::Vector2i> &zeroZero);

template <class E, class UrnaryFunc>
int getCutoffIndex(const std::string &name, const std::vector<E> &list,
                   UrnaryFunc selector) {
  /*constexpr int max = 30;
  constexpr int minNodes = 3;

  double cutoffDelta, cutoffScore, cutOffTotal, instantCutoff;

  {
    std::ifstream in(FLAGS_outputV1 + name, std::ios::in | std::ios::binary);
    int num;
    in.read(reinterpret_cast<char *>(&num), sizeof(num));
    place::posInfo tmp;
    in.read(reinterpret_cast<char *>(&tmp), sizeof(tmp));
    double score = tmp.score;
    const double firstScore = score;

    std::vector<double> scores, totalDeltas, deltas;
    scores.push_back(score);
    for (int i = 0; i < std::min(num - 1, 50); ++i) {
      in.read(reinterpret_cast<char *>(&tmp), sizeof(tmp));
      score = tmp.score;
      const double delta = score - scores.back();
      const double totalDelta = score - firstScore;
      if (delta > 0.0001)
        deltas.push_back(delta);
      scores.push_back(score);
      totalDeltas.push_back(totalDelta);
    }

    double averageScore, sigmaScore, averageDelta, sigmaDelta, averageTotal,
        sigmaTotal;
    std::tie(averageScore, sigmaScore) =
        place::aveAndStdev(scores.begin(), scores.end());
    std::tie(averageDelta, sigmaDelta) =
        place::aveAndStdev(deltas.begin(), deltas.end());
    std::tie(averageTotal, sigmaTotal) =
        place::aveAndStdev(totalDeltas.begin(), totalDeltas.end());

    instantCutoff = std::max(0.02, averageDelta + 3 * sigmaDelta);
    cutOffTotal = std::max(0.10, averageTotal - 1.5 * sigmaTotal);
    cutoffScore = averageScore;
    cutoffDelta = std::max(0.01, averageDelta + 2 * sigmaDelta);
    if (!FLAGS_quietMode) {
      std::cout << "Score:  " << averageScore << "  " << sigmaScore << std::endl
                << "Total Delta:  " << averageTotal << "  " << sigmaTotal
                << std::endl
                << "Delta:  " << averageDelta << "  " << sigmaDelta << std::endl
                << std::endl
                << "Instant: " << instantCutoff << "  "
                << "Cutoff total: " << cutOffTotal << "  "
                << "Cutoff score: " << cutoffScore << "  "
                << "Cutff delta: " << cutoffDelta << std::endl;
    }
  }

  bool exceeded = false, instant = false;
  double prevScore = selector(list.front());
  const double firstScore = prevScore;
  int i = 0;
  for (auto &e : list) {
    const double score = selector(e);

    if (score - prevScore > instantCutoff)
      instant = true;

    if (i >= max || score - prevScore > instantCutoff ||
        (i >= minNodes && exceeded))
      break;

    if (score - firstScore > cutOffTotal || score > cutoffScore ||
        score - prevScore > cutoffDelta)
      exceeded = true;

    prevScore = score;
    ++i;
  }*/

  // return std::min(instant ? i : i + 1, (int)list.size());
  // return std::max(5, std::min(instant ? i : i + 1, (int)list.size()));
  return std::min(5, (int)list.size());
}

template <typename Scalar>
void sparseToImage(const Eigen::SparseMatrix<Scalar> &toImage,
                   cv::Mat &imageOut) {

  imageOut =
      cv::Mat(toImage.rows(), toImage.cols(), CV_8UC1, cv::Scalar::all(255));

  Scalar maxV = 0;
  for (int i = 0; i < toImage.outerSize(); ++i) {
    for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(toImage, i); it;
         ++it) {
      maxV = std::max(maxV, it.value());
    }
  }

  for (int i = 0; i < toImage.outerSize(); ++i) {
    for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(toImage, i); it;
         ++it) {
      imageOut.at<uchar>(it.row(), it.col()) =
          255 - 255 * it.value() / static_cast<double>(maxV);
    }
  }
}

template <typename Scalar>
cv::Mat sparseToImage(const Eigen::SparseMatrix<Scalar> &toImage) {

  cv::Mat image(toImage.rows(), toImage.cols(), CV_8UC1, cv::Scalar::all(255));
  Scalar maxV = 0;
  for (int i = 0; i < toImage.outerSize(); ++i) {
    for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(toImage, i); it;
         ++it) {
      maxV = std::max(maxV, it.value());
    }
  }

  for (int i = 0; i < toImage.outerSize(); ++i) {
    for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(toImage, i); it;
         ++it) {
      image.at<uchar>(it.row(), it.col()) =
          255 - 255 * it.value() / static_cast<double>(maxV);
    }
  }
  return image;
}

} // namespace place

#endif