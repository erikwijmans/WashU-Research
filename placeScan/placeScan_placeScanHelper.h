#pragma once
#ifndef PLACESCAN_PLACE_SCAN_HELPER_HPP_
#define PLACESCAN_PLACE_SCAN_HELPER_HPP_

#include <scan_gflags.h>

#include <string>

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/StdVector>
#include <gflags/gflags.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <scan_typedefs.hpp>

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

void loadInScans(const std::string &scanName, const std::string &zerosFile,
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
                     const std::string &preDone);

void displayOutput(
    const std::vector<Eigen::SparseMatrix<double>> &rSSparseTrimmed,
    const std::vector<const place::posInfo *> &minima);

void loadInTruePlacement(const std::string &scanName,
                         const std::vector<Eigen::Vector2i> &zeroZero);

void displayTruePlacement(const std::vector<Eigen::SparseMatrix<double>> &,
                          const std::vector<posInfo> &scores,
                          const std::vector<Eigen::Vector2i> &zeroZero);

cv::Mat sparseToImage(const Eigen::SparseMatrix<double> &toImage);

void sparseToImage(const Eigen::SparseMatrix<double> &toImage,
                   cv::Mat &imageOut);

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
    const std::vector<const place::posInfo *> &minima);

void removeMinimumConnectedComponents(cv::Mat &image);

template <class E, class UrnaryFunc>
int getCutoffIndex(const std::vector<E> &list, UrnaryFunc selector) {
  constexpr int max = 30;
  constexpr int minNodes = 3;
  static bool first = true;
  static double cutoffDelta, cutoffScore, cutOffTotal, instantCutoff;
  if (first) {
    std::vector<std::string> names;
    parseFolder(FLAGS_outputV1, names, [](const std::string &s) {
      return s.find(".dat") != std::string::npos;
    });
    std::vector<double> scores, deltas, totalDeltas;
    for (auto &n : names) {
      std::ifstream in(FLAGS_outputV1 + n, std::ios::in | std::ios::binary);
      int num;
      in.read(reinterpret_cast<char *>(&num), sizeof(num));
      place::posInfo tmp;
      in.read(reinterpret_cast<char *>(&tmp), sizeof(tmp));
      double score = tmp.score;
      const double firstScore = score;
      scores.push_back(score);
      for (int i = 0; i < std::min(num - 1, 100); ++i) {
        in.read(reinterpret_cast<char *>(&tmp), sizeof(tmp));
        score = tmp.score;
        const double delta = score - scores.back();
        const double totalDelta = score - firstScore;
        if (delta > 0.0001)
          deltas.push_back(delta);
        scores.push_back(score);
        totalDeltas.push_back(totalDelta);
      }
    }
    double averageScore, sigmaScore, averageDelta, sigmaDelta, averageTotal,
        sigmaTotal;
    std::tie(averageScore, sigmaScore) =
        place::aveAndStdev(scores.begin(), scores.end());
    std::tie(averageDelta, sigmaDelta) =
        place::aveAndStdev(deltas.begin(), deltas.end());
    std::tie(averageTotal, sigmaTotal) =
        place::aveAndStdev(totalDeltas.begin(), totalDeltas.end());

    instantCutoff = averageDelta + 6.5 * sigmaDelta;
    cutOffTotal = averageTotal - 1.5 * sigmaTotal;
    cutoffScore = averageScore - 2.0 * sigmaScore;
    cutoffDelta = averageDelta + 4 * sigmaDelta;
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
    first = false;
  }

  bool exceeded = false;
  double prevScore = selector(list[0]);
  const double firstScore = prevScore;
  int i;
  for (i = 1; i < list.size(); ++i) {
    const double score = selector(list[i]);
    if (i >= max || score - prevScore > instantCutoff ||
        (i >= minNodes && exceeded))
      break;

    if (score - firstScore > cutOffTotal || score > cutoffScore ||
        score - prevScore > cutoffDelta)
      exceeded = true;

    prevScore = score;
  }

  return std::min(i != 1 ? i + 1 : i, (int)list.size());
}
} // namespace place

#endif