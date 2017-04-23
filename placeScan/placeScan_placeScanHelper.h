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

void parseFolders(std::vector<fs::path> &pointFileNames,
                  std::vector<fs::path> &zerosFileNames,
                  std::vector<fs::path> *freeFileNames);

void loadInScans(const fs::path &scanName, const fs::path &zerosFile,
                 std::vector<cv::Mat> &rotatedScans,
                 std::vector<Eigen::Vector2i> &zeroZero);

void loadInScans(const fs::path &scanName, std::vector<cv::Mat> &rotatedScans);

void loadInScansAndMasks(const fs::path &scanName, const fs::path &zerosFile,
                         const fs::path &maskName,
                         std::vector<cv::Mat> &rotatedScans,
                         std::vector<cv::Mat> &masks,
                         std::vector<Eigen::Vector2i> &zeroZero);

void trimScans(const std::vector<cv::Mat> &toTrim,
               std::vector<cv::Mat> &trimmedScans,
               std::vector<Eigen::Vector2i> &zeroZero);

void savePlacement(const std::vector<const place::posInfo *> &minima,
                   const fs::path &outName,
                   const std::vector<Eigen::Vector2i> &zeroZero);

bool reshowPlacement(const fs::path &scanName, const fs::path &zerosFile,
                     const fs::path &doorName, const place::DoorDetector &d,
                     const fs::path &preDone);

void scanToSparse(const cv::Mat &scan, Eigen::SparseMatrix<double> &sparse);

Eigen::SparseMatrix<double> scanToSparse(const cv::Mat &scan);

void erodeSparse(const Eigen::SparseMatrix<double> &src,
                 Eigen::SparseMatrix<double> &dst);

void removeMinimumConnectedComponents(cv::Mat &image);

std::vector<std::vector<place::Door>>
loadInDoors(const fs::path &name, const std::vector<Eigen::Vector2i> &zeroZero);

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