#include "highOrder.h"

#include <iostream>
#include <list>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

double gaussianWeight(const Eigen::Array2d &pos, const Eigen::Array2d &s) {
  return std::exp(-(pos.square() / (2 * s.square())).sum());
}

void place::createHigherOrderTermsV2(
    const std::vector<std::vector<Eigen::MatrixXb>> &freeSpace,
    const std::vector<std::vector<Eigen::Vector2i>> &zeroZeros,
    const std::vector<place::node> &nodes,
    multi::Labeler::HighOrderV2 &highOrder) {

  const double scale = buildingScale.getScale();
  Eigen::ArrayXH2 hMap(floorPlan.rows, floorPlan.cols);

  for (int a = 0; a < nodes.size(); ++a) {
    auto &currentNode = nodes[a];
    auto &currentScan = freeSpace[currentNode.color][currentNode.rotation];
    auto &zeroZero = zeroZeros[currentNode.color][currentNode.rotation];
    const int xOffset = currentNode.x - zeroZero[0],
              yOffset = currentNode.y - zeroZero[1];

    constexpr double maxRange = 1.5;
    double totalWeight = 0;

    const Eigen::Array2d sigma = Eigen::Array2d(5.0, 5.0);

    for (int j = 0; j < currentScan.rows(); ++j) {
      if (j + yOffset < 0 || j + yOffset >= floorPlan.rows)
        continue;
      for (int i = 0; i < currentScan.cols(); ++i) {
        if (i + xOffset < 0 || i + xOffset >= floorPlan.cols)
          continue;

        const Eigen::Array2d pos =
            Eigen::Array2d(i - zeroZero[0], j - zeroZero[1]) / scale;

        totalWeight += currentScan(j, i) ? gaussianWeight(pos, sigma) : 0;
      }
    }

    for (int j = 0; j < currentScan.rows(); ++j) {
      if (j + yOffset < 0 || j + yOffset >= hMap.rows())
        continue;
      for (int i = 0; i < currentScan.cols(); ++i) {
        if (i + xOffset < 0 || i + xOffset >= hMap.cols())
          continue;

        if (currentScan(j, i)) {
          const Eigen::Array2d pos =
              Eigen::Array2d(i - zeroZero[0], j - zeroZero[1]) / scale;

          auto &h = hMap(j + yOffset, i + xOffset);

          h.incident.emplace_back(a);
          h.weights.emplace_back(maxRange * gaussianWeight(pos, sigma) /
                                 totalWeight);
        }
      }
    }
  }

  auto const data = hMap.data();

  for (int i = 0; i < hMap.size(); ++i) {

    auto &key = (data + i)->incident;

    if (key.size()) {

      auto &w = (data + i)->weights;
      Eigen::VectorXd weights(w.size());
      for (int i = 0; i < w.size(); ++i)
        weights[i] = w[i];

      auto it = highOrder.find(key);
      if (it != highOrder.cend())
        it->second += weights;
      else
        highOrder.emplace(key, weights);
    }
  }
}