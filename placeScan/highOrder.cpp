#include "highOrder.h"

#include <iostream>
#include <list>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

static void dispHMap(const Eigen::ArrayXH &hMap,
                     const multi::Labeler::HighOrder &highOrder) {
  double average = 0;
  size_t count = 0;
  auto dataPtr = hMap.data();
  for (int i = 0; i < hMap.size(); ++i) {
    if (!(dataPtr + i)->count)
      continue;
    auto it = highOrder.find((dataPtr + i)->incident);
    if (it != highOrder.cend()) {
      assert(it->second.c != 0);
      average += it->second.w;
      ++count;
    }
  }
  average /= count;

  double sigma = 0;
  for (int i = 0; i < hMap.size(); ++i) {
    if (!(dataPtr + i)->count)
      continue;
    auto it = highOrder.find((dataPtr + i)->incident);
    if (it != highOrder.cend())
      sigma += (it->second.w - average) * (it->second.w - average);
  }
  sigma /= count - 1;
  sigma = sqrt(sigma);

  std::cout << average << "  " << sigma << std::endl;

  cv::Mat heatMap(hMap.rows(), hMap.cols(), CV_8UC3);
  fpColor.copyTo(heatMap);
  for (int j = 0; j < heatMap.rows; ++j) {
    uchar *dst = heatMap.ptr<uchar>(j);
    for (int i = 0; i < heatMap.cols; ++i) {
      if (!hMap(j, i).count)
        continue;
      auto it = highOrder.find(hMap(j, i).incident);
      if (it != highOrder.cend() && hMap(j, i).incident.size()) {
        const int gray = cv::saturate_cast<uchar>(
            255.0 * ((it->second.w - average) / sigma + 1.0) / 2.0);
        int r, g, b;
        if (gray < 128) {
          r = 0;
          g = 2 * gray;
          b = 255 - g;
        } else {
          r = 2 * (gray - 128);
          g = 255 - r;
          b = 0;
        }
        dst[3 * i + 0] = b;
        dst[3 * i + 1] = g;
        dst[3 * i + 2] = r;
      }
    }
  }

  cvNamedWindow("Preview", CV_WINDOW_NORMAL);
  cv::imshow("Preview", heatMap);
  cv::waitKey(0);
}

static void dispHMap(Eigen::ArrayXH &hMap) {
  double average = 0;
  int count = 0;
  auto const dataPtr = hMap.data();
  for (int i = 0; i < hMap.size(); ++i) {
    const double weight = (dataPtr + i)->weight;
    if (weight) {
      average += weight;
      ++count;
    }
  }
  average /= count;

  double sigma = 0;
  for (int i = 0; i < hMap.size(); ++i) {
    const double weight = (dataPtr + i)->weight;
    if (weight) {
      sigma += (weight - average) * (weight - average);
    }
  }
  sigma /= count - 1;
  sigma = std::sqrt(sigma);

  cv::Mat heatMap(hMap.rows(), hMap.cols(), CV_8UC3);
  fpColor.copyTo(heatMap);
  for (int j = 0; j < heatMap.rows; ++j) {
    uchar *dst = heatMap.ptr<uchar>(j);
    for (int i = 0; i < heatMap.cols; ++i) {
      if (hMap(j, i).weight) {
        const int gray = cv::saturate_cast<uchar>(
            255.0 * ((hMap(j, i).weight - average) / sigma + 1.0) / 2.0);
        int r, g, b;
        if (gray < 128) {
          r = 0;
          g = 2 * gray;
          b = 255 - g;
        } else {
          r = 2 * (gray - 128);
          g = 255 - r;
          b = 0;
        }
        dst[3 * i + 0] = b;
        dst[3 * i + 1] = g;
        dst[3 * i + 2] = r;
      }
    }
  }

  cvNamedWindow("Preview", CV_WINDOW_NORMAL);
  cv::imshow("Preview", heatMap);
  cv::waitKey(0);
}

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

void place::displayHighOrder(
    const multi::Labeler::HighOrder highOrder,
    const std::vector<place::R2Node> &nodes,
    const std::vector<std::vector<Eigen::MatrixXb>> &scans,
    const std::vector<std::vector<Eigen::Vector2i>> &zeroZeros) {
  for (auto &it : highOrder) {
    auto &key = it.first;
    cv::Mat output(fpColor.rows, fpColor.cols, CV_8UC3);
    fpColor.copyTo(output);
    cv::Mat_<cv::Vec3b> _output = output;
    for (auto &i : key) {
      const place::node &nodeA = nodes[i];

      auto &aScan = scans[nodeA.color][nodeA.rotation];

      auto &zeroZeroA = zeroZeros[nodeA.color][nodeA.rotation];

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
    }
    cvNamedWindow("Preview", CV_WINDOW_NORMAL);
    cv::imshow("Preview", output);
    if (!FLAGS_quietMode) {
      std::cout << it.second.w << ":  ";
      for (auto &i : key)
        std::cout << i << "_";
      std::cout << std::endl;
    }
    cv::waitKey(0);
    ~output;
  }
}
