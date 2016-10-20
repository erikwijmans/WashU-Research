#include "placeScan_panoramaMatcher.h"
#include "placeScan_placeScanHelper2.h"

#include <iostream>
#include <scan_typedefs.hpp>

static constexpr int NCCSize = 11;
static_assert(NCCSize % 2 == 1, "NCCSize isn't odd");

static constexpr double degreesToRadians(double degrees) {
  return degrees * PI / 180.0;
}

static Eigen::Vector3d cartesianToPolar(const Eigen::Vector3d &coords) {
  double r = coords.norm();
  double theta = atan2(coords[1], coords[0]);
  double phi = acos(coords[2] / r);
  return Eigen::Vector3d(r, theta, phi);
}

static Eigen::Vector3d polarToCartesian(const Eigen::Vector3d &coords) {
  double x = coords[0] * cos(coords[1]) * sin(coords[2]);
  double y = coords[0] * sin(coords[1]) * sin(coords[2]);
  double z = coords[0] * cos(coords[2]);
  return Eigen::Vector3d(x, y, z);
}

static Eigen::Vector2d pointCloudToPanorama(const Eigen::Vector3d &worldCoord,
                                            double &r,
                                            const cv::Size &resolution) {
  auto polar = cartesianToPolar(worldCoord);
  double x = (polar[1] / PI + 1.0) * (resolution.width - 1.0) / 2.0;
  double y = polar[2] / maxPhi * (resolution.height - 1.0);
  r = polar[0];
  return Eigen::Vector2d(x, y);
}

static Eigen::Vector2d pointCloudToPanorama(const Eigen::Vector3d &worldCoord,
                                            const cv::Size &resolution) {
  double tmp;
  return pointCloudToPanorama(worldCoord, tmp, resolution);
}

static Eigen::Vector2d voxelSpaceToPanorama(const Eigen::Vector3d &voxelCoord,
                                            double &r,
                                            const cv::Size &resolution) {
  Eigen::Vector3d local = voxelCoord;
  local[1] *= -1.0;
  return pointCloudToPanorama(local, r, resolution);
}

static Eigen::Vector2d voxelSpaceToPanorama(const Eigen::Vector3d &voxelCoord,
                                            const cv::Size &resolution) {
  double tmp;
  return voxelSpaceToPanorama(voxelCoord, tmp, resolution);
}

static Eigen::Vector3d panoramaToPointCloud(const Eigen::Vector2d &panoCoord,
                                            const double r,
                                            const cv::Size &resolution) {
  double theta = (2.0 * panoCoord[0] / (resolution.width - 1.0) + 1.0) * PI;
  double phi = panoCoord[1] * maxPhi / (resolution.height - 1.0);
  return polarToCartesian(Eigen::Vector3d(r, theta, phi));
}

static Eigen::Vector3d panoramaToVoxelSpace(const Eigen::Vector2d &panoCoord,
                                            const double r,
                                            const cv::Size &resolution) {
  auto voxel = panoramaToPointCloud(panoCoord, r, resolution);
  voxel[1] *= -1.0;
  return voxel;
}

static double aveDepth(const Eigen::RowMatrixXf &depthMap, const int x,
                       const int y) {
  int count = 0;
  double ave = 0;
  constexpr int limit = NCCSize / 2;
  static_assert(limit * 2 + 1 == NCCSize, "limit is wrong");

  for (int j = -limit; j <= limit; ++j) {
    for (int i = -limit; i <= limit; ++i) {
      if (x + i < 0 || x + i >= depthMap.cols())
        continue;
      if (y + j < 0 || y + j >= depthMap.rows())
        continue;
      if (!depthMap(y + j, x + i))
        continue;
      ave += depthMap(y + j, x + i);
      ++count;
    }
  }
  return count > 0 ? ave / count : 0;
}

template <typename T, typename S>
static inline double getAngle(const T &a, const S &b) {
  if (a.norm() == 0 || b.norm() == 0)
    return PI;
  return std::acos(static_cast<double>(a.dot(b)) / (a.norm() * b.norm()));
}

double NCC(const cv::Mat_<cv::Vec3b> &a, const cv::Mat_<cv::Vec3b> &b) {
  assert(a.rows == b.rows);
  assert(a.cols == b.cols);
  assert(a.channels() == b.channels());

  Eigen::VectorXd aveA = Eigen::VectorXd::Zero(a.channels()),
                  aveB = Eigen::VectorXd::Zero(b.channels());
  for (int j = 0; j < a.rows; ++j) {
    for (int i = 0; i < a.cols; ++i) {
      for (int c = 0; c < a.channels(); ++c) {
        aveA[c] += a(j, i)[c];
        aveB[c] += b(j, i)[c];
      }
    }
  }
  aveA /= a.rows * a.cols;
  aveB /= a.rows * a.cols;

  double AB = 0, AA = 0, BB = 0;
  for (int j = 0; j < a.rows; ++j) {
    for (int i = 0; i < a.cols; ++i) {
      for (int c = 0; c < a.channels(); ++c) {
        AA += (a(j, i)[c] - aveA[c]) * (a(j, i)[c] - aveA[c]);
        BB += (b(j, i)[c] - aveB[c]) * (b(j, i)[c] - aveB[c]);
        AB += (b(j, i)[c] - aveB[c]) * (a(j, i)[c] - aveA[c]);
      }
    }
  }
  return AB / sqrt(AA * BB);
}

#define viz 1

void pano::compareNCC2(place::Panorama &panoA, place::Panorama &panoB,
                       const Eigen::Matrix3d &RA, const Eigen::Matrix3d &RB,
                       const Eigen::Vector3d &aToB, const Eigen::Vector3d &bToA,
                       place::edge &e) {

  constexpr double cutoffAngle = degreesToRadians(20);
  constexpr double maxDiff = 0.3;
  constexpr double occulisionCutoff = 0.3;
  constexpr int offset = NCCSize / 2;
  constexpr int maxLvlDiff = 3;
  constexpr double simThreshold = 0.4;
  static_assert(offset * 2 + 1 == NCCSize, "offset isn't correct");

  const Eigen::Matrix3d aToBRotMat = RB * RA.inverse();
  const Eigen::Vector3d aToBTrans = RB * aToB;

  const Eigen::Matrix3d bToARotMat = RA * RB.inverse();
  const Eigen::Vector3d bToATrans = RA * bToA;

  double score = 0.0;
  int count = 0, numSim = 0, numDiff = 0;

  const std::vector<cv::Point2f> &truePointsInA = panoA.keypoints;
  const std::vector<cv::Point2f> &truePointsInB = panoB.keypoints;

  const Eigen::RowMatrixXf &rMapA = panoA.rMap;
  const Eigen::RowMatrixXf &rMapB = panoB.rMap;

  const Eigen::Vector2i aMaxes(std::min(panoA[0].cols, (int)rMapA.cols()),
                               std::min(panoA[0].rows, (int)rMapA.rows())),
      bMaxes(std::min(panoB[0].cols, (int)rMapB.cols()),
             std::min(panoB[0].rows, (int)rMapB.rows()));

#pragma omp parallel reduction(+ : count, score, numSim,                       \
                               numDiff) if (!FLAGS_debugMode || !viz)
  {
#pragma omp for nowait
    for (int i = 0; i < truePointsInA.size(); ++i) {
      const cv::Point2f &kp = truePointsInA[i];
      Eigen::Vector2d coordA(kp.x, kp.y);
      const double r = rMapA(coordA[1], coordA[0]);
      if (r < 0.5)
        continue;
      auto aVoxelSpace = panoramaToVoxelSpace(coordA, r, panoA[0].size());
      auto bVoxelSpace = aToBRotMat * aVoxelSpace + aToBTrans;
      auto coordB = voxelSpaceToPanorama(bVoxelSpace, panoB[0].size());

      if (coordB[0] < 0 || coordB[0] >= bMaxes[0])
        continue;
      if (coordB[1] < 0 || coordB[1] >= bMaxes[1])
        continue;

      const double radiusA = aVoxelSpace.norm();
      const double radiusB = bVoxelSpace.norm();
      const double trueRadiusB = rMapB(coordB[1], coordB[0]);
      const double aveDepthB = aveDepth(rMapB, coordB[0], coordB[1]);
      const double aveDepthA = aveDepth(rMapA, coordA[0], coordA[1]);
      const double angle = getAngle(aToBRotMat * aVoxelSpace, bVoxelSpace);
      Eigen::Vector3d bNormal =
          panoB.surfaceNormals(coordB[1], coordB[0]).cast<double>();
      bNormal[1] *= -1.0;
      bNormal = bToARotMat * bNormal;
      bNormal[1] *= -1.0;
      const double normalAngle = getAngle(
          panoA.surfaceNormals(coordA[1], coordA[0]), bNormal.cast<float>());
      if (angle > cutoffAngle ||
          std::abs(radiusB - trueRadiusB) > occulisionCutoff ||
          std::abs(aveDepthA - r) > maxDiff ||
          std::abs(trueRadiusB - aveDepthB) > maxDiff ||
          normalAngle > cutoffAngle)
        continue;

      int aLevel =
          std::max(0.0, std::round(std::log(radiusB / radiusA) /
                                   std::log(place::Panorama::ScalingFactor)));
      int bLevel =
          std::max(0.0, std::round(std::log(radiusA / radiusB) /
                                   std::log(place::Panorama::ScalingFactor)));

      auto a = voxelSpaceToPanorama(aVoxelSpace, panoA[aLevel].size());
      auto b = voxelSpaceToPanorama(bVoxelSpace, panoB[bLevel].size());

      const cv::Mat_<cv::Vec3b> aLvlImg = panoA[aLevel];
      const cv::Mat_<cv::Vec3b> bLvlImg = panoB[bLevel];

      if (a[1] - offset < 0 || a[1] + offset >= aLvlImg.rows ||
          a[0] - offset < 0 || a[0] + offset >= aLvlImg.cols)
        continue;

      if (b[1] - offset < 0 || b[1] + offset >= bLvlImg.rows ||
          b[0] - offset < 0 || b[0] + offset >= bLvlImg.cols)
        continue;

      auto NCCB = bLvlImg(cv::Range(b[1] - offset, b[1] + offset + 1),
                          cv::Range(b[0] - offset, b[0] + offset + 1));
      auto NCCA = aLvlImg(cv::Range(a[1] - offset, a[1] + offset + 1),
                          cv::Range(a[0] - offset, a[0] + offset + 1));

      auto mean = cv::mean(NCCA);
      if (mean.val[0] > 240 && mean.val[1] > 240 && mean.val[2] > 240)
        continue;

      mean = cv::mean(NCCB);
      if (mean.val[0] > 240 && mean.val[1] > 240 && mean.val[2] > 240)
        continue;

      const double ncc = NCC(NCCA, NCCB);
      if (!Eigen::numext::isfinite(ncc))
        continue;
      score += ncc;
      ++count;

      if (ncc > simThreshold)
        ++numSim;
      else
        ++numDiff;

#if viz
      if (FLAGS_debugMode) {
        cv::Mat_<cv::Vec3b> out1(aLvlImg.size());
        aLvlImg.copyTo(out1);
        cv::Mat_<cv::Vec3b> out2(bLvlImg.size());
        bLvlImg.copyTo(out2);

        for (int j = 0; j < NCCSize; ++j) {
          for (int i = 0; i < NCCSize; ++i) {
            out1(j - offset + a[1], i - offset + a[0])[0] = 0;
            out1(j - offset + a[1], i - offset + a[0])[1] = 0;
            out1(j - offset + a[1], i - offset + a[0])[2] = 255;

            out2(j - offset + b[1], i - offset + b[0])[0] = 0;
            out2(j - offset + b[1], i - offset + b[0])[1] = 0;
            out2(j - offset + b[1], i - offset + b[0])[2] = 255;
          }
        }
        std::cout << "aLevel: " << aLevel << std::endl;
        std::cout << "bLevel: " << bLevel << std::endl;
        std::cout << "aRadius: " << radiusA << std::endl;
        std::cout << "bRadius: " << radiusB << std::endl;

        std::cout << "NCC: " << ncc << std::endl;
        cvNamedWindow("A", CV_WINDOW_NORMAL);
        cv::imshow("A", out1);

        cvNamedWindow("B", CV_WINDOW_NORMAL);
        cv::imshow("B", out2);

        cvNamedWindow("NCCA", CV_WINDOW_NORMAL);
        cv::imshow("NCCA", NCCA);

        cvNamedWindow("NCCB", CV_WINDOW_NORMAL);
        cv::imshow("NCCB", NCCB);

        cv::waitKey(0);
      }
#endif
    }

#pragma omp for nowait
    for (int i = 0; i < truePointsInB.size(); ++i) {
      const cv::Point2f &kp = truePointsInB[i];
      Eigen::Vector2d coordB(kp.x, kp.y);
      const double r = rMapB(coordB[1], coordB[0]);
      if (r < 0.5)
        continue;
      auto bVoxelSpace = panoramaToVoxelSpace(coordB, r, panoB[0].size());
      auto aVoxelSpace = bToARotMat * bVoxelSpace + bToATrans;
      auto coordA = voxelSpaceToPanorama(aVoxelSpace, panoA[0].size());

      if (coordA[0] < 0 || coordA[0] >= aMaxes[0])
        continue;
      if (coordA[1] < 0 || coordA[1] >= aMaxes[1])
        continue;

      const double radiusA = aVoxelSpace.norm();
      const double radiusB = bVoxelSpace.norm();
      const double trueRadiusA = rMapA(coordA[1], coordA[0]);
      const double aveDepthB = aveDepth(rMapB, coordB[0], coordB[1]);
      const double aveDepthA = aveDepth(rMapA, coordA[0], coordA[1]);
      const double angle = getAngle(aVoxelSpace, bToARotMat * bVoxelSpace);
      Eigen::Vector3d aNormal =
          panoA.surfaceNormals(coordA[1], coordA[0]).cast<double>();
      aNormal[1] *= -1.0;
      aNormal = aToBRotMat * aNormal;
      aNormal[1] *= -1.0;
      const double normalAngle = getAngle(
          aNormal.cast<float>(), panoB.surfaceNormals(coordB[1], coordB[0]));
      if (angle > cutoffAngle ||
          std::abs(radiusA - trueRadiusA) > occulisionCutoff ||
          std::abs(aveDepthA - trueRadiusA) > maxDiff ||
          std::abs(r - aveDepthB) > maxDiff || normalAngle > cutoffAngle)
        continue;

      int aLevel =
          std::max(0.0, std::round(std::log(radiusB / radiusA) /
                                   std::log(place::Panorama::ScalingFactor)));
      int bLevel =
          std::max(0.0, std::round(std::log(radiusA / radiusB) /
                                   std::log(place::Panorama::ScalingFactor)));

      auto a = voxelSpaceToPanorama(aVoxelSpace, panoA[aLevel].size());
      auto b = voxelSpaceToPanorama(bVoxelSpace, panoB[bLevel].size());

      const cv::Mat_<cv::Vec3b> aLvlImg = panoA[aLevel];
      const cv::Mat_<cv::Vec3b> bLvlImg = panoB[bLevel];

      if (a[1] - offset < 0 || a[1] + offset >= aLvlImg.rows ||
          a[0] - offset < 0 || a[0] + offset >= aLvlImg.cols)
        continue;

      if (b[1] - offset < 0 || b[1] + offset >= bLvlImg.rows ||
          b[0] - offset < 0 || b[0] + offset >= bLvlImg.cols)
        continue;

      auto NCCB = bLvlImg(cv::Range(b[1] - offset, b[1] + offset + 1),
                          cv::Range(b[0] - offset, b[0] + offset + 1));
      auto NCCA = aLvlImg(cv::Range(a[1] - offset, a[1] + offset + 1),
                          cv::Range(a[0] - offset, a[0] + offset + 1));

      const double ncc = NCC(NCCA, NCCB);
      if (!Eigen::numext::isfinite(ncc))
        continue;
      score += ncc;
      ++count;

      if (ncc > simThreshold)
        ++numSim;
      else
        ++numDiff;
#if viz
      if (FLAGS_debugMode) {
        cv::Mat_<cv::Vec3b> out1(aLvlImg.rows, aLvlImg.cols);
        aLvlImg.copyTo(out1);
        cv::Mat_<cv::Vec3b> out2(bLvlImg.rows, bLvlImg.cols);
        bLvlImg.copyTo(out2);

        for (int j = 0; j < NCCSize; ++j) {
          for (int i = 0; i < NCCSize; ++i) {
            out1(j - offset + a[1], i - offset + a[0])[0] = 0;
            out1(j - offset + a[1], i - offset + a[0])[1] = 0;
            out1(j - offset + a[1], i - offset + a[0])[2] = 255;

            out2(j - offset + b[1], i - offset + b[0])[0] = 0;
            out2(j - offset + b[1], i - offset + b[0])[1] = 0;
            out2(j - offset + b[1], i - offset + b[0])[2] = 255;
          }
        }
        std::cout << "aLevel: " << aLevel << std::endl;
        std::cout << "bLevel: " << bLevel << std::endl;

        std::cout << "NCC: " << ncc << std::endl;
        cvNamedWindow("A", CV_WINDOW_NORMAL);
        cv::imshow("A", out1);

        cvNamedWindow("B", CV_WINDOW_NORMAL);
        cv::imshow("B", out2);

        cvNamedWindow("NCCA", CV_WINDOW_NORMAL);
        cv::imshow("NCCA", NCCA);

        cvNamedWindow("NCCB", CV_WINDOW_NORMAL);
        cv::imshow("NCCB", NCCB);

        cv::waitKey(0);
      }
#endif
    }
  }
  score /= count;

  constexpr double precent = 0.025;
  const double expectedCount =
      (truePointsInA.size() + truePointsInB.size()) / 2.0;

  count /= 2.0;
  const double significance = sigmoidWeight(count, expectedCount * precent);

#if viz
  if (FLAGS_debugMode) {
    std::cout << count << ", " << significance << std::endl;
    std::cout << score << std::endl;
  }
#endif

  if (!Eigen::numext::isfinite(score))
    return;

  e.panoW = score;
  e.panoSignificance = significance;
  e.numSim = numSim;
  e.numDiff = numDiff;
}
