/**
  This file is responsbile for taking in cloud normals and extracting
  the 3 dominate directions of the room (x, y, and z-axis) and then
  calculating the 4 rotation matrices needed to align one of the room
  axises that is prependicular to the z-axis to the Manhattan world axises
*/

#include "getRotations.h"
#include "preprocessor.h"

#include <algorithm>
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/StdVector>
#include <fstream>
#include <iostream>
#include <random>

#include <omp.h>

void satoshiRansacManhattan1(const std::vector<Eigen::Vector3d> &,
                             Eigen::Vector3d &);
void satoshiRansacManhattan2(const std::vector<Eigen::Vector3d> &,
                             const Eigen::Vector3d &, Eigen::Vector3d &,
                             Eigen::Vector3d &);
void getMajorAngles(const Eigen::Vector3d &, std::vector<Eigen::Matrix3d> &);

void getRotations(const pcl::PointCloud<NormalType>::Ptr &cloud_normals,
                  const std::string &outName) {

  if (!FLAGS_redo) {
    std::ifstream in(outName, std::ios::in | std::ios::binary);
    if (in.is_open())
      return;
  }

  // NB: Convert pcl to eigen so linear algebra is easier
  std::vector<Eigen::Vector3d> normals;
  normals.reserve(cloud_normals->size());
  for (auto &n : *cloud_normals)
    normals.emplace_back(n.normal_x, n.normal_y, n.normal_z);

  if (!FLAGS_quietMode)
    std::cout << "N size: " << normals.size() << std::endl;

  Eigen::Vector3d d1, d2, d3;
  satoshiRansacManhattan1(normals, d1);
  if (!FLAGS_quietMode) {
    std::cout << "D1: " << d1 << std::endl << std::endl;
  }
  // NB: Select normals that are perpendicular to the first
  // dominate direction
  std::vector<Eigen::Vector3d> N2;
  for (auto &n : normals)
    if (std::asin(n.cross(d1).norm()) > PI / 2.0 - 0.02)
      N2.push_back(n);

  if (!FLAGS_quietMode)
    std::cout << "N2 size: " << N2.size() << std::endl;

  satoshiRansacManhattan2(N2, d1, d2, d3);

  if (!FLAGS_quietMode) {
    std::cout << "D2: " << d2 << std::endl << std::endl;
    std::cout << "D3: " << d3 << std::endl << std::endl;
  }

  std::vector<Eigen::Matrix3d> R(4);

  if (std::abs(d1[2]) < 0.02)
    getMajorAngles(d1, R);
  else if (std::abs(d2[2]) < 0.02)
    getMajorAngles(d2, R);
  else
    getMajorAngles(d3, R);

  if (!FLAGS_quietMode) {
    for (auto &r : R)
      std::cout << r << std::endl << std::endl;
  }

  if (FLAGS_save) {
    std::ofstream binaryWriter(outName, std::ios::out | std::ios::binary);
    for (int i = 0; i < R.size(); ++i) {
      binaryWriter.write(reinterpret_cast<const char *>(R[i].data()),
                         sizeof(Eigen::Matrix3d));
    }
    binaryWriter.close();
  }
}

#pragma omp declare reduction(+ : Eigen::Vector3d : omp_out += omp_in)

/**
  Gets the first dominate direction.  Dominate direction extraction
  is done using RANSAC.  N is all normals and M is the ouput
*/
void satoshiRansacManhattan1(const std::vector<Eigen::Vector3d> &N,
                             Eigen::Vector3d &M) {
  const int m = N.size();

  volatile double maxInliers = 0, K = 1e5;
  volatile int k = 0;

  static std::random_device seed;
  static std::mt19937_64 gen(seed());
  std::uniform_int_distribution<int> dist(0, m - 1);

  while (k < K) {
    // random sampling
    int randomIndex = dist(gen);
    // compute the model parameters
    const Eigen::Vector3d &nest = N[randomIndex];

    // Count the number of inliers
    double numInliers = 0;
    Eigen::Vector3d average = Eigen::Vector3d::Zero();
#pragma omp parallel for reduction(+ : average, numInliers)
    for (int i = 0; i < m; ++i) {
      auto &n = N[i];
      // nest and n are both unit vectors, so this is |angle|
      // between them
      if (std::acos(std::abs(nest.dot(n))) < 0.02) {
        ++numInliers;
        // NB: All normals that are inliers with the estimate
        // are averaged together to get the best estimate
        // of the dominate direction
        if (nest.dot(n) < 0)
          average -= n;
        else
          average += n;
      }
    }

    if (numInliers > maxInliers) {
      maxInliers = numInliers;

      M = average / average.norm();
      // NB: Ransac formula to check for consensus
      double w = (numInliers - 3) / m;
      double p = std::max(0.001, std::pow(w, 3));
      K = log(1 - 0.999) / log(1 - p);
    }
    ++k;
  }
}

/**
  Extracts the remaining two dominate directions simultaneously.
  N is all normals perpendicular to n1, the first dominate direction.
  M1 and M2 are the outputs for the two remaining dominate directions.
  This method follows a very similar the first version of it
*/
void satoshiRansacManhattan2(const std::vector<Eigen::Vector3d> &N,
                             const Eigen::Vector3d &n1, Eigen::Vector3d &M1,
                             Eigen::Vector3d &M2) {
  const int m = N.size();

  volatile double maxInliers = 0, K = 1.0e5;
  volatile int k = 0;

  static std::random_device seed;
  static std::mt19937_64 gen(seed());
  std::uniform_int_distribution<int> dist(0, m - 1);

  while (k < K) {
    // random sampling
    int randomIndex = dist(gen);
    // compute the model parameters
    const Eigen::Vector3d &nest = N[randomIndex];

    const Eigen::Vector3d nest2 = nest.cross(n1);

    // counting inliers and outliers
    double numInliers = 0;
    Eigen::Vector3d average = Eigen::Vector3d::Zero();
#pragma omp parallel for reduction(+ : average, numInliers)
    for (int i = 0; i < m; ++i) {
      auto &n = N[i];
      Eigen::Vector3d x;
      if (std::min(std::acos(std::abs(nest.dot(n))),
                   std::acos(std::abs(nest2.dot(n)))) < 0.02) {
        if (std::acos(std::abs(nest.dot(n))) < 0.02) {
          x = n;
        } else {
          x = n.cross(n1);
        }

        if (nest.dot(x) < 0)
          average -= x;
        else
          average += x;
        ++numInliers;
      }
    }

    if (numInliers > maxInliers) {
      maxInliers = numInliers;

      average /= average.norm();
      M1 = average;
      M2 = average.cross(n1);

      double w = (maxInliers - 3) / m;
      double p = std::max(0.001, std::pow(w, 3));
      K = log(1 - 0.999) / log(1 - p);
    }
    ++k;
  }
}

void getMajorAngles(const Eigen::Vector3d &M, std::vector<Eigen::Matrix3d> &R) {
  // Angle between the dominate direction and the x-axis
  const double theta = atan2(M[1], M[0]);
  // Calculate the rotation matrix require to rotate a given axis onto
  // the dominate direction.  Order is X, Y, -X, -Y
  for (int i = 0; i < NUM_ROTS; ++i) {
    Eigen::Matrix3d out = Eigen::Matrix3d::Identity();
    out(0, 0) = cos(-theta + 2 * PI / NUM_ROTS * i);
    out(1, 1) = cos(-theta + 2 * PI / NUM_ROTS * i);
    out(0, 1) = -sin(-theta + 2 * PI / NUM_ROTS * i);
    out(1, 0) = sin(-theta + 2 * PI / NUM_ROTS * i);
    R[i] = out;
  }
}