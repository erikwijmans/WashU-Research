#include "getRotations.h"
#include "preprocessor.h"

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/StdVector>
#include <random>
#include <iostream>
#include <algorithm>
#include <fstream>

#include <omp.h>

void satoshiRansacManhattan1(const std::vector<Eigen::Vector3d> &, Eigen::Vector3d &);
void satoshiRansacManhattan2(const std::vector<Eigen::Vector3d> &, const Eigen::Vector3d &,
  Eigen::Vector3d &, Eigen::Vector3d &);
void getMajorAngles(const Eigen::Vector3d &, std::vector<Eigen::Matrix3d> &);
Eigen::Matrix3d getRotationMatrix(const Eigen::Vector3d &, const Eigen::Vector3d &);

void getRotations(const pcl::PointCloud<NormalType>::Ptr & cloud_normals,
  const std::string & outName) {

  if (!FLAGS_redo) {
    std::ifstream in (outName, std::ios::in | std::ios::binary);
    if (in.is_open())
      return;
  }

  std::vector<Eigen::Vector3d> normals;

  normals.reserve(cloud_normals->size());
  for (auto & n : *cloud_normals)
    normals.emplace_back(n.normal_x, n.normal_y, n.normal_z);

  if(!FLAGS_quietMode)
    std::cout << "N size: " << normals.size() << std::endl;

  Eigen::Vector3d d1, d2, d3;
  satoshiRansacManhattan1(normals, d1);
  if (!FLAGS_quietMode) {
    std::cout << "D1: " << d1 << std::endl << std::endl;
  }
  std::vector<Eigen::Vector3d> N2;
  for (auto & n : normals)
    if(std::asin(n.cross(d1).norm()) > PI/2.0 - 0.02)
      N2.push_back(n);

  if(!FLAGS_quietMode)
    std::cout << "N2 size: " << N2.size() << std::endl;

  satoshiRansacManhattan2(N2, d1, d2, d3);

  if(!FLAGS_quietMode) {
    std::cout << "D2: " << d2 << std::endl << std::endl;
    std::cout << "D3: " << d3 << std::endl << std::endl;
  }

  std::vector<Eigen::Matrix3d> R (4);

  if(std::abs(d1[2]) < 0.02)
    getMajorAngles(d1, R);
  else if( std::abs(d2[2]) < 0.02)
    getMajorAngles(d2, R);
  else
    getMajorAngles(d3, R);

  if (FLAGS_save) {
    std::ofstream binaryWriter (outName, std::ios::out | std::ios::binary);
    for(int i = 0; i < R.size(); ++i) {
      binaryWriter.write(reinterpret_cast<const char *> (R[i].data()),
        sizeof(Eigen::Matrix3d));
    }
    binaryWriter.close();
  }
}


void satoshiRansacManhattan1(const std::vector<Eigen::Vector3d> & N, Eigen::Vector3d & M) {
  const int m = N.size();

  volatile double maxInliers = 0, K = 1e5;
  volatile int k = 0;

  #pragma omp parallel shared(k, K, maxInliers, N, M)
  {
    std::random_device seed;
    std::mt19937_64 gen(seed());
    std::uniform_int_distribution<int> dist(0, m - 1);
    for (int i = 0; i < 5000; ++i)
      dist(gen);

    while (k < K) {
      // random sampling
      int randomIndex = dist(gen);
      // compute the model parameters
      const Eigen::Vector3d & nest = N[randomIndex];

      // counting inliers and outliers
      double numInliers = 0;
      Eigen::Vector3d average = Eigen::Vector3d::Zero();
      for(auto & n : N) {
        if (std::acos(std::abs(nest.dot(n))) < 0.02) {
          ++numInliers;
          if (nest.dot(n) < 0)
            average -= n;
          else
            average += n;
        }
      }

      #pragma omp crtical
      {
        if(numInliers > maxInliers) {
          maxInliers = numInliers;

          M = average/average.norm();

          double w = (numInliers-3)/m;
          double p = std::max(0.001, std::pow(w,3));
          K = log(1-0.999)/log(1-p);
        }
        if(k > 10000) k = 10*K;
        ++k;
      }
    }
  }
}

void satoshiRansacManhattan2(const std::vector<Eigen::Vector3d> & N, const Eigen::Vector3d & n1,
  Eigen::Vector3d & M1, Eigen::Vector3d & M2) {
  const int m = N.size();


  volatile double maxInliers = 0, K = 1.0e5;
  volatile int k = 0;

  #pragma omp parallel shared (k, K, maxInliers, N, n1, M1, M2)
  {
    std::random_device seed;
    std::mt19937_64 gen(seed());
    std::uniform_int_distribution<int> dist(0, m - 1);
    for (int i = 0; i < 5000; ++i)
      dist(gen);

    while (k < K) {
      // random sampling
      int randomIndex = dist(gen);
      // compute the model parameters
      const Eigen::Vector3d & nest = N[randomIndex];

      const Eigen::Vector3d nest2 = nest.cross(n1);

      // counting inliers and outliers
      double numInliers = 0;
      Eigen::Vector3d average = Eigen::Vector3d::Zero();
      Eigen::Vector3d x;
      for(auto & n : N) {
        if(std::min(std::acos(std::abs(nest.dot(n))), std::acos(std::abs(nest2.dot(n)))) < 0.02) {
          if(std::acos(std::abs(nest.dot(n))) < 0.02) {
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

      #pragma omp crtical
      {
        if(numInliers > maxInliers) {
          maxInliers = numInliers;

          average /= average.norm();
          M1 = average;
          M2 = average.cross(n1);

          double w = (maxInliers-3)/m;
          double p = std::max(0.001, std::pow(w,3));
          K = log(1-0.999)/log(1-p);
        }

        if(k > 10000) k = 10*K;
        ++k;
      }
    }
  }
}

void getMajorAngles(const Eigen::Vector3d & M, std::vector<Eigen::Matrix3d> & R) {
  R[0] = getRotationMatrix(Eigen::Vector3d::UnitX(), M);
  R[1] = getRotationMatrix(Eigen::Vector3d::UnitY(), M);
  R[2] = getRotationMatrix(-1.0*Eigen::Vector3d::UnitX(), M);
  R[3] = getRotationMatrix(-1.0*Eigen::Vector3d::UnitY(), M);
}


static Eigen::Matrix3d crossProductMatrix(const Eigen::Vector3d & vector) {
  Eigen::Matrix3d scratch = Eigen::Matrix3d::Zero();

  scratch(1,0) = vector[2];
  scratch(2,0) = -vector[1];
  scratch(0,1) = -vector[2];
  scratch(2,1) = vector[0];
  scratch(0,2) = vector[1];
  scratch(1,2) = -vector[0];

  return scratch;
}


Eigen::Matrix3d getRotationMatrix(const Eigen::Vector3d & end, const Eigen::Vector3d & start) {

  if(std::acos(std::abs(start.dot(end))) < 0.005) {
    if( start.dot(end) > 0)
      return Eigen::Matrix3d::Identity();
    if(start.dot(end) < 0)
      return -1.0*Eigen::Matrix3d::Identity();
  }

  Eigen::Vector3d v = start.cross(end);
  double s = v.dot(v);
  double c = start.dot(end);

  Eigen::Matrix3d vx = crossProductMatrix(v);

  Eigen::Matrix3d out = Eigen::Matrix3d::Identity() + vx + vx*vx*(1 - c)/s;

  out(2, 2) = 1.0;
  out(0, 2) = 0;
  out(1, 2) = 0;
  out(2, 0) = 0;
  out(2, 1) = 0;

  return out;
}