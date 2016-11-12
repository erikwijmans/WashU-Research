#pragma once
#ifndef SCAN_TYPEDEFS_HPP
#define SCAN_TYPEDEFS_HPP

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/StdVector>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <unordered_map>

#include "scan_gflags.h"
#include <omp.h>

#define NUM_ROTS 4

constexpr double PI = 3.14159265358979323846;
constexpr double maxPhi = 2.61946;

class BuildingScale {
private:
  double scale = -1;

public:
  double getScale();
  void update(double scale);
};

extern BuildingScale buildingScale;

namespace std {
template <> struct hash<std::vector<int>> {
  static constexpr double A = 1.6180339887498948482 * 1e5;
  hash<double> h;
  size_t operator()(const std::vector<int> &k) const;
};
} // std

namespace place {
class edge {
public:
  double pA, feA, feB, fx;
  double w, wSignificance;
  double panoW, panoSignificance;
  double distance;
  double hWeight = 0.0;
  int numSim = 0, numDiff = 0;
  edge()
      : pA{0}, feA{0}, feB{0}, fx{0}, w{0}, wSignificance{0}, panoW{0},
        panoSignificance{0}, distance{0} {};
  edge(double pA, double feA, double feB, double fx, double w,
       double wSignificance)
      : pA{pA}, feA{feA}, feB{feB}, w{w}, wSignificance{wSignificance},
        panoW{0}, panoSignificance{0}, distance{0} {};
  double getWeight() const;
  friend std::ostream &operator<<(std::ostream &os, const place::edge &print);
};

class Wall {
public:
  struct side {
    std::vector<int> incident, owners;
    double distance = 10000.0;
    double weight = 0.0;
    int count = 0;
  };
  Wall() : normal{nullptr}, s{nullptr} {};
  ~Wall();
  void init(const Eigen::Vector2d &n);
  side &getSide(const Eigen::Vector2d &ray);
  const Eigen::Vector2d &getNormal();
  void changeNormal(const Eigen::Vector2d &n);

private:
  Eigen::Vector2d *normal;
  side *s;
};

struct H {
  std::vector<int> incident;
  std::vector<double> weights;
};
} // place

namespace Eigen {
typedef Array<Eigen::Vector3f, Dynamic, Dynamic, RowMajor> ArrayXV3f;
typedef Array<place::edge, Dynamic, Dynamic> MatrixXE;
typedef Array<place::Wall::side, Dynamic, Dynamic> ArrayXH;
typedef Array<place::H, Dynamic, Dynamic> ArrayXH2;
typedef Matrix<char, Dynamic, Dynamic> MatrixXb;
typedef Matrix<int, Dynamic, Dynamic, RowMajor> RowMatrixXi;
typedef Matrix<char, Dynamic, Dynamic, RowMajor> RowMatrixXb;
typedef Matrix<double, Dynamic, Dynamic, RowMajor> RowMatrixXd;
typedef Matrix<float, Dynamic, Dynamic, RowMajor> RowMatrixXf;
} // Eigen

struct SHOT1344WithXYZ {
  std::shared_ptr<Eigen::VectorXf> descriptor;
  Eigen::Vector3d position;

  SHOT1344WithXYZ() : descriptor{std::make_shared<Eigen::VectorXf>(1344)} {};

  void writeToFile(std::ofstream &out);
  void loadFromFile(std::ifstream &in);
};

struct SPARSE352WithXYZ {
  typedef Eigen::SparseVector<float> VecType;
  std::shared_ptr<VecType> descriptor;
  Eigen::Vector3d position;

  SPARSE352WithXYZ() : descriptor{std::make_shared<VecType>(352)} {};
  void writeToFile(std::ofstream &out);
  void loadFromFile(std::ifstream &in);
};

struct SPARSE1344WithXYZ {
  typedef Eigen::SparseVector<float> VecType;
  std::shared_ptr<VecType> descriptor;
  Eigen::Vector3d position;

  SPARSE1344WithXYZ() : descriptor{std::make_shared<VecType>(1344)} {};

  void writeToFile(std::ofstream &out);
  void loadFromFile(std::ifstream &in);
};

namespace scan {
struct PointXYZRGBA {
  Eigen::Vector3f point;
  float intensity;
  unsigned char rgb[3];

  void writeToFile(std::ofstream &out);
  void loadFromFile(std::ifstream &in);
};

struct PointXYZRGB {
  Eigen::Vector3f point;
  unsigned char rgb[3];

  void writeToFile(std::ofstream &out);
  void loadFromFile(std::ifstream &in);
};
} // scan

namespace place {
struct posInfo {
  double score = 0;
  double scanFP = 0;
  double fpScan = 0;
  double doorCount = 0;
  double doorUxp = 0;
  int rotation = 0;
  int fpPixels = 0;
  int scanPixels = 0;
  int x = 0;
  int y = 0;

  friend std::ostream &operator<<(std::ostream &os,
                                  const place::posInfo *print);
  friend std::ostream &operator<<(std::ostream &os,
                                  const place::posInfo &print);

  bool operator==(const posInfo &o) const {
    return o.rotation == rotation && o.x == x && o.y == y;
  }
};

struct ExclusionMap {
  typedef Eigen::Array<const place::posInfo *, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>
      Map;
  Map *maps;
  double exclusionSize;
  int rows, cols;

  ExclusionMap(double exclusionSize, int rows, int cols);
  ExclusionMap(double exclusionSize, int rows, int cols, int numRots);
  ~ExclusionMap();

  Map &operator[](int r);
};

struct VoxelGrid {
  std::vector<Eigen::MatrixXb> v;
  Eigen::Vector3i zZ;
  size_t c;

  void writeToFile(std::ofstream &out);
  void loadFromFile(std::ifstream &in);
};

struct moreInfo {
  posInfo s;
  int scanNum;
};

struct node : public posInfo {
  double w;
  double nw;
  double hWeight = 0.0;
  int color, id;
  node(const posInfo &s, double w, double nw, int color, int id)
      : posInfo{s}, w{w}, nw{nw}, color{color}, id{id} {};
  double getWeight() const;
};

struct SelectedNode : public node {
  double agreement, norm;
  int label, numberOfCandidates;
  bool locked;
  SelectedNode(const node &o, double agreement, int label, bool locked,
               int numberOfCandidates)
      : node{o}, agreement{agreement}, norm{0}, label{label}, locked{locked},
        numberOfCandidates{numberOfCandidates} {};

  friend std::ostream &operator<<(std::ostream &os,
                                  const place::SelectedNode &p);
};

struct R2Node : public node {
  double agreement;
  bool locked;
  R2Node(const node &o, bool locked) : node{o}, agreement{0}, locked{locked} {};

  R2Node(const posInfo &s, double w, double nw, int color, int id, bool locked)
      : node{s, w, nw, color, id}, locked{locked} {};
  R2Node(const SelectedNode &s)
      : node{s}, agreement{s.agreement}, locked{s.locked} {};
};

struct rect {
  int X1;
  int Y1;
  int X2;
  int Y2;
};

struct MetaData {
  Eigen::Vector3i zZ;
  int x, y, z;
  double vox, s;
  void writeToFile(std::ofstream &out);
  void loadFromFile(std::ifstream &in);
};

class cube {
public:
  int X1, Y1, Z1;
  int X2, Y2, Z2;
  cube() : X1{0}, Y1{0}, Z1{0}, X2{0}, Y2{0}, Z2{0} {};
  int volume() {
    const int width = this->X2 - this->X1;
    const int length = this->Y2 - this->Y1;
    const int height = this->Z2 - this->Z1;
    return width * length * height;
  };

  friend std::ostream &operator<<(std::ostream &os, const place::cube &print);
};

struct Panorama {
  static constexpr double ScalingFactor = 1.2599210498948732;

  double floorCoord;
  std::vector<cv::Mat> imgs;
  Eigen::RowMatrixXf rMap;
  std::vector<cv::Point2f> keypoints;
  Eigen::ArrayXV3f surfaceNormals;
  void writeToFile(const std::string &imgName, const std::string &dataName);
  void loadFromFile(const std::string &imgName, const std::string &dataName);
  Panorama();

  const cv::Mat &operator[](int n);
};

template <class It, class UnaryFunc, class UnaryPredicate>
std::tuple<double, double> aveAndStdev(It first, It last, UnaryFunc selector,
                                       UnaryPredicate filter) {
  double average = 0;
  int count = 0;
  std::for_each(first, last, [&](auto &e) {
    if (filter(e)) {
      auto val = selector(e);
      if (Eigen::numext::isfinite(val)) {
        average += val;
        ++count;
      }
    }
  });
  average /= count;

  double sigma = 0;
  std::for_each(first, last, [&](auto &e) {
    if (filter(e)) {
      auto val = selector(e);
      if (Eigen::numext::isfinite(val))
        sigma += (val - average) * (val - average);
    }
  });
  sigma /= count - 1;
  sigma = std::sqrt(sigma);
  return std::make_tuple(average, sigma);
}

template <class It, class UnaryFunc>
std::tuple<double, double> aveAndStdev(It first, It last, UnaryFunc selector) {
  return aveAndStdev(first, last, selector, [](auto &e) { return true; });
}

template <class It> std::tuple<double, double> aveAndStdev(It first, It last) {
  return aveAndStdev(first, last, [](auto &e) { return e; });
}

struct Door {
  Eigen::Vector3d corner;
  Eigen::Vector3d xAxis;
  Eigen::Vector3d zAxis;
  double h, w;
  Door(){};
  Door(const Eigen::Vector3d &c, const Eigen::Vector3d &x,
       const Eigen::Vector3d &z, double h, double w)
      : corner{c}, xAxis{x}, zAxis{z}, h{h}, w{w} {};

  void writeToFile(std::ofstream &out) const;
  void loadFromFile(std::ifstream &in);
};

} // place

double sigmoidWeight(double seen, double expected);

double gaussianWeight(const Eigen::Array2d &pos, const Eigen::Array2d &s);

namespace std {
template <> struct hash<place::posInfo> {
  static constexpr double A = 1.6180339887498948482 * 1e5;
  hash<double> h;
  size_t operator()(const place::posInfo &e) const;
};
} // std

cv::Vec3b randomColor();

namespace cv {
int rectshow(const std::string &name, const cv::Mat &img);
int rectshow(const cv::Mat &img);
} // cv

template <typename MatrixType>
void saveMatrixAsSparse(const MatrixType &mat, std::ofstream &out) {
  typedef typename MatrixType::Scalar Scalar;
  int numNonZeros = 0, rows = mat.rows(), cols = mat.cols();
  const Scalar *dataPtr = mat.data();
  for (int i = 0; i < mat.size(); ++i)
    if (*(dataPtr + i))
      ++numNonZeros;

  out.write(reinterpret_cast<const char *>(&numNonZeros), sizeof(numNonZeros));
  out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
  out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));

  for (int i = 0; i < mat.size(); ++i) {
    if (*(dataPtr + i)) {
      out.write(reinterpret_cast<const char *>(&i), sizeof(i));
      out.write(reinterpret_cast<const char *>(dataPtr + i), sizeof(Scalar));
    }
  }
}

template <typename MatrixType>
void loadMatrixFromSparse(MatrixType &mat, std::ifstream &in) {
  typedef typename MatrixType::Scalar Scalar;
  int numNonZeros, rows, cols;

  in.read(reinterpret_cast<char *>(&numNonZeros), sizeof(numNonZeros));
  in.read(reinterpret_cast<char *>(&rows), sizeof(rows));
  in.read(reinterpret_cast<char *>(&cols), sizeof(cols));

  mat = MatrixType::Zero(rows, cols);
  Scalar *dataPtr = mat.data();

  for (int i = 0; i < numNonZeros; ++i) {
    int index;
    in.read(reinterpret_cast<char *>(&index), sizeof(index));
    in.read(reinterpret_cast<char *>(dataPtr + index), sizeof(Scalar));
  }
}

template <typename SparseMatrixType>
void saveSparseMatrix(SparseMatrixType &mat, std::ofstream &out) {
  typedef typename SparseMatrixType::Scalar Scalar;

  int rows = mat.rows(), cols = mat.cols(), numNonZeros = mat.nonZeros();
  out.write(reinterpret_cast<const char *>(&numNonZeros), sizeof(numNonZeros));
  out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
  out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));

  for (int i = 0; i < mat.outerSize(); ++i) {
    for (typename SparseMatrixType::InnerIterator it(mat, i); it; ++it) {
      int index = it.col() * rows + it.row();
      Scalar value = it.value();
      out.write(reinterpret_cast<const char *>(&index), sizeof(index));
      out.write(reinterpret_cast<const char *>(&value), sizeof(Scalar));
    }
  }
}

template <typename SparseMatrixType>
void loadSparseMatrix(SparseMatrixType &mat, std::ifstream &in) {
  typedef typename SparseMatrixType::Scalar Scalar;
  typedef Eigen::Triplet<Scalar> TripType;

  int rows, cols, numNonZeros;
  in.read(reinterpret_cast<char *>(&numNonZeros), sizeof(numNonZeros));
  in.read(reinterpret_cast<char *>(&rows), sizeof(rows));
  in.read(reinterpret_cast<char *>(&cols), sizeof(cols));
  mat.resize(rows, cols);
  mat.reserve(numNonZeros);
  std::vector<TripType> tripletList;
  tripletList.reserve(numNonZeros);

  for (int i = 0; i < numNonZeros; ++i) {
    int index;
    Scalar value;
    in.read(reinterpret_cast<char *>(&index), sizeof(index));
    in.read(reinterpret_cast<char *>(&value), sizeof(Scalar));
    int col = floor(index / rows);
    int row = index % rows;
    tripletList.push_back(TripType(row, col, value));
  }
  mat.setFromTriplets(tripletList.begin(), tripletList.end());
}

template <typename SparseVectorType>
void saveSpareVector(const SparseVectorType &vec, std::ofstream &out) {
  typedef typename SparseVectorType::Scalar Scalar;
  int nonZeros = vec.nonZeros(), size = vec.size();
  out.write(reinterpret_cast<const char *>(&nonZeros), sizeof(nonZeros));
  out.write(reinterpret_cast<const char *>(&size), sizeof(size));
  for (int i = 0; i < vec.outerSize(); ++i) {
    for (typename SparseVectorType::InnerIterator it(vec, i); it; ++it) {
      Scalar value = it.value();
      short row = it.row();
      out.write(reinterpret_cast<const char *>(&value), sizeof(Scalar));
      out.write(reinterpret_cast<const char *>(&row), sizeof(row));
    }
  }
}

template <typename SparseVectorType>
void loadSparseVetor(SparseVectorType &vec, std::ifstream &in) {
  typedef typename SparseVectorType::Scalar Scalar;
  int nonZeros, size;
  in.read(reinterpret_cast<char *>(&nonZeros), sizeof(nonZeros));
  in.read(reinterpret_cast<char *>(&size), sizeof(size));
  vec.resize(size);
  vec.reserve(nonZeros);
  for (int i = 0; i < nonZeros; ++i) {
    Scalar value;
    short row;
    in.read(reinterpret_cast<char *>(&value), sizeof(Scalar));
    in.read(reinterpret_cast<char *>(&row), sizeof(row));
    vec.coeffRef(row) = value;
  }
}

inline bool fexists(const std::string &file) {
  std::ifstream in(file, std::ios::in);
  return in.is_open();
}

namespace std {
template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows,
          int _MaxCols>
struct hash<
    Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>> {
  hash<_Scalar> h;
  std::size_t operator()(const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options,
                                             _MaxRows, _MaxCols> &k) const {
    size_t seed = 0;
    auto dataPtr = k.data();
    for (int i = 0; i < k.size(); ++i) {
      seed ^= h(*(dataPtr + i)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};
} // std

template <unsigned N, unsigned L, typename... Targs> struct tuple_printer {
  static void print(std::ostream &os, std::tuple<Targs...> &print) {
    os << std::get<N>(print) << ", ";
    tuple_printer<N + 1, L, Targs...>::print(os, print);
  }
};

template <unsigned N, typename... Targs> struct tuple_printer<N, N, Targs...> {
  static void print(std::ostream &os, std::tuple<Targs...> &print) {
    os << std::get<N>(print);
  }
};

template <typename... Targs>
std::ostream &operator<<(std::ostream &os, std::tuple<Targs...> &print) {
  os << "(";
  tuple_printer<0, sizeof...(Targs)-1, Targs...>::print(os, print);
  os << ")";
  return os;
}

#endif // SCAN_TYPEDEFS_HPP
