#pragma once

#ifndef SCAN_TYPEDEFS_HPP
#define SCAN_TYPEDEFS_HPP

#include <boost/cstdint.hpp>
#include <boost/noncopyable.hpp>
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/StdVector>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fstream>
#include <glog/logging.h>
#include <iostream>
#include <memory>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <unordered_map>

#include <omp.h>

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

using namespace fmt::literals;

constexpr double PI = 3.14159265358979323846;

namespace std {
template <> struct hash<std::vector<int>> {
  static constexpr double A = 1.6180339887498948482 * 1e5;
  hash<double> h;
  size_t operator()(const std::vector<int> &k) const;
};

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

template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows,
          int _MaxCols>
Eigen::Array<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>
sqrt(Eigen::Array<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> &a) {
  return a.sqrt();
}
template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows,
          int _MaxCols>
Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>
sqrt(Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> &m) {
  return m.array().sqrt().matrix();
}
} // std

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

/* Light wrappers to the functional programming functions so they only take
   an iterable instead of two iterators */
namespace std {
template <class Iterable, class UnaryFunction>
UnaryFunction for_each(Iterable &i, UnaryFunction f) {
  return for_each(i.begin(), i.end(), f);
}

template <class Iterable, class UnaryPredicate>
auto remove_if(Iterable &i, UnaryPredicate p) {
  return remove_if(i.begin(), i.end(), p);
}

template <class Iterable, class Compare> void sort(Iterable &i, Compare comp) {
  sort(i.begin(), i.end(), comp);
}

template <class Iterable> void sort(Iterable &i) { sort(i.begin(), i.end()); }
} // std

namespace utils {
template <class It, class E, class UnaryFunc, class UnaryPredicate>
std::tuple<E, E> ave_and_stdev(It first, It last, E init, UnaryFunc selector,
                               UnaryPredicate filter) {

  E average = init;
  int count = 0;
  std::for_each(first, last, [&](auto &e) {
    if (filter(e)) {
      auto val = selector(e);
      average += val;
      ++count;
    }
  });
  average /= count;

  E sigma = init;
  std::for_each(first, last, [&](auto &e) {
    if (filter(e)) {
      auto val = selector(e);
      sigma += (val - average) * (val - average);
    }
  });
  sigma /= count - 1;
  sigma = std::sqrt(sigma);

  return std::make_tuple(average, sigma);
}

template <class Iterable, class E, class UnaryFunc, class UnaryPredicate>
std::tuple<E, E> ave_and_stdev(Iterable &i, E init, UnaryFunc selector,
                               UnaryPredicate filter) {
  return ave_and_stdev(i.begin(), i.end(), init, selector, filter);
}

template <class It, class E, class UnaryFunc>
std::tuple<E, E> ave_and_stdev(It first, It last, E init, UnaryFunc selector) {
  return ave_and_stdev(first, last, init, selector,
                       [](auto &e) { return true; });
}

template <class Iterable, class E, class UnaryFunc>
std::tuple<E, E> ave_and_stdev(Iterable &i, E init, UnaryFunc selector) {
  return ave_and_stdev(i.begin(), i.end(), init, selector);
}

template <class It, class E>
std::tuple<E, E> ave_and_stdev(It first, It last, E init) {
  return ave_and_stdev(first, last, init, [](auto &e) { return e; });
}

template <class Iterable, class E>
std::tuple<E, E> ave_and_stdev(Iterable &i, E init) {
  return ave_and_stdev(i.begin(), i.end(), init);
}

template <class It>
std::tuple<double, double> ave_and_stdev(It first, It last) {
  return ave_and_stdev(first, last, 0.0, [](auto &e) { return e; });
}

template <class Iterable>
std::tuple<double, double> ave_and_stdev(Iterable &i) {
  return ave_and_stdev(i.begin(), i.end(), 0.0);
}

fs::directory_iterator folder_to_iterator(const fs::path &p);
void parse_folder(const std::string &name, std::vector<fs::path> &out);
void parse_folder(const fs::path &p, std::vector<fs::path> &out);
template <class UrnaryPredicate>
void parse_folder(const fs::path &p, std::vector<fs::path> &out,
                  UrnaryPredicate filter) {
  for (auto &file : folder_to_iterator(p)) {
    if (filter(file))
      out.push_back(file);
  }
  std::sort(out);
}
template <class UrnaryPredicate>
void parse_folder(const std::string &name, std::vector<fs::path> &out,
                  UrnaryPredicate filter) {
  parse_folder(fs::path(name), out, filter);
}

std::vector<fs::path> parse_folder(const std::string &name);
std::vector<fs::path> parse_folder(const fs::path &p);
template <class UrnaryPredicate>
std::vector<fs::path> parse_folder(const fs::path &p, UrnaryPredicate filter) {
  std::vector<fs::path> out;
  parse_folder(p, out, filter);
  return out;
}
template <class UrnaryPredicate>
std::vector<fs::path> parse_folder(const std::string &name,
                                   UrnaryPredicate filter) {
  return parse_folder(fs::path(name), filter);
}

double sigmoidWeight(double seen, double expected);

cv::Vec3b randomColor();

template <class RandAccIterator, class E, class PurePairwiseFunc, class Reducer>
auto ransac(const RandAccIterator &data, E init, PurePairwiseFunc is_inlier,
            Reducer reducer) {

  const int m = data.size();

  double maxInliers = 0, K = 1e5;
  int k = 0;

  static std::random_device seed;
  static std::mt19937_64 gen(seed());
  std::uniform_int_distribution<int> dist(0, m - 1);
  auto res = init();

  while (k < K) {
    // random sampling
    int randomIndex = dist(gen);
    // compute the model parameters
    auto &est = data[randomIndex];

    // Count the number of inliers
    double numInliers = 0;
    auto average = init();
#pragma omp parallel
    {
      double privateInliers = 0;
      auto private_ave = init();
#pragma omp for nowait schedule(static)
      for (int i = 0; i < m; ++i) {
        auto &d = data[i];
        if (is_inlier(d, est)) {
          ++privateInliers;
          private_ave = reducer(private_ave, d, est);
        }
      }

#pragma omp critical
      {
        average += private_ave;
        numInliers += privateInliers;
      }
    }

    if (numInliers > maxInliers) {
      maxInliers = numInliers;

      res = average / numInliers;
      // NB: Ransac formula to check for consensus
      double w = (numInliers - 3) / m;
      double p = std::max(0.001, std::pow(w, 3));
      K = log(1 - 0.999) / log(1 - p);
    }
    ++k;
  }

  return res;
}

template <class RandAccIterator, class E, class PurePairwiseFunc>
auto ransac(const RandAccIterator &data, E init, PurePairwiseFunc is_inlier) {
  return ransac(data, init, is_inlier, [](const auto &ave, const auto &n,
                                          const auto &est) { return ave + n; });
}

class progress_display : private boost::noncopyable {
public:
  explicit progress_display(unsigned long expected_count_,
                            std::ostream &os = std::cout,
                            const std::string &s1 = "\n",
                            const std::string &s2 = "",
                            const std::string &s3 = "");

  void restart(unsigned long expected_count_);

  unsigned long operator+=(unsigned long increment);

  unsigned long operator++() { return operator+=(1); }
  unsigned long count() const { return _count; }
  unsigned long expected_count() const { return _expected_count; }

private:
  std::ostream &m_os;     // may not be present in all imps
  const std::string m_s1; // string is more general, safer than
  const std::string m_s2; //  const char *, and efficiency or size are
  const std::string m_s3; //  not issues

  unsigned long _count, _expected_count, _next_tic_count;
  unsigned int _tic;
  void display_tic();
};

inline std::ifstream open(const fs::path &n,
                          std::ios_base::openmode mode = std::ios_base::in) {
  CHECK(fs::exists(n)) << "Could not open: " << n;
  return std::ifstream(n.string(), mode);
}
} // utils

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
  tuple_printer<0, sizeof...(Targs) - 1, Targs...>::print(os, print);
  os << ")";
  return os;
}

template <typename... Targs>
std::ostream &operator<<(std::ostream &os, std::tuple<Targs...> &&print) {
  os << "(";
  tuple_printer<0, sizeof...(Targs) - 1, Targs...>::print(os, print);
  os << ")";
  return os;
}

#include "RGBD_utils.hpp"

#endif // SCAN_TYPEDEFS_HPP
