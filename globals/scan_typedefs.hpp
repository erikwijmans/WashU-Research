#pragma once
#ifndef SCAN_TYPEDEFS_HPP
#define SCAN_TYPEDEFS_HPP

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <eigen3/Eigen/StdVector>
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/Sparse>
#include <fstream>
#include <memory>
#include <omp.h>

#define NUM_ROTS 4

const double PI = 3.14159265358979323846;
const double maxPhi = 2.61946;

namespace place {
  class edge {
    public:
      double pA, feA, feB, fx;
      double w, shotW;
      double panoW;
      edge () : pA {0}, feA {0}, feB {0}, fx {0}, w {0}, shotW {0}, panoW{0}
      {};
      edge (double pA, double feA, double feB, double fx, double w,
        double shotW) : pA {pA}, feA {feA}, feB {feB}, w {w}, shotW {shotW},
        panoW {0}
      {};
  };

  typedef struct {
    double weight = 0.0;
    std::vector<int> incident;
  } hOrder;
} // place

namespace Eigen {
  typedef Array<Eigen::Vector3f, Dynamic, Dynamic, RowMajor> ArrayXV3f;
  typedef Array<place::edge, Dynamic, Dynamic> MatrixXE;
  typedef Array<place::hOrder, Dynamic, Dynamic> ArrayXH;
  typedef Matrix<char, Dynamic, Dynamic> MatrixXb;
  typedef Matrix< int, Dynamic, Dynamic, RowMajor > RowMatrixXi;
  typedef Matrix<char, Dynamic, Dynamic, RowMajor> RowMatrixXb;
  typedef Matrix<double, Dynamic, Dynamic, RowMajor> RowMatrixXd;
  typedef Matrix<float, Dynamic, Dynamic, RowMajor> RowMatrixXf;
} // Eigen

typedef struct SHOT1344WithXYZ {
  std::shared_ptr<Eigen::VectorXf> descriptor;
  Eigen::Vector3d position;

  SHOT1344WithXYZ() : descriptor {std::make_shared<Eigen::VectorXf> (1344)}
  {
  };

  void writeToFile(std::ofstream & out) {
    out.write(reinterpret_cast<const char *>(descriptor->data()),
      descriptor->size()*sizeof(float));
    out.write(reinterpret_cast<const char *>(position.data()), sizeof(position));
  }

  void loadFromFile(std::ifstream & in) {
    in.read(reinterpret_cast<char *>(descriptor->data()),
      descriptor->size()*sizeof(float));
    in.read(reinterpret_cast<char *>(position.data()), sizeof(position));
  }

} SHOT1344WithXYZ;

template<typename MatrixType>
void saveMatrixAsSparse(const MatrixType & mat, std::ofstream & out) {
  typedef typename MatrixType::Scalar Scalar;
  int numNonZeros = 0, rows = mat.rows(), cols = mat.cols();
  const Scalar * dataPtr = mat.data();
  for (int i = 0; i < mat.size(); ++i)
    if (*(dataPtr + i)) ++numNonZeros;

  out.write(reinterpret_cast<const char *>(&numNonZeros), sizeof(numNonZeros));
  out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
  out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));

  for (int i = 0; i < mat.size(); ++i) {
    if (*(dataPtr + i)) {
      out.write(reinterpret_cast<const char *>(&i),sizeof(i));
      out.write(reinterpret_cast<const char *>(dataPtr + i), sizeof(Scalar));
    }
  }
}


template<typename MatrixType>
void loadMatrixFromSparse(MatrixType & mat, std::ifstream & in) {
  typedef typename MatrixType::Scalar Scalar;
  int numNonZeros, rows, cols;

  in.read(reinterpret_cast<char *>(&numNonZeros), sizeof(numNonZeros));
  in.read(reinterpret_cast<char *>(&rows), sizeof(rows));
  in.read(reinterpret_cast<char *>(&cols), sizeof(cols));

  mat = MatrixType::Zero(rows, cols);
  Scalar * dataPtr = mat.data();

  for (int i = 0; i < numNonZeros; ++i) {
    int index;
    in.read(reinterpret_cast<char *>(&index), sizeof(index));
    in.read(reinterpret_cast<char *>(dataPtr + index), sizeof(Scalar));
  }
}

template<typename SparseMatrixType>
void saveSparseMatrix(SparseMatrixType & mat, std::ofstream & out) {
  typedef typename SparseMatrixType::Scalar Scalar;

  int rows = mat.rows(), cols = mat.cols(), numNonZeros = mat.nonZeros();
  out.write(reinterpret_cast<const char *>(&numNonZeros), sizeof(numNonZeros));
  out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
  out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));

  for (int i = 0; mat.outerSize(); ++i) {
    for (typename SparseMatrixType::InnerIterator it (mat, i); it; ++it) {
      int index = it.col()*rows + it.row();
      Scalar value = it.value();
      out.write(reinterpret_cast<const char *>(&index),sizeof(index));
      out.write(reinterpret_cast<const char *>(&value), sizeof(Scalar));
    }
  }
}

template<typename SparseMatrixType>
void loadSparseMatrix(SparseMatrixType & mat, std::ifstream & in) {
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
    int col = floor(index/rows);
    int row = index % rows;
    tripletList.push_back(TripType (row, col, value));
  }
  mat.setFromTriplets(tripletList.begin(), tripletList.end());
}

template<typename SparseVectorType>
void saveSpareVector(const SparseVectorType & vec, std::ofstream & out) {
  typedef typename SparseVectorType::Scalar Scalar;
  int nonZeros = vec.nonZeros(), size = vec.size();
  out.write(reinterpret_cast<const char *>(&nonZeros), sizeof(nonZeros));
  out.write(reinterpret_cast<const char *>(&size), sizeof(size));
  for (int i = 0; i < vec.outerSize(); ++i) {
    for (typename SparseVectorType::InnerIterator it (vec, i); it; ++it) {
      Scalar value = it.value();
      short row = it.row();
      out.write(reinterpret_cast<const char *>(&value), sizeof(Scalar));
      out.write(reinterpret_cast<const char *>(&row), sizeof(row));
    }
  }
}

template<typename SparseVectorType>
void loadSparseVetor(SparseVectorType & vec, std::ifstream & in) {
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

typedef struct SPARSE352WithXYZ {
  typedef Eigen::SparseVector<float> VecType;
  std::shared_ptr<VecType> descriptor;
  Eigen::Vector3d position;

  SPARSE352WithXYZ() : descriptor {std::make_shared<VecType> (352)}
  {
  };

  void writeToFile(std::ofstream & out) {
    saveSpareVector(*descriptor, out);
    out.write(reinterpret_cast<const char *>(position.data()), sizeof(position));
  }

  void loadFromFile(std::ifstream & in) {
    loadSparseVetor(*descriptor, in);
    in.read(reinterpret_cast<char *>(position.data()), sizeof(position));
  }

} SPARSE352WithXYZ;

typedef struct SPARSE1344WithXYZ {
  typedef Eigen::SparseVector<float> VecType;
  std::shared_ptr<VecType> descriptor;
  Eigen::Vector3d position;

  SPARSE1344WithXYZ() : descriptor {std::make_shared<VecType> (1344)}
  {
  };

  void writeToFile(std::ofstream & out) {
    saveSpareVector(*descriptor, out);
    out.write(reinterpret_cast<const char *>(position.data()), sizeof(position));
  }

  void loadFromFile(std::ifstream & in) {
    loadSparseVetor(*descriptor, in);
    in.read(reinterpret_cast<char *>(position.data()), sizeof(position));
  }

} SPARSE1344WithXYZ;

namespace scan {
  typedef struct {
    Eigen::Vector3f point;
    float intensity;
    unsigned char rgb [3];

    void writeToFile(std::ofstream & out) {
      out.write(reinterpret_cast<const char *>(point.data()), sizeof(point));
      out.write(reinterpret_cast<const char *>(&intensity), sizeof(intensity));
      out.write(reinterpret_cast<const char *>(rgb), 3*sizeof(char));
    }
    void loadFromFile(std::ifstream & in) {
      in.read(reinterpret_cast<char *>(point.data()), sizeof(point));
      in.read(reinterpret_cast<char *>(&intensity), sizeof(intensity));
      in.read(reinterpret_cast<char *>(rgb), 3*sizeof(char));
    }

  } PointXYZRGBA;

  typedef struct {
    Eigen::Vector3f point;
    unsigned char rgb [3];

    void writeToFile(std::ofstream & out) {
      out.write(reinterpret_cast<const char *>(point.data()), sizeof(point));
      out.write(reinterpret_cast<const char *>(rgb), 3*sizeof(char));
    }
    void loadFromFile(std::ifstream & in) {
      in.read(reinterpret_cast<char *>(point.data()), sizeof(point));
      in.read(reinterpret_cast<char *>(rgb), 3*sizeof(char));
    }
  } PointXYZRGB;
} // scan

namespace place {
  typedef struct {
    double score = 0;
    double scanFP = 0;
    double fpScan = 0;
    int rotation = 0;
    int fpPixels = 0;
    int scanPixels = 0;
    int x = 0;
    int y = 0;
  } posInfo;

  typedef struct {
    const posInfo *** maps;
    double exclusionX, exclusionY;
    int rows, cols;
  } exclusionMap;

  typedef struct {
    std::vector<Eigen::MatrixXb> v;
    Eigen::Vector3i zZ;
    size_t c;

    void writeToFile(std::ofstream & out) {
      int numZ = v.size();
      out.write(reinterpret_cast<const char *>(&numZ), sizeof(numZ));
      for (int k = 0; k < numZ; ++k) {
        saveMatrixAsSparse(v[k], out);
      }
      out.write(reinterpret_cast<const char *>(zZ.data()), sizeof(zZ));
      out.write(reinterpret_cast<const char *>(&c), sizeof(c));
    };

    void loadFromFile(std::ifstream & in) {
      int numZ;
      in.read(reinterpret_cast<char *>(&numZ), sizeof(numZ));
      v.resize(numZ);
      for (int k = 0; k < numZ; ++k) {
        loadMatrixFromSparse(v[k], in);
      }
      in.read(reinterpret_cast<char *>(zZ.data()), sizeof(zZ));
      in.read(reinterpret_cast<char *>(&c), sizeof(c));

    }
  } voxelGrid;

  typedef struct {
    posInfo s;
    int scanNum;
  } moreInfo;

  typedef struct {
    posInfo s;
    double w;
    double nw;
    int color;
  } node;

  typedef struct {
    int X1;
    int Y1;
    int X2;
    int Y2;
  } rect;

  typedef struct {
    Eigen::Vector3i zZ;
    int x, y, z;
    double vox, s;
    void writeToFile(std::ofstream & out) {
      out.write(reinterpret_cast<const char *>(zZ.data()), sizeof(zZ));
      out.write(reinterpret_cast<const char *>(&x), sizeof(x));
      out.write(reinterpret_cast<const char *>(&y), sizeof(y));
      out.write(reinterpret_cast<const char *>(&z), sizeof(z));
      out.write(reinterpret_cast<const char *>(&vox), sizeof(vox));
      out.write(reinterpret_cast<const char *>(&s), sizeof(s));
    };

    void loadFromFile(std::ifstream & in) {
      in.read(reinterpret_cast<char *>(zZ.data()), sizeof(zZ));
      in.read(reinterpret_cast<char *>(&x), sizeof(x));
      in.read(reinterpret_cast<char *>(&y), sizeof(y));
      in.read(reinterpret_cast<char *>(&z), sizeof(z));
      in.read(reinterpret_cast<char *>(&vox), sizeof(vox));
      in.read(reinterpret_cast<char *>(&s), sizeof(s));
    };
  } metaData;

  class cube {
    public:
      int X1, Y1, Z1;
      int X2, Y2, Z2;
      cube() : X1 {0}, X2 {0}, Y1 {0}, Y2 {0}, Z1 {0}, Z2 {0}
      {};
      int volume() {
        const int width = this->X2 - this->X1;
        const int length = this->Y2 - this->Y1;
        const int height = this->Z2 - this->Z1;
        return width*length*height;
      };
  };

  typedef struct Panorama {
    std::vector<cv::Mat> imgs;
    Eigen::RowMatrixXf rMap;
    std::vector<cv::Point2f> keypoints;
    Eigen::ArrayXV3f surfaceNormals;
    void writeToFile(const std::string & imgName,
      const std::string & dataName) {
      cv::imwrite(imgName, imgs[0]);

      std::ofstream out (dataName, std::ios::out | std::ios::binary);
      int rows = rMap.rows();
      int cols = rMap.cols();
      out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
      out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
      const float * dataPtr = rMap.data();
      for (int i = 0; i < rMap.size(); ++i)
        out.write(reinterpret_cast<const char *>(dataPtr + i), sizeof(float));

      const int numKeypoints = keypoints.size();
      out.write(reinterpret_cast<const char *>(&numKeypoints), sizeof(numKeypoints));
      for (auto & kp : keypoints) {
        out.write(reinterpret_cast<const char *>(&kp.x), sizeof(float));
        out.write(reinterpret_cast<const char *>(&kp.y), sizeof(float));
      }

      rows = surfaceNormals.rows();
      cols = surfaceNormals.cols();
      out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
      out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
      const auto * nPtr = surfaceNormals.data();
      for (int i = 0; i < surfaceNormals.size(); ++i) {
        out.write(reinterpret_cast<const char *>((nPtr + i)->data()),
          3*sizeof(float));
      }

      out.close();
    }

    void loadFromFile(const std::string & imgName,
      const std::string & dataName) {
      imgs.resize(1);
      imgs[0] = cv::imread(imgName);

      int rows, cols, numKeypoints;
      std::ifstream in (dataName, std::ios::in | std::ios::binary);
      in.read(reinterpret_cast<char *>(&rows), sizeof(rows));
      in.read(reinterpret_cast<char *>(&cols), sizeof(cols));
      rMap.resize(rows, cols);
      float * dataPtr = rMap.data();
      for (int i = 0; i < rMap.size(); ++i)
        in.read(reinterpret_cast<char *>(dataPtr + i), sizeof(float));

      in.read(reinterpret_cast<char *>(&numKeypoints), sizeof(numKeypoints));
      keypoints.resize(numKeypoints);
      for (auto & kp : keypoints) {
        in.read(reinterpret_cast<char *>(&kp.x), sizeof(float));
        in.read(reinterpret_cast<char *>(&kp.y), sizeof(float));
      }
      in.read(reinterpret_cast<char *>(&rows), sizeof(rows));
      in.read(reinterpret_cast<char *>(&cols), sizeof(cols));
      surfaceNormals.resize(rows, cols);
      auto * nPtr = surfaceNormals.data();
      for (int i = 0; i < surfaceNormals.size(); ++i) {
        in.read(reinterpret_cast<char *>((nPtr + i)->data()),
          3*sizeof(float));
      }
      in.close();
    }

    const cv::Mat & operator[](int n) {
      if (imgs.size() > n && imgs[n].data) {
        return imgs[n];
      } else{
        #pragma omp critical
        {
          if (imgs.size() <= n)
            imgs.resize(n + 1);

          if (!imgs[n].data) {
            const double scale = pow(2, -n/2.0);
            cv::resize(imgs[0], imgs[n], cv::Size(), scale, scale, CV_INTER_AREA);
          }
        }
        return imgs[n];
      }
    }
  } Panorama;
} // place

static std::ostream & operator<<(std::ostream & os, const place::cube & print) {
  os << "(" << print.X1 << ", " << print.Y1 << ", " << print.Z1 << ")" << std::endl;
  os << "      " << "(" << print.X2 << ", " << print.Y2 << ", " << print.Z2 <<  ")";
  return os;
}

static std::ostream & operator<<(std::ostream & os, const place::edge & print) {
  os << "edge: " << print.w << " shot: " << print.shotW;
  os << "  pano: " << print.panoW << std::endl;
  os << print.pA << "  " << print.feA << std::endl;
  os << print.fx << "  " << print.feB;
  return os;
}

static std::ostream & operator<<(std::ostream & os, const place::posInfo * print) {
  os << print->score <<"      " << print->x << "      "
    << print->y << "      " << print->rotation << std::endl;
  os << print->scanFP << "      " << print->fpScan << std::endl;
  os << print->scanPixels << "    " << print->fpPixels;
  return os;
}

static std::ostream & operator<<(std::ostream & os, const place::posInfo & print) {
  os << &print;
  return os;
}

#endif // SCAN_TYPEDEFS_HPP
