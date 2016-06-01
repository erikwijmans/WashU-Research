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
#include <unordered_map>

#include <omp.h>

#define NUM_ROTS 4

constexpr double PI = 3.14159265358979323846;
constexpr double maxPhi = 2.61946;

typedef struct no_case_hash {
  std::hash<std::string> hasher;
  std::locale loc;
  size_t operator()(const std::string & s) const;
  std::string lower_case (const std::string & s) const;
} no_case_hash;

extern const std::unordered_map<std::string, double,
  no_case_hash> buildingToScale;

namespace std {
  template <>
  struct hash<std::vector<int> >
  {
    static constexpr double A = 1.6180339887498948482*1e5;
    hash<double> h;
    size_t operator()(const std::vector<int> & k) const;
  };
} // std

namespace place {
  class edge {
    public:
      double pA, feA, feB, fx;
      double w, wSignificance;
      double panoW, panoSignificance;
      int numSim = 0, numDiff = 0;
      edge () : pA {0}, feA {0}, feB {0}, fx {0}, w {0},
      wSignificance {0}, panoW{0}, panoSignificance {0}
      {};
      edge (double pA, double feA, double feB, double fx, double w,
        double wSignificance) : pA {pA}, feA {feA}, feB {feB}, w {w}, wSignificance {wSignificance},
        panoW {0}, panoSignificance {0}
      {};
      friend std::ostream & operator<<(std::ostream & os, const place::edge & print);
  };

  typedef struct {
    std::vector<int> incident;
    double weight = 0.0;
    int count = 0;
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

  void writeToFile(std::ofstream & out);
  void loadFromFile(std::ifstream & in);

} SHOT1344WithXYZ;

typedef struct SPARSE352WithXYZ {
  typedef Eigen::SparseVector<float> VecType;
  std::shared_ptr<VecType> descriptor;
  Eigen::Vector3d position;

  SPARSE352WithXYZ() : descriptor {std::make_shared<VecType> (352)}
  {
  };
  void writeToFile(std::ofstream & out);
  void loadFromFile(std::ifstream & in);

} SPARSE352WithXYZ;

typedef struct SPARSE1344WithXYZ {
  typedef Eigen::SparseVector<float> VecType;
  std::shared_ptr<VecType> descriptor;
  Eigen::Vector3d position;

  SPARSE1344WithXYZ() : descriptor {std::make_shared<VecType> (1344)}
  {
  };

  void writeToFile(std::ofstream & out);
  void loadFromFile(std::ifstream & in);

} SPARSE1344WithXYZ;

namespace scan {
  typedef struct PointXYZRGBA {
    Eigen::Vector3f point;
    float intensity;
    unsigned char rgb [3];

    void writeToFile(std::ofstream & out);
    void loadFromFile(std::ifstream & in);
  } PointXYZRGBA;

  typedef struct PointXYZRGB {
    Eigen::Vector3f point;
    unsigned char rgb [3];

    void writeToFile(std::ofstream & out);
    void loadFromFile(std::ifstream & in);
  } PointXYZRGB;
} // scan

namespace place {
  typedef struct posInfo {
    double score = 0;
    double scanFP = 0;
    double fpScan = 0;
    int rotation = 0;
    int fpPixels = 0;
    int scanPixels = 0;
    int x = 0;
    int y = 0;

    friend std::ostream & operator<<(std::ostream & os, const place::posInfo * print);
    friend std::ostream & operator<<(std::ostream & os, const place::posInfo & print);
  } posInfo;

  typedef struct {
    const posInfo *** maps;
    double exclusionX, exclusionY;
    int rows, cols;
  } exclusionMap;

  typedef struct VoxelGrid {
    std::vector<Eigen::MatrixXb> v;
    Eigen::Vector3i zZ;
    size_t c;

    void writeToFile(std::ofstream & out);
    void loadFromFile(std::ifstream & in);
  } VoxelGrid;

  typedef struct {
    posInfo s;
    int scanNum;
  } moreInfo;

  typedef struct node {
    posInfo s;
    double w;
    double nw;
    int color, pos, groupSize;
    node (const posInfo & s, double w, double nw, int color, int pos) :
      s {s}, w {w}, nw {nw}, color {color}, pos {pos}
      {};
  } node;

  typedef struct SelectedNode : public node {
    double agreement;
    int label;
    bool locked;
    SelectedNode (const node & o, double agreement,
      int label, bool locked) :
      node {o}, agreement {agreement}, locked {locked},
      label {label}
      {};

    friend std::ostream & operator<<(std::ostream & os,
                          const place::SelectedNode & p);
  } SelectedNode;

  typedef struct R2Node : public node {
    bool locked;
    R2Node (const node & o, bool locked) :
      node {o}, locked {locked}
      {};

    R2Node (const posInfo & s, double w, double nw, int color,
            int pos, bool locked) :
      node {s, w, nw, color, pos}, locked {locked}
      {};
    R2Node (const SelectedNode & s) :
      node {s.s, s.w, s.nw, s.color, s.pos},
        locked {s.locked}
      {};
  } R2Node;

  typedef struct {
    int X1;
    int Y1;
    int X2;
    int Y2;
  } rect;

  typedef struct MetaData {
    Eigen::Vector3i zZ;
    int x, y, z;
    double vox, s;
    void writeToFile(std::ofstream & out);
    void loadFromFile(std::ifstream & in);

  } MetaData;

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

      friend std::ostream & operator<<(std::ostream & os, const place::cube & print);
  };

  typedef struct Panorama {
    std::vector<cv::Mat> imgs;
    Eigen::RowMatrixXf rMap;
    std::vector<cv::Point2f> keypoints;
    Eigen::ArrayXV3f surfaceNormals;
    void writeToFile(const std::string & imgName,
      const std::string & dataName);
    void loadFromFile(const std::string & imgName,
      const std::string & dataName);

    const cv::Mat & operator[](int n);
  } Panorama;
} // place

#endif // SCAN_TYPEDEFS_HPP
