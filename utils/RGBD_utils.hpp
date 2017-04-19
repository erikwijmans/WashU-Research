#pragma once
#ifndef RGBD_UTILS_HPP
#define RGBD_UTILS_HPP

#include "scan_gflags.h"
#include <etw_utils.hpp>

#define NUM_ROTS 4
constexpr double maxPhi = 2.61946;

class BuildingScale {
private:
  double scale = -1;

public:
  double getScale();
  void update(double scale);
};

extern BuildingScale buildingScale;

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
  void writeToFile(const fs::path &imgName, const fs::path &dataName);
  void loadFromFile(const fs::path &imgName, const fs::path &dataName);
  Panorama();

  const cv::Mat &operator[](int n);
};

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

namespace std {
template <> struct hash<place::posInfo> {
  static constexpr double A = 1.6180339887498948482 * 1e5;
  hash<double> h;
  size_t operator()(const place::posInfo &e) const;
};
} // std

namespace hidden {
inline std::tuple<std::string, std::string>
parse_name(const std::string &name) {
  return std::make_tuple(name.substr(0, 3), name.substr(name.find(".") - 3, 3));
}

} // hidden

inline std::tuple<std::string, std::string> parse_name(const fs::path &p) {
  return hidden::parse_name(p.filename().string());
}

#endif // RGBD_UTILS_HPP
