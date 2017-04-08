#include "RGBD_utils.hpp"

double BuildingScale::getScale() {
  if (this->scale == -1) {
    std::ifstream in(FLAGS_dataPath + "/scale.txt");
    CHECK(in.is_open()) << "Could not open " << FLAGS_dataPath << "/scale.txt"
                        << std::endl;
    in >> this->scale;
    in.close();
  }
  return this->scale;
}

void BuildingScale::update(double scale) {
  this->scale = scale;
  std::ofstream out(FLAGS_dataPath + "/scale.txt");
  CHECK(out.is_open()) << "Could not open " << FLAGS_dataPath << "/scale.txt"
                       << std::endl;
  out << scale;
  out.close();
}

BuildingScale buildingScale;

void SHOT1344WithXYZ::writeToFile(std::ofstream &out) {
  out.write(reinterpret_cast<const char *>(descriptor->data()),
            descriptor->size() * sizeof(float));
  out.write(reinterpret_cast<const char *>(position.data()), sizeof(position));
}

void SHOT1344WithXYZ::loadFromFile(std::ifstream &in) {
  in.read(reinterpret_cast<char *>(descriptor->data()),
          descriptor->size() * sizeof(float));
  in.read(reinterpret_cast<char *>(position.data()), sizeof(position));
}

void SPARSE352WithXYZ::writeToFile(std::ofstream &out) {
  saveSpareVector(*descriptor, out);
  out.write(reinterpret_cast<const char *>(position.data()), sizeof(position));
}

void SPARSE352WithXYZ::loadFromFile(std::ifstream &in) {
  loadSparseVetor(*descriptor, in);
  in.read(reinterpret_cast<char *>(position.data()), sizeof(position));
}

void SPARSE1344WithXYZ::writeToFile(std::ofstream &out) {
  saveSpareVector(*descriptor, out);
  out.write(reinterpret_cast<const char *>(position.data()), sizeof(position));
}

void SPARSE1344WithXYZ::loadFromFile(std::ifstream &in) {
  loadSparseVetor(*descriptor, in);
  in.read(reinterpret_cast<char *>(position.data()), sizeof(position));
}

void scan::PointXYZRGBA::writeToFile(std::ofstream &out) {
  out.write(reinterpret_cast<const char *>(point.data()), sizeof(point));
  out.write(reinterpret_cast<const char *>(&intensity), sizeof(intensity));
  out.write(reinterpret_cast<const char *>(rgb), 3 * sizeof(char));
}
void scan::PointXYZRGBA::loadFromFile(std::ifstream &in) {
  in.read(reinterpret_cast<char *>(point.data()), sizeof(point));
  in.read(reinterpret_cast<char *>(&intensity), sizeof(intensity));
  in.read(reinterpret_cast<char *>(rgb), 3 * sizeof(char));
}

void scan::PointXYZRGB::writeToFile(std::ofstream &out) {
  out.write(reinterpret_cast<const char *>(point.data()), sizeof(point));
  out.write(reinterpret_cast<const char *>(rgb), 3 * sizeof(char));
}
void scan::PointXYZRGB::loadFromFile(std::ifstream &in) {
  in.read(reinterpret_cast<char *>(point.data()), sizeof(point));
  in.read(reinterpret_cast<char *>(rgb), 3 * sizeof(char));
}

void place::VoxelGrid::writeToFile(std::ofstream &out) {
  int numZ = v.size();
  out.write(reinterpret_cast<const char *>(&numZ), sizeof(numZ));
  for (int k = 0; k < numZ; ++k) {
    saveMatrixAsSparse(v[k], out);
  }
  out.write(reinterpret_cast<const char *>(zZ.data()), sizeof(zZ));
  out.write(reinterpret_cast<const char *>(&c), sizeof(c));
}

void place::VoxelGrid::loadFromFile(std::ifstream &in) {
  int numZ;
  in.read(reinterpret_cast<char *>(&numZ), sizeof(numZ));
  v.resize(numZ);
  for (int k = 0; k < numZ; ++k) {
    loadMatrixFromSparse(v[k], in);
  }
  in.read(reinterpret_cast<char *>(zZ.data()), sizeof(zZ));
  in.read(reinterpret_cast<char *>(&c), sizeof(c));
}

void place::MetaData::writeToFile(std::ofstream &out) {
  out.write(reinterpret_cast<const char *>(zZ.data()), sizeof(zZ));
  out.write(reinterpret_cast<const char *>(&x), sizeof(x));
  out.write(reinterpret_cast<const char *>(&y), sizeof(y));
  out.write(reinterpret_cast<const char *>(&z), sizeof(z));
  out.write(reinterpret_cast<const char *>(&vox), sizeof(vox));
  out.write(reinterpret_cast<const char *>(&s), sizeof(s));
}

void place::MetaData::loadFromFile(std::ifstream &in) {
  in.read(reinterpret_cast<char *>(zZ.data()), sizeof(zZ));
  in.read(reinterpret_cast<char *>(&x), sizeof(x));
  in.read(reinterpret_cast<char *>(&y), sizeof(y));
  in.read(reinterpret_cast<char *>(&z), sizeof(z));
  in.read(reinterpret_cast<char *>(&vox), sizeof(vox));
  in.read(reinterpret_cast<char *>(&s), sizeof(s));
}

place::Panorama::Panorama() : imgs{20} {};

void place::Panorama::writeToFile(const fs::path &imgName,
                                  const fs::path &dataName) {
  cv::imwrite(imgName.string(), imgs[0]);

  std::ofstream out(dataName.string(), std::ios::out | std::ios::binary);
  int rows = rMap.rows();
  int cols = rMap.cols();
  out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
  out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
  const float *dataPtr = rMap.data();
  out.write(reinterpret_cast<const char *>(dataPtr),
            sizeof(float) * rMap.size());

  const int numKeypoints = keypoints.size();
  out.write(reinterpret_cast<const char *>(&numKeypoints),
            sizeof(numKeypoints));
  for (auto &kp : keypoints) {
    out.write(reinterpret_cast<const char *>(&kp.x), sizeof(float));
    out.write(reinterpret_cast<const char *>(&kp.y), sizeof(float));
  }

  rows = surfaceNormals.rows();
  cols = surfaceNormals.cols();
  out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
  out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
  const auto *nPtr = surfaceNormals.data();
  for (int i = 0; i < surfaceNormals.size(); ++i) {
    out.write(reinterpret_cast<const char *>((nPtr + i)->data()),
              3 * sizeof(float));
  }
  out.write(reinterpret_cast<const char *>(&floorCoord), sizeof(floorCoord));
  out.close();
}

void place::Panorama::loadFromFile(const fs::path &imgName,
                                   const fs::path &dataName) {
  imgs[0] = cv::imread(imgName.string());

  int rows, cols, numKeypoints;
  std::ifstream in(dataName.string(), std::ios::in | std::ios::binary);
  in.read(reinterpret_cast<char *>(&rows), sizeof(rows));
  in.read(reinterpret_cast<char *>(&cols), sizeof(cols));
  rMap.resize(rows, cols);
  float *dataPtr = rMap.data();
  in.read(reinterpret_cast<char *>(dataPtr), sizeof(float) * rMap.size());

  in.read(reinterpret_cast<char *>(&numKeypoints), sizeof(numKeypoints));
  keypoints.resize(numKeypoints);
  for (auto &kp : keypoints) {
    in.read(reinterpret_cast<char *>(&kp.x), sizeof(float));
    in.read(reinterpret_cast<char *>(&kp.y), sizeof(float));
  }
  in.read(reinterpret_cast<char *>(&rows), sizeof(rows));
  in.read(reinterpret_cast<char *>(&cols), sizeof(cols));
  surfaceNormals.resize(rows, cols);
  auto *nPtr = surfaceNormals.data();
  for (int i = 0; i < surfaceNormals.size(); ++i) {
    in.read(reinterpret_cast<char *>((nPtr + i)->data()), 3 * sizeof(float));
  }
  in.read(reinterpret_cast<char *>(&floorCoord), sizeof(floorCoord));
  in.close();

  if (floorCoord > -1.5 || floorCoord < -1.7)
    floorCoord = -1.6;
}

const cv::Mat &place::Panorama::operator[](int n) {
  if (imgs.size() <= n || !imgs[n].data) {
#pragma omp critical
    {
      if (imgs.size() <= n || !imgs[n].data) {
        if (imgs.size() <= n)
          imgs.resize(n + 1);

        const double scale = pow(ScalingFactor, -n);
        cv::resize(imgs[0], imgs[n], cv::Size(), scale, scale, CV_INTER_AREA);
      }
    }
  }
  return imgs[n];
}

double place::edge::getWeight() const {
  return (0.5 * w * wSignificance + panoW * panoSignificance) *
             std::max(0.25, std::min(distance, 1.5)) +
         hWeight;
}

double place::node::getWeight() const { return nw; };

place::ExclusionMap::ExclusionMap(double exclusionSize, int rows, int cols,
                                  int numRots)
    : maps{new Map[numRots]},
      exclusionSize{exclusionSize}, rows{rows}, cols{cols} {
  for (int i = 0; i < numRots; ++i)
    maps[i] = Map::Zero(rows, cols);
}

place::ExclusionMap::ExclusionMap(double exclusionSize, int rows, int cols)
    : maps{new Map[NUM_ROTS]},
      exclusionSize{exclusionSize}, rows{rows}, cols{cols} {
  for (int i = 0; i < NUM_ROTS; ++i)
    maps[i] = Map::Zero(rows, cols);
}

place::ExclusionMap::Map &place::ExclusionMap::operator[](int r) {
  return maps[r];
}

place::ExclusionMap::~ExclusionMap() { delete[] maps; }

void place::Wall::init(const Eigen::Vector2d &n) {
  normal = new Eigen::Vector2d(n);
  *normal /= normal->norm();
  s = new side[2];
}
void place::Wall::changeNormal(const Eigen::Vector2d &n) {
  *normal = n;
  *normal /= normal->norm();
}

const Eigen::Vector2d &place::Wall::getNormal() { return *normal; }

place::Wall::side &place::Wall::getSide(const Eigen::Vector2d &ray) {
  const double theta = std::acos(normal->dot(ray) / (ray.norm()));
  if (theta < 0)
    return s[0];
  else
    return s[1];
}

place::Wall::~Wall() {
  if (normal)
    delete normal;
  if (s)
    delete[] s;
}

size_t std::hash<place::posInfo>::operator()(const place::posInfo &e) const {
  size_t seed = h(A * e.rotation) + 0x9e3779b9;
  seed ^= h(A * e.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= h(A * e.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

  return seed;
}

std::ostream &place::operator<<(std::ostream &os, const place::cube &print) {
  os << "({}, {}, {})"_format(print.X1, print.Y1, print.Z1) << std::endl
     << "({}, {}, {})"_format(print.X2, print.Y2, print.Z2);
  return os;
}

std::ostream &place::operator<<(std::ostream &os, const place::edge &print) {
  fmt::print(os,
             "edge: {} {}\tpano: {} {}\t{} {}\tDistance: {}\tHighOrder: {}\n"
             "{}\t{}\n"
             "{}\t{}\n"
             "weight: {}\n",
             print.w, print.wSignificance, print.panoW, print.panoSignificance,
             print.numSim, print.numDiff, print.distance, print.hWeight,
             print.pA, print.feA, print.fx, print.feB, print.getWeight());
  return os;
}

std::ostream &place::operator<<(std::ostream &os, const place::posInfo *print) {
  os << print->score << "  " << print->x << "  " << print->y << "  "
     << print->rotation << "  " << print->doorUxp << "  " << print->doorCount
     << std::endl;
  os << print->scanFP << "  " << print->fpScan << std::endl;
  os << print->scanPixels << " " << print->fpPixels;
  return os;
}

std::ostream &place::operator<<(std::ostream &os, const place::posInfo &print) {
  os << &print;
  return os;
}

std::ostream &place::operator<<(std::ostream &os,
                                const place::SelectedNode &p) {
  os << "Color: " << p.color << "  ";
  os << "Agreement: " << p.agreement << "  ";
  os << "Norm: " << p.norm << "  ";
  os << "Label #: " << p.label << "  ";
  if (!p.locked)
    os << "Selected for relabeling";
  else
    os << "Locked";
  os << std::endl;
  return os;
}

void place::Door::writeToFile(std::ofstream &out) const {
  out.write(reinterpret_cast<const char *>(corner.data()), 3 * sizeof(double));
  out.write(reinterpret_cast<const char *>(xAxis.data()), 3 * sizeof(double));
  out.write(reinterpret_cast<const char *>(zAxis.data()), 3 * sizeof(double));
  out.write(reinterpret_cast<const char *>(&h), sizeof(double));
  out.write(reinterpret_cast<const char *>(&w), sizeof(double));
}

void place::Door::loadFromFile(std::ifstream &in) {
  in.read(reinterpret_cast<char *>(corner.data()), 3 * sizeof(double));
  in.read(reinterpret_cast<char *>(xAxis.data()), 3 * sizeof(double));
  in.read(reinterpret_cast<char *>(zAxis.data()), 3 * sizeof(double));
  in.read(reinterpret_cast<char *>(&h), sizeof(double));
  in.read(reinterpret_cast<char *>(&w), sizeof(double));
}