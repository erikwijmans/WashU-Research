#include "scan_typedefs.hpp"
#include <locale>
#include <opencv2/core.hpp>

double BuildingScale::getScale() {
  if (this->scale == -1) {
    std::ifstream in(FLAGS_dataPath + "/scale.txt");
    in >> this->scale;
    in.close();
  }
  return this->scale;
}

void BuildingScale::update(double scale) {
  this->scale = scale;
  std::ofstream out(FLAGS_dataPath + "/scale.txt");
  out << scale;
  out.close();
}

size_t std::hash<std::vector<int>>::
operator()(const std::vector<int> &k) const {
  size_t seed = 0;
  for (auto v : k) {
    seed ^= h(v * A) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  return seed;
}

BuildingScale buildingScale;

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

  for (int i = 0; mat.outerSize(); ++i) {
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

void place::Panorama::writeToFile(const std::string &imgName,
                                  const std::string &dataName) {
  cv::imwrite(imgName, imgs[0]);

  std::ofstream out(dataName, std::ios::out | std::ios::binary);
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

  out.close();
}

void place::Panorama::loadFromFile(const std::string &imgName,
                                   const std::string &dataName) {
  imgs.resize(1);
  imgs[0] = cv::imread(imgName);

  int rows, cols, numKeypoints;
  std::ifstream in(dataName, std::ios::in | std::ios::binary);
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
  in.close();
}

const cv::Mat &place::Panorama::operator[](int n) {
  if (imgs.size() <= n || !imgs[n].data) {
#pragma omp critical
    {
      if (imgs.size() <= n)
        imgs.resize(n + 1);

      const double scale = pow(2, -n / 2.0);
      cv::resize(imgs[0], imgs[n], cv::Size(), scale, scale, CV_INTER_AREA);
    }
  }
  return imgs[n];
}

double place::edge::getWeight() const {
  return w * wSignificance + panoW * panoSignificance;
}

place::exclusionMap::exclusionMap(double exclusionX, double exclusionY,
                                  int rows, int cols)
    : maps{new const place::posInfo **[NUM_ROTS]}, exclusionX{exclusionX},
      exclusionY{exclusionY}, rows{rows}, cols{cols} {
  for (int i = 0; i < NUM_ROTS; ++i)
    // 2d array with one access index:  [<colNumber>*rows + <rowNumber>]
    maps[i] = new const place::posInfo *[rows * cols]();
}

place::exclusionMap::~exclusionMap() {
  for (int i = 0; i < NUM_ROTS; ++i)
    delete[] maps[i];
  delete[] maps;
}

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

std::ostream &place::operator<<(std::ostream &os, const place::cube &print) {
  os << "(" << print.X1 << ", " << print.Y1 << ", " << print.Z1 << ")"
     << std::endl;
  os << "      "
     << "(" << print.X2 << ", " << print.Y2 << ", " << print.Z2 << ")";
  return os;
}

std::ostream &place::operator<<(std::ostream &os, const place::edge &print) {
  os << "edge: " << print.w << "  " << print.wSignificance;
  os << "  pano: " << print.panoW << "  " << print.panoSignificance;
  os << "  " << print.numSim << "  " << print.numDiff << std::endl;
  os << print.pA << "  " << print.feA << std::endl;
  os << print.fx << "  " << print.feB;
  return os;
}

std::ostream &place::operator<<(std::ostream &os, const place::posInfo *print) {
  os << print->score << "  " << print->x << "  " << print->y << "  "
     << print->rotation << std::endl;
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