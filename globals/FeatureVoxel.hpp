#ifndef FEATURE_VOXEL_HPP
#define FEATURE_VOXEL_HPP

#include <scan_typedefs.hpp>
#include <vector>
#include <fstream>
#include <memory>
#include <iostream>

namespace voxel {

  template<typename T>
  class FeatureVoxel {
    public:
      typedef T Scalar;
      typedef Eigen::MatrixXi MatType;
      typedef std::vector<MatType> GridType;
      typedef Eigen::SparseVector<Scalar> DescripType;
      typedef std::shared_ptr<DescripType> DescripPtr;
      FeatureVoxel() : nextID {2}, numNonZeros {0}
        {};
      FeatureVoxel(const GridType & voxelGrid,
        int numNonZeros) :  nextID {2}, voxelGrid {voxelGrid},
        numNonZeros {numNonZeros}
        {};
      FeatureVoxel(const GridType & voxelGrid) : nextID {2}, voxelGrid {voxelGrid}
      {
        updateNumNonZeros();
      };
      FeatureVoxel(const GridType && voxelGrid) : nextID {2}, voxelGrid {voxelGrid}
      {
        updateNumNonZeros();
      };
      FeatureVoxel(const GridType && voxelGrid, const std::vector<DescripPtr> & featureVectors) :
        voxelGrid {voxelGrid},
        featureVectors {featureVectors}
      {
        this->nextID = featureVectors.size() + 2;
        updateNumNonZeros();
      };
      void setVoxelGrid(const GridType & voxelGrid,
        int numNonZeros) {
        this->voxelGrid = voxelGrid;
        this->numNonZeros = numNonZeros; };
      void setVoxelGrid(const GridType & voxelGrid) {
        this->voxelGrid = voxelGrid;
        updateNumNonZeros();
      }
      void setZeroZero(const Eigen::Vector3i & zZ) {
        this->zeroZero = zZ;
      };
      void writeToFile(const std::string & name) {
        if (numNonZeros == 0) updateNumNonZeros();

        if (getNumZ() == 0 || getNumFeatures() == 0 ||
          getNumY() == 0 || getNumX() == 0 || zeroZero.norm() == 0) {
          std::cout << "[voxel::FeatureVoxel::writeToFile] Feature Voxel has unset members" << std::endl;
          exit(1);
        }

        std::ofstream out (name, std::ios::out | std::ios::binary);

        int z = getNumZ();
        out.write(reinterpret_cast<const char *>(&z), sizeof(z));

        for (int k = 0; k < z; ++k)
          saveMatrixAsSparse(this->voxelGrid[k], out);


        int numFeatureVectors = this->featureVectors.size();
        out.write(reinterpret_cast<const char *>(&numFeatureVectors), sizeof(int));

        for (auto& v : this->featureVectors)
          saveSpareVector(*v, out);

        out.write(reinterpret_cast<const char *>(&nextID), sizeof(nextID));
        out.write(reinterpret_cast<const char *>(&numNonZeros), sizeof(numNonZeros));
        out.write(reinterpret_cast<const char *>(zeroZero.data()), sizeof(zeroZero));
      };
      void loadFromFile(const std::string & name) {
        clear();
        std::ifstream in (name, std::ios::in | std::ios::binary);

        if (!in.is_open()) {
          std::cout << "[voxel::FeatureVoxel::loadFromFile] Cloud not open: " << name << std::endl;
          exit (2);
        }

        int z;
        in.read(reinterpret_cast<char *>(&z), sizeof(z));

        this->voxelGrid.resize(z);

        for (int k = 0; k < z; ++k)
          loadMatrixFromSparse(this->voxelGrid[k], in);

        int numFeatureVectors;
        in.read(reinterpret_cast<char *>(&numFeatureVectors), sizeof(int));
        this->featureVectors.assign(numFeatureVectors, std::make_shared<DescripType>());
        for(auto & v : this->featureVectors)
          loadSparseVetor(*v, in);

        in.read(reinterpret_cast<char *>(&nextID), sizeof(nextID));
        in.read(reinterpret_cast<char *>(&numNonZeros), sizeof(numNonZeros));
        in.read(reinterpret_cast<char *>(zeroZero.data()), sizeof(zeroZero));
      };
      int addFeatureVector(int x, int y, int z,
        const DescripPtr & featureVector) {
        this->voxelGrid[z](y,x) = nextID;
        featureVectors.push_back(featureVector);
        return nextID++;
      };
      void setFeatureVectors(const std::vector<DescripPtr> & featureVectors) {
        this->featureVectors = featureVectors;
        this->nextID = featureVectors.size() + 2;
      }
      void updateNumNonZeros() {
        this->numNonZeros = 0;
        for (int k = 0; k < voxelGrid.size(); ++k) {
          const int * dataPtr = voxelGrid[k].data();
          for (int i = 0; i < voxelGrid[k].size(); ++i) {
            if (*(dataPtr+i))
              ++this->numNonZeros;
          }
        }
      };
      MatType & operator[](int n) {
        return this->voxelGrid[n];
      };
      void clear() {
        voxelGrid.clear();
        featureVectors.clear();
      };
      const Eigen::Vector3i & getZeroZero() const {
        return zeroZero;
      };
      DescripPtr getFeatureVector(int ID) const {
        if (ID - 2 >= featureVectors.size() || ID - 2 < 0)
          return NULL;
        else
          return this->featureVectors[ID-2];
      };
      DescripPtr getFeatureVector(int x, int y, int z) const {
        if (x < 0 || x >= getNumX())
          return NULL;
        if (y < 0 || y >= getNumY())
          return NULL;
        if (z < 0 || z >= getNumZ())
          return NULL;
        int ID = this->voxelGrid[z](y, x);
        return this->getFeatureVector(ID);
      };

      const MatType & operator[](int n) const {
        return this->voxelGrid[n];
      };
      const GridType & getGrid() const { return this->voxelGrid; };
      const std::vector<DescripPtr> & getAllFeatureVectors() const { return this->featureVectors; };
      int getNumZ() const { return voxelGrid.size(); };
      int getNumY() const { return getNumZ() ? voxelGrid[0].rows() : 0; };
      int getNumX() const { return getNumZ() ? voxelGrid[0].cols() : 0; };
      int getNumNonZeros() const { return numNonZeros; };
      int getNumFeatures() const { return featureVectors.size(); };
      int getID(int x, int y, int z) const {
        return voxelGrid[z](y,x);
      };
    private:
      std::vector<DescripPtr> featureVectors;
      GridType voxelGrid;
      int nextID;
      int numNonZeros;
      Eigen::Vector3i zeroZero = Eigen::Vector3i::Zero();
  };

  template<typename T>
  class SparseFeatureVoxel {
    public:
      typedef T Scalar;
      typedef Eigen::SparseMatrix<int> MatType;
      typedef std::vector<MatType> GridType;
      typedef Eigen::SparseVector<Scalar> DescripType;
      typedef std::shared_ptr<DescripType> DescripPtr;
      SparseFeatureVoxel() : nextID {2}, numNonZeros {0}
        {};
      SparseFeatureVoxel(const GridType & voxelGrid,
        int numNonZeros) :  nextID {2}, voxelGrid {voxelGrid},
        numNonZeros {numNonZeros}
        {};
      SparseFeatureVoxel(const GridType & voxelGrid) : nextID {2}, voxelGrid {voxelGrid}
      {
        updateNumNonZeros();
      };
      SparseFeatureVoxel(const GridType && voxelGrid) : nextID {2}, voxelGrid {voxelGrid}
      {
        updateNumNonZeros();
      };
      SparseFeatureVoxel(const GridType && voxelGrid, const std::vector<DescripPtr> & featureVectors) :
        voxelGrid {voxelGrid},
        featureVectors {featureVectors}
      {
        this->nextID = featureVectors.size() + 2;
        updateNumNonZeros();
      };
      void setVoxelGrid(const GridType & voxelGrid,
        int numNonZeros) {
        this->voxelGrid = voxelGrid;
        this->numNonZeros = numNonZeros; };
      void setVoxelGrid(const GridType & voxelGrid) {
        this->voxelGrid = voxelGrid;
        updateNumNonZeros();
      }
      void setZeroZero(const Eigen::Vector3i & zZ) {
        this->zeroZero = zZ;
      };
      void writeToFile(const std::string & name) {
        if (numNonZeros == 0) updateNumNonZeros();

        if (getNumZ() == 0 || getNumFeatures() == 0 ||
          getNumY() == 0 || getNumX() == 0 || zeroZero.norm() == 0) {
          std::cout << "[voxel::FeatureVoxel::writeToFile] Feature Voxel has unset members" << std::endl;
          exit(1);
        }

        std::ofstream out (name, std::ios::out | std::ios::binary);

        int z = getNumZ();
        out.write(reinterpret_cast<const char *>(&z), sizeof(z));

        for (int k = 0; k < z; ++k)
          saveSparseMatrix(this->voxelGrid[k], out);

        int numFeatureVectors = this->featureVectors.size();
        out.write(reinterpret_cast<const char *>(&numFeatureVectors), sizeof(int));

        for (auto& v : this->featureVectors)
          saveSpareVector(*v, out);

        out.write(reinterpret_cast<const char *>(&nextID), sizeof(nextID));
        out.write(reinterpret_cast<const char *>(&numNonZeros), sizeof(numNonZeros));
        out.write(reinterpret_cast<const char *>(zeroZero.data()), sizeof(zeroZero));
      };
      void loadFromFile(const std::string & name) {
        clear();
        std::ifstream in (name, std::ios::in | std::ios::binary);

        if (!in.is_open()) {
          std::cout << "[voxel::FeatureVoxel::loadFromFile] Cloud not open: " << name << std::endl;
          exit (2);
        }

        int z;
        in.read(reinterpret_cast<char *>(&z), sizeof(z));

        this->voxelGrid.resize(z);

        for (int k = 0; k < z; ++k)
          loadSparseMatrix(this->voxelGrid[k], in);

        int numFeatureVectors;
        in.read(reinterpret_cast<char *>(&numFeatureVectors), sizeof(int));
        this->featureVectors.assign(numFeatureVectors, std::make_shared<DescripType>());
        for(auto & v : this->featureVectors)
          loadSparseVetor(*v, in);

        in.read(reinterpret_cast<char *>(&nextID), sizeof(nextID));
        in.read(reinterpret_cast<char *>(&numNonZeros), sizeof(numNonZeros));
        in.read(reinterpret_cast<char *>(zeroZero.data()), sizeof(zeroZero));
      };
      int addFeatureVector(int x, int y, int z,
        const DescripPtr & featureVector) {
        this->voxelGrid[z].coeffRef(y,x) = nextID;
        featureVectors.push_back(featureVector);
        return nextID++;
      };
      void setFeatureVectors(const std::vector<DescripPtr> & featureVectors) {
        this->featureVectors = featureVectors;
        this->nextID = featureVectors.size() + 2;
      }
      void updateNumNonZeros() {
        this->numNonZeros = 0;
        for (auto & mat : voxelGrid) {
          mat.prune(1);
          this->numNonZeros += mat.nonZeros();
        }
      };
      MatType & operator[](int n) {
        return this->voxelGrid[n];
      };
      const Eigen::Vector3i & getZeroZero() const {
        return zeroZero;
      };
      DescripPtr getFeatureVector(int ID) const {
        if (ID - 2 >= featureVectors.size() || ID - 2 < 0)
          return NULL;
        else
          return this->featureVectors[ID-2];
      };
      const MatType & operator[](int n) const {
        return this->voxelGrid[n];
      };
      const GridType & getGrid() const { return this->voxelGrid; };
      const std::vector<DescripPtr> & getAllFeatureVectors() const { return this->featureVectors; };
      int getNumZ() const { return voxelGrid.size(); };
      int getNumY() const { return getNumZ() ? voxelGrid[0].rows() : 0; };
      int getNumX() const { return getNumZ() ? voxelGrid[0].cols() : 0; };
      int getNumNonZeros() const { return numNonZeros; };
      int getNumFeatures() const { return featureVectors.size(); };
      void clear() {
        voxelGrid.clear();
        featureVectors.clear();
      }

    private:
      std::vector<DescripPtr> featureVectors;
      GridType voxelGrid;
      int nextID;
      int numNonZeros;
      Eigen::Vector3i zeroZero = Eigen::Vector3i::Zero();
  };
} // voxel

#endif // FEATURE_VOXEL_HPP
