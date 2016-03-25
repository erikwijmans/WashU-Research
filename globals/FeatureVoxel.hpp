#ifndef FEATURE_VOXEL_HPP
#define FEATURE_VOXEL_HPP

#include <scan_typedefs.hpp>
#include <vector>
#include <fstream>

namespace voxel {
	class FeatureVoxel {
		private:
			std::vector<Eigen::Vector1344f> featureVectors;
			std::vector<Eigen::MatrixXi> voxelGrid;
			nextID;
		public:
			FeatureVoxel();
			FeatureVoxel(const std::vector<Eigen::MatrixXi> & voxelGrid);
			~FeatureVoxel();
			void setVoxelGrid(const std::vector<Eigen::MatrixXi> & voxelGrid);
			void setVoxelGrid(const std::vector<Eigen::MatrixXi> & voxelGrid,
				int numNonZeros);
			void writeToFile(const std::string & name);
			void loadFromFile(const std::string & name);
			void addFeatureVector(int x, int y, int z, 
				const Eigen::Vector1344f & featureVector);
			Eigen::MatrixXi & getSlice(int z);
			Eigen::Vector1344f * getDescriptor(int ID);
			Eigen::Vector1344f * getDescriptor(int x, y, z);
	};
} // voxel

voxel::FeatureVoxel::FeatureVoxel() {
	int nextID = 1;
}

voxel::FeatureVoxel::FeatureVoxel(const std::vector<Eigen::MatrixXi> & voxelGrid) {
	int nextID = 1;
	this->voxelGrid = voxelGrid;
}

void voxel::FeatureVoxel::setVoxelGrid(const std::vector<Eigen::MatrixXi> & voxelGrid) {
	this->voxelGrid = voxelGrid;
}

void voxel::FeatureVoxel::setVoxelGrid(const std::vector<Eigen::MatrixXi> & voxelGrid,
	int numNonZeros) {
	this->voxelGrid = voxelGrid;
	this->numNonZeros = numNonZeros;
}

void voxel::FeatureVoxel::addFeatureVector(int x, int y, int z,
	const Eigen::Vector1344f & featureVector) {
	this->voxelGrid[z](y,x) = nextID++;
	featureVector.push_back(featureVector);
	++numFeatureVectors;
}

Eigen::MatrixXi & voxel::FeatureVoxel::getSlice(int z) {
	return this->voxelGrid[z];
}

Eigen::Vector1344f * voxel::FeatureVoxel::getDescriptor(int ID) {
	if (ID - 1 >= featureVectors.size() || ID - 1 < 0)
		return NULL;
	else
		return &this->featureVectors[ID-1];
}

Eigen::Vector1344f * voxel::FeatureVoxel::getDescriptor(int x, int y, int z) {
	int ID = this->voxelGrid[z](y,x);
	return this->getDescriptor(ID);
}

void voxel::FeatureVoxel::writeToFile(const std::string & name) {
	std::ofstream out (name, std::ios::out | std::ios::binary);
	int z = this->voxelGrid.size();
	int y = this->voxelGrid[0].rows();
	int x = this->voxelGrid[0].cols();

	out.write(reinterpret_cast<const char *>(& z), sizeof(z));
  out.write(reinterpret_cast<const char *>(& y), sizeof(y));
  out.write(reinterpret_cast<const char *>(& x), sizeof(x));

  for (int k = 0; k < z; ++k) {
  	const int * dataPtr = this->voxelGrid[k].data();
  	for (int i = 0; i < y*x; ++i) {
  		out.write(reinterpret_cast<const char *>(dataPtr+i), sizeof(int) - 1);
  	}
  }

  int numFeatureVectors = this->featureVectors.size();
  out.write(reinterpret_cast<const char *>(&numFeatureVectors), sizeof(int));

  for (auto& v : this->featureVectors)
  	out.write(reinterpret_cast<const char *>(&v), sizeof(Eigen::Vector1344f));
  
}

void voxel::FeatureVoxel::loadFromFile(const std::string & name) {
	std::ifstream in (name, std::ios::in | std::ios::binary);
	int x, y, z;
	in.read(reinterpret_cast<char *>(&z), sizeof(z));
	in.read(reinterpret_cast<char *>(&y), sizeof(y));
	in.read(reinterpret_cast<char *>(&x), sizeof(x));

	this->voxelGrid.assign(z, Eigen::MatrixXi::Zero(y,x));

	for (int k = 0; k < z; ++k) {
  	const int * dataPtr = this->voxelGrid[k].data();
  	for (int i = 0; i < y*x; ++i) {
  		in.read(reinterpret_cast<char *>(dataPtr+i), sizeof(int) - 1);
  	}
  }
	int numFeatureVectors;
	in.read(reinterpret_cast<char *>(&numFeatureVectors), sizeof(int));
	this->featureVectors.resize(numFeatureVectors);
	for(auto & v : this->featureVectors)
		in.read(reinterpret_cast<char *>(&v), sizeof(Eigen::Vector1344f));

}

#endif // FEATURE_VOXEL_HPP
