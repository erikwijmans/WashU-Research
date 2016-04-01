#ifndef FEATURE_VOXEL_HPP
#define FEATURE_VOXEL_HPP

#include <scan_typedefs.hpp>
#include <vector>
#include <fstream>
#include <memory>

namespace voxel {
	template<typename F>
	class FeatureVoxel {
		typedef Eigen::MatrixXi matType;
		typedef std::vector<matType> gridType;
		typedef F descriptorType;
		typedef std::shared_ptr<descriptorType> DescripPtr;
		typedef std::shared_ptr<const descriptorType> ConstDescripPtr;
		private:
			std::vector<DescripPtr> featureVectors;
			gridType voxelGrid;
			int nextID;
			int numNonZeros;
		public:
			FeatureVoxel() : nextID {2}, numNonZeros {0}
				{};
			FeatureVoxel(const gridType & voxelGrid,
				int numNonZeros) :  nextID {2}, voxelGrid {voxelGrid},
				numNonZeros {numNonZeros}
				{};
			FeatureVoxel(const gridType & voxelGrid) : nextID {2}, voxelGrid {voxelGrid}
			{
				updateNumNonZeros();
			};
			FeatureVoxel(const gridType && voxelGrid) : nextID {2}, voxelGrid {voxelGrid}
			{
				updateNumNonZeros();
			};
			FeatureVoxel(const gridType && voxelGrid, const std::vector<DescripPtr> & featureVectors) :
				nextID {2}, voxelGrid {voxelGrid},
				featureVectors {featureVectors}
			{
				updateNumNonZeros();
			};
			void setVoxelGrid(const gridType & voxelGrid,
				int numNonZeros) {
				this->voxelGrid = voxelGrid; 
				this->numNonZeros = numNonZeros; };
			void writeToFile(const std::string & name) {
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
			  	out.write(reinterpret_cast<const char *>(v->data()), sizeof(F));
			  
			  out.write(reinterpret_cast<const char *>(&nextID), sizeof(nextID));
			  out.write(reinterpret_cast<const char *>(&numNonZeros), sizeof(numNonZeros));

			};
			void loadFromFile(const std::string & name) {
				std::ifstream in (name, std::ios::in | std::ios::binary);
				int x, y, z;
				in.read(reinterpret_cast<char *>(&z), sizeof(z));
				in.read(reinterpret_cast<char *>(&y), sizeof(y));
				in.read(reinterpret_cast<char *>(&x), sizeof(x));

				this->voxelGrid.assign(z, Eigen::MatrixXi::Zero(y,x));

				for (int k = 0; k < z; ++k) {
			  	int * dataPtr = this->voxelGrid[k].data();
			  	for (int i = 0; i < y*x; ++i) {
			  		in.read(reinterpret_cast<char *>(dataPtr+i), sizeof(int) - 1);
			  	}
			  }
				int numFeatureVectors;
				in.read(reinterpret_cast<char *>(&numFeatureVectors), sizeof(int));
				this->featureVectors.assign(numFeatureVectors, std::make_shared<F>());
				for(auto & v : this->featureVectors)
					in.read(reinterpret_cast<char *>(v->data()), sizeof(F));

				in.read(reinterpret_cast<char *>(&nextID), sizeof(nextID));
				in.read(reinterpret_cast<char *>(&numNonZeros), sizeof(numNonZeros));
			};
			int addFeatureVector(int x, int y, int z, 
				const DescripPtr & featureVector) {
				int id = nextID++;
				this->voxelGrid[z](y,x) = id;
				featureVectors.push_back(featureVector);
				return id;
			};
			void setFeatureVectors(const std::vector<DescripPtr> & featureVectors) {
				this->featureVectors = featureVectors;
			}
			void updateNumNonZeros() {
				int numNonZeros = 0;
				for (int k = 0; k < voxelGrid.size(); ++k) {
					const int * dataPtr = voxelGrid[k].data();
					for (int i = 0; i < voxelGrid[k].size(); ++i) {
						if (*(dataPtr+i))
							++numNonZeros;
					}
				}
			};
			matType & operator[](int n) {
				return this->voxelGrid[n];
			};
			DescripPtr getFeatureVector(int ID){
				if (ID - 2 >= featureVectors.size() || ID - 2 < 0)
					return NULL;
				else
					return this->featureVectors[ID-2];
			};
			DescripPtr getFeatureVector(int x, int y, int z){
				int ID = this->voxelGrid[z](y,x);
				return this->getDescriptor(ID);
			};
			const matType & operator[](int n) const {
				return this->voxelGrid[n];
			};
			ConstDescripPtr getFeatureVector(int ID) const {
				if (ID - 2 >= featureVectors.size() || ID - 2 < 0)
					return NULL;
				else
					return this->featureVectors[ID-2];
			};
			ConstDescripPtr getFeatureVector(int x, int y, int z) const {
				int ID = this->voxelGrid[z](y,x);
				return this->getDescriptor(ID);
			};
			const gridType & getGrid() const { return this->voxelGrid; };
			const std::vector<DescripPtr> & getAllFeatureVectors() const { return this->featureVectors; };
			int getNumZ() const { return voxelGrid.size(); };
			int getNumY() const { return voxelGrid[0].rows(); };
			int getNumX() const { return voxelGrid[0].cols(); };
			int getNumNonZeros() const { return numNonZeros; };
			int getNumFeatures() const { return featureVectors.size(); };
			int getID(int x, int y, int z) const {
				return voxelGrid[z](y,x);
			};
			void clear() {
				voxelGrid.clear();
				featureVectors.clear();
			}
	};
} // voxel

#endif // FEATURE_VOXEL_HPP
