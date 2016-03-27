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
			int nextID;
		public:
			FeatureVoxel();
			FeatureVoxel(const std::vector<Eigen::MatrixXi> & voxelGrid);
			~FeatureVoxel();
			void setVoxelGrid(const std::vector<Eigen::MatrixXi> & voxelGrid);
			void writeToFile(const std::string & name);
			void loadFromFile(const std::string & name);
			void addFeatureVector(int x, int y, int z, 
				const Eigen::Vector1344f & featureVector);
			Eigen::MatrixXi & getSlice(int z);
			Eigen::Vector1344f * getDescriptor(int ID);
			Eigen::Vector1344f * getDescriptor(int x, int y, int z);
	};
} // voxel

#endif // FEATURE_VOXEL_HPP
