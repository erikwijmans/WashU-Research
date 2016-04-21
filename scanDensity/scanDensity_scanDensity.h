#ifndef SCAN_DENSITY_SCAN_DENSITY_H
#define SCAN_DENSITY_SCAN_DENSITY_H

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/StdVector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <string>
#include <dirent.h>
#include <fstream>
#include <math.h>
#include <time.h>
#include <memory>

#include <scan_gflags.h>
#include <scan_typedefs.hpp>

class DensityMapsManager {
	public:
		typedef std::shared_ptr<const std::vector<Eigen::Vector3f> > PointsPtr;
		typedef std::shared_ptr<const std::vector<Eigen::Matrix3d> > MatPtr;
		typedef std::shared_ptr<const std::vector<SPARSE352WithXYZ> > FeaturePtr;
		/* Constructs argv and argc, then calls the constructor with them */
		DensityMapsManager(const std::string & commandLine);
		DensityMapsManager(int argc, char * argv[]);
		/*Runs 2D and 3D based on flags in the range specified */
		void resetFlags(const std::string & commandLine);
		void resetFlags(int argc, char * argv[]);
		void run();
		bool hasNext();
		bool exists2D();
		bool exists3D();
		void setNext();
		void get2DPointNames(std::vector<std::string> & names);
		void get3DPointNames(std::vector<std::string> & names);
		void get2DFreeNames(std::vector<std::string> & names);
		void get3DFreeNames(std::vector<std::string> & names);
		std::string getZerosName();
		std::string getMetaDataName();
		PointsPtr getPointsWithCenter() {
			return pointsWithCenter; };
		PointsPtr getPointsNoCenter() {
			return pointsNoCenter; };	
		MatPtr getR() {
			return R; };
		FeaturePtr getFeatureVectors() {
			return featureVectors;
		}
		void setScale(double newScale) { FLAGS_scale = newScale; };
		double getScale() { return FLAGS_scale; };

	private:
		std::vector<std::string> binaryNames, rotationsFiles, featureNames;
		std::shared_ptr<std::vector<Eigen::Vector3f> > pointsWithCenter;
		std::shared_ptr<std::vector<Eigen::Vector3f> > pointsNoCenter;
		std::shared_ptr<std::vector<Eigen::Matrix3d> > R;
		std::shared_ptr<std::vector<SPARSE352WithXYZ> > featureVectors;
		std::string rotationFile, fileName, scanNumber, buildName, featName;
		int current;
};


class BoundingBox {
	private:
		Eigen::Vector3f average, sigma, range;
		DensityMapsManager::PointsPtr points;
	public:
		typedef std::shared_ptr<BoundingBox> Ptr;
		typedef std::shared_ptr<const BoundingBox> ConstPtr;
		template<typename... Targs>
		static inline Ptr Create(Targs... args) {
			return std::make_shared<BoundingBox> (args...);
		};
		BoundingBox(const DensityMapsManager::PointsPtr & points, 
			Eigen::Vector3f && range);
		BoundingBox(const DensityMapsManager::PointsPtr & points, 
			Eigen::Vector3f & range);
		void run();
		void setRange(Eigen::Vector3f && range);
		void setRange(Eigen::Vector3f & range);
		void getBoundingBox(Eigen::Vector3f & min, Eigen::Vector3f & max) const;
};

class CloudAnalyzer2D {
	private:
		BoundingBox::ConstPtr bBox;
		DensityMapsManager::PointsPtr points;
		DensityMapsManager::MatPtr R;
		std::vector<Eigen::MatrixXi> pointsPerVoxel;
		std::vector<cv::Mat> pointEvidence, freeSpaceEvidence;
		Eigen::Vector3f pointMin, pointMax;
		Eigen::Vector3d zeroZero;
		Eigen::Vector2i imageZeroZero;
		const int numZ = 100;
		int numY, numX;
		float zScale, scale;
	public:
		typedef std::shared_ptr<CloudAnalyzer2D> Ptr;
		CloudAnalyzer2D(const DensityMapsManager::PointsPtr & points,
			const DensityMapsManager::MatPtr & R,
			const BoundingBox::ConstPtr & bBox);
		void initalize(double scale);
		void examinePointEvidence();
		void examineFreeSpaceEvidence();
		const std::vector<cv::Mat> & getPointEvidence();
		const std::vector<cv::Mat> & getFreeSpaceEvidence();
		Eigen::Vector2i getImageZeroZero();
		float getScale();
};

#endif // SCAN_DENSITY_SCAN_DENSITY_H
