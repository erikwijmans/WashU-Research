#ifndef SCAN_DENSITY_SCAN_DENSITY_H
#define SCAN_DENSITY_SCAN_DENSITY_H

#define NUM_ROTS 4

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/StdVector>
#include <opencv2/core.hpp>
#include <gflags/gflags.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <string>
#include <dirent.h>
#include <vector>
#include <fstream>
#include <math.h>
#include <time.h>

DECLARE_bool(pe);
DECLARE_bool(fe);
DECLARE_bool(quiteMode);
DECLARE_bool(preview);
DECLARE_bool(redo);
DECLARE_bool(3D);
DECLARE_bool(2D);
DECLARE_string(inFolder);
DECLARE_string(outFolder);
DECLARE_string(zerosFolder);
DECLARE_string(voxelFolder);
DECLARE_string(rotFolder);
DECLARE_double(scale);
DECLARE_int32(startIndex);
DECLARE_int32(numScans);


class DensityMapsManager {
	private:
		std::vector<std::string> binaryNames, rotationsFiles;
		std::vector<Eigen::Vector3f> pointsWithCenter;
		std::vector<Eigen::Vector3f> pointsNoCenter;
		std::vector<Eigen::Matrix3d> R;
		std::string rotationFile, fileName, scanNumber, buildName;
		int current;
	public:
		/* Constructs argv and argc, then called the constructor with them */
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
		const std::vector<Eigen::Vector3f> * getPointsWithCenter() {
			return &pointsWithCenter; };
		const std::vector<Eigen::Vector3f> * getPointsNoCenter() {
			return &pointsNoCenter; };	
		const std::vector<Eigen::Matrix3d> * getR() {
			return &R; };
		void setScale(double newScale) { FLAGS_scale = newScale; };
		double getScale() { return FLAGS_scale; };
};


class BoundingBox {
	private:
		Eigen::Vector3f average, sigma, range;
		const std::vector<Eigen::Vector3f> * points;
	public:
		BoundingBox(const std::vector<Eigen::Vector3f> * points, Eigen::Vector3f && range);
		BoundingBox(const std::vector<Eigen::Vector3f> * points, Eigen::Vector3f & range);
		void run();
		void setRange(Eigen::Vector3f && range);
		void setRange(Eigen::Vector3f & range);
		void getBoundingBox(Eigen::Vector3f & min, Eigen::Vector3f & max) const;
};

class CloudAnalyzer2D {
	private:
		const BoundingBox * bBox;
		const std::vector<Eigen::Vector3f> * points;
		const std::vector<Eigen::Matrix3d> * R;
		std::vector<Eigen::MatrixXi> pointsPerVoxel;
		std::vector<cv::Mat> pointEvidence, freeSpaceEvidence;
		Eigen::Vector3f pointMin, pointMax;
		Eigen::Vector3d zeroZero;
		Eigen::Vector2i imageZeroZero;
		const int numZ = 100;
		int numY, numX;
		float zScale, scale;
	public:
		CloudAnalyzer2D(const std::vector<Eigen::Vector3f> * points,
			const std::vector<Eigen::Matrix3d> * R,
			const BoundingBox * bBox);
		void initalize(double scale);
		void examinePointEvidence();
		void examineFreeSpaceEvidence();
		const std::vector<cv::Mat> & getPointEvidence();
		const std::vector<cv::Mat> & getFreeSpaceEvidence();
		Eigen::Vector2i getImageZeroZero();
		float getScale();
};

#endif // SCAN_DENSITY_SCAN_DENSITY_H
