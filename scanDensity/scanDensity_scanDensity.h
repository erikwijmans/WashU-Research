#ifndef SCAN_DENSITY_SCAN_DENSITY_H
#define SCAN_DENSITY_SCAN_DENSITY_H

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


class DensityMaps {
	private:
		std::vector<std::string> binaryNames, rotationsFiles;
	public:
		DensityMaps(int argc, char * argv[]);
		void run();
		void run(int startIndex, int numScans);
		void setScale(double newScale) {FLAGS_scale = newScale;};
		double getScale() {return FLAGS_scale;};
};


typedef struct {
	float * p;
	int size;
	int stride;
} pointsInfo;

void examinePointEvidence(const std::vector<Eigen::Vector3f> & points,
	const std::vector<Eigen::Matrix3d> & R,
	const float* pointMin, const float * pointMax, 
	const std::string & outputFolder, const std::string & scanNumber,
	std::ofstream & zZOut);
void createBoundingBox(float *, float *, const std::vector<Eigen::Vector3f> &);
void examineFreeSpaceEvidence(const std::vector<Eigen::Vector3f> &,
	const std::vector<Eigen::Matrix3d> & R,
	const float*, const float *,
   const std::string &, const std::string &);
void showSlices(const Eigen::MatrixXi & numTimesSeen,
    const int numZ, const int numY, const int numX, const std::string &);
void collapseFreeSpaceEvidence(const std::vector<Eigen::MatrixXi> & numTimesSeen,
	const std::vector<Eigen::Matrix3d> & R, const Eigen::Vector3d & zeroZero, 
	const int numZ, const int numY, const int numX,
	const std::string & scanNumber);
void displayCollapsed(const Eigen::MatrixXd & numTimesSeen,
	const std::vector<Eigen::Matrix3d> & R, const Eigen::Vector3d & zeroZero, 
	const std::string & scanNumber);
void displayPointEvenidence(const Eigen::MatrixXf & numTimesSeen,
	const std::vector<Eigen::Matrix3d> & R, const Eigen::Vector3d & zeroZero, 
	const std::string & imageName,
	const int bias, std::ofstream & zZOut);

void analyzeScan(const std::string & fileName, const std::string & outputFolder, const std::string & rotationFile);

#endif // SCAN_DENSITY_SCAN_DENSITY_H
