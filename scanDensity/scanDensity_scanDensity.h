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
DECLARE_string(inFolder);
DECLARE_string(outFolder);
DECLARE_string(zerosFolder);
DECLARE_string(voxelFolder);
DECLARE_string(rotFolder);
DECLARE_double(scale);
DECLARE_int32(startIndex);
DECLARE_int32(numScans);

void examinePointEvidence(const std::vector<Eigen::Vector3f> &, const float *, const float *,
    const std::string &, const std::string &);
void createBoundingBox(float *, float *, const std::vector<Eigen::Vector3f> &);
void examineFreeSpaceEvidence(const std::vector<Eigen::Vector3f> &, const float*, const float *,
    const std::string &, const std::string &);
void showSlices(const Eigen::MatrixXi & numTimesSeen,
    const int numZ, const int numY, const int numX, const std::string &);
void collapseFreeSpaceEvidence(const std::vector<Eigen::MatrixXi> &, const int, const int,
    const int, const std::string &, const std::string &);
void displayCollapsed(const Eigen::MatrixXd &, const int, const int, const std::string &);
void displayPointEvenidence(const Eigen::MatrixXf &, const std::string &, const int);

void analyzeScan(const std::string & fileName, const std::string & rotationFile, const std::string & outputFolder);

#endif // SCAN_DENSITY_SCAN_DENSITY_H
