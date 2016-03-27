#include "placeScan_panoramaMatcher.h"

#include <iostream>

static const int NCCSize = 7;
static const double panoResolution = 500;
static const double PI = 3.14159265358979323846;

static Eigen::Vector3d cartesianToPolar(const Eigen::Vector3d & coords) {
	double r = coords.norm();
	double theta = atan2(coords[1], coords[0]);
	double phi = acos(coords[2]/r);
	return Eigen::Vector3d (r, theta, phi);
}

static Eigen::Vector3d polarToCartesian(const Eigen::Vector3d & coords) {
	double x = coords[0]*cos(coords[1])*sin(coords[2]);
	double y = -1.0*coords[0]*sin(coords[1])*sin(coords[2]);
	double z = coords[0]*cos(coords[2]);
	return Eigen::Vector3d (x,y,z);
}

static Eigen::Vector2d pointCloudToPanorama(const Eigen::Vector3d & worldCoord,
	double & r) {
	Eigen::Vector3d local = worldCoord;
	local[1] *= -1.0;
	auto polar = cartesianToPolar(local);
	double x = (polar[1]/PI + 1.0)*panoResolution;
	double y = polar[2]/PI*panoResolution;
	r = polar[0];
	return Eigen::Vector2d (x, y);
}

static Eigen::Vector2d pointCloudToPanorama(const Eigen::Vector3d & worldCoord) {
	Eigen::Vector3d local = worldCoord;
	local[1] *= -1.0;
	auto polar = cartesianToPolar(local);
	double x = (polar[1]/PI + 1.0)*panoResolution;
	double y = polar[2]/PI*panoResolution;
	return Eigen::Vector2d (x, y);
}

static Eigen::Vector3d panoramaToPointCloud(const Eigen::Vector2d & panoCoord,
	const double r) {
	double theta = (panoCoord[0]/panoResolution - 1.0)*PI;
	double phi = panoCoord[1]*PI/panoResolution;
	return polarToCartesian(Eigen::Vector3d (r, theta, phi));
}

double pano::compare(const cv::Mat & panoA, const cv::Mat & panoB,
		const Eigen::Matrix3d & RA, const Eigen::Matrix3d & RB, 
		const std::vector<Eigen::Vector3d> & pointsInA,
		const Eigen::Vector3d & translation) {

	const Eigen::Matrix3d deltaRot = RB*RA.inverse();
	const Eigen::Vector3d rotatedTranslation = RB*translation;

	const cv::Mat_<cv::Vec3b> _panoA = panoA;
	const cv::Mat_<cv::Vec3b> _panoB = panoB;

	int count = 0;
	double score = 0.0, r;
	for (int i = 0; i < pointsInA.size(); ++i) {
		auto panoCoordsA = pointCloudToPanorama(pointsInA[i], r);
		Eigen::Vector2d NCCSquareA [NCCSize][NCCSize];
		Eigen::Vector2d NCCSquareB [NCCSize][NCCSize];
		Eigen::Vector2d offset (floor(NCCSize/2), floor(NCCSize/2));
		for (int j = 0; j < NCCSize; ++j) {
			for (int i = 0; i < NCCSize; ++i) {
				NCCSquareA[j][i] = panoCoordsA - offset + Eigen::Vector2d(j,i);
			}
		}

		for (int j = 0; j < NCCSize; ++j) {
			for (int i = 0; i < NCCSize; ++i) {
				auto aWorld = panoramaToPointCloud(NCCSquareA[j][i], r);
				auto bWorld = deltaRot*aWorld + rotatedTranslation;
				NCCSquareB[j][i] = pointCloudToPanorama(bWorld);
			}
		} 

		cv::Mat_<cv::Vec3b> _aPatch (NCCSize, NCCSize, cv::Vec3b (0, 0, 0));
		cv::Mat_<cv::Vec3b> _bPatch (NCCSize, NCCSize, cv::Vec3b (0, 0, 0));
		double skip = false;


		cv::Mat outA (panoA.rows, panoA.cols, CV_8UC3);
		cv::Mat_<cv::Vec3b> _outA = outA;
		panoA.copyTo(outA);

		cv::Mat outB (panoB.rows, panoB.cols, CV_8UC3);
		cv::Mat_<cv::Vec3b> _outB = outB;
		panoB.copyTo(outB);
		
		for (int j = 0; j < NCCSize; ++j) {
			for (int i = 0; i < NCCSize; ++i) {
				auto & A = NCCSquareA[j][i];
				auto & B = NCCSquareB[j][i];

				if(A[1] >= 0 && A[1] < panoA.rows
					&& A[0] >= 0 && A[0] < panoA.cols) {
					_aPatch(j,i) = _panoA(A[1], A[0]);

					_outA(A[1], A[0])[0] = 0;
					_outA(A[1], A[0])[1] = 0;
					_outA(A[1], A[0])[2] = 255;
				} else
					skip = true;

				if(B[1] >= 0 && B[1] < panoB.rows
					&& B[0] >= 0 && B[0] < panoB.cols) {

					_bPatch(j,i) = _panoB(B[1], B[0]);

					_outB(B[1], B[0])[0] = 0;
					_outB(B[1], B[0])[1] = 0;
					_outB(B[1], B[0])[2] = 255;
				} else
					skip = true;
			}
		}


		if (skip)
			continue;

		Eigen::Vector3d averageA = Eigen::Vector3d::Zero();
		Eigen::Vector3d averageB = Eigen::Vector3d::Zero();

		for (int j = 0; j < NCCSize; ++j) {
			for (int i = 0; i < NCCSize; ++i) {
				for (int c = 0; c < panoA.channels(); ++c) {
					averageA[c] += _aPatch(j,i)[c];
					averageB[c] += _bPatch(j,i)[c];
				}
			}
		}
		averageA /= NCCSize*NCCSize;
		averageB /= NCCSize*NCCSize;

		double AB = 0.0, AA = 0.0, BB = 0.0;
		for (int j = 0; j < NCCSize; ++j) {
			for (int i = 0; i < NCCSize; ++i) {
				for (int c = 0; c < panoA.channels(); ++c) {
					AA += (_aPatch(j,i)[c] - averageA[c])*(_aPatch(j,i)[c] - averageA[c]);
					BB += (_bPatch(j,i)[c] - averageB[c])*(_bPatch(j,i)[c] - averageB[c]);
					AB += (_bPatch(j,i)[c] - averageB[c])*(_aPatch(j,i)[c] - averageA[c]);
				}
			}
		}

		if (AB == 0 || AA == 0 || BB == 0)
			continue;

		++count;
		score += AB/sqrt(AA*BB);
		std::cout << AB/sqrt(AA*BB) << std::endl;

		cvNamedWindow("PanoA", CV_WINDOW_NORMAL);
		cvNamedWindow("PanoB", CV_WINDOW_NORMAL);
		cv::imshow("PanoA", outA);
		cv::imshow("PanoB", outB);
		cv::waitKey(0);
	}
	score /= count;
	std::cout << score << std::endl;

	


	return score;
}

void pano::voxelToWorld(std::vector<Eigen::Vector3d> & points, 
	const Eigen::Matrix3d & R, const Eigen::Vector3i & zeroZero,
	const double metersPerVoxel) {

	for (auto & p : points) {
		p = R*(p - zeroZero.cast<double>())*metersPerVoxel;
	}
}