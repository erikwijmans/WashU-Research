#include "placeScan_panoramaMatcher.h"

#include <iostream>
#include <fstream>
#include <opencv2/xfeatures2d.hpp>

static constexpr int NCCSize = 11;
static_assert(NCCSize%2 == 1, "NCCSize isn't odd");

static constexpr double degreesToRadians(double degrees) {
	return degrees*PI/180.0;
}

static Eigen::Vector3d cartesianToPolar(const Eigen::Vector3d & coords) {
	double r = coords.norm();
	double theta = atan2(coords[1], coords[0]);
	double phi = acos(coords[2]/r);
	return Eigen::Vector3d (r, theta, phi);
}

static Eigen::Vector3d polarToCartesian(const Eigen::Vector3d & coords) {
	double x = coords[0]*cos(coords[1])*sin(coords[2]);
	double y = coords[0]*sin(coords[1])*sin(coords[2]);
	double z = coords[0]*cos(coords[2]);
	return Eigen::Vector3d (x,y,z);
}

static Eigen::Vector2d pointCloudToPanorama(const Eigen::Vector3d & worldCoord,
	double & r, const cv::Size & resolution) {
	auto polar = cartesianToPolar(worldCoord);
	double x = (polar[1]/PI + 1.0)*(resolution.width - 1.0)/2.0;
	double y = polar[2]/maxPhi*(resolution.height - 1.0);
	r = polar[0];
	return Eigen::Vector2d (x, y);
}

static Eigen::Vector2d pointCloudToPanorama(const Eigen::Vector3d & worldCoord,
	const cv::Size & resolution) {
	double tmp;
	return pointCloudToPanorama(worldCoord, tmp, resolution);
}

static Eigen::Vector2d voxelSpaceToPanorama(const Eigen::Vector3d & voxelCoord,
	double & r, const cv::Size & resolution) {
	Eigen::Vector3d local = voxelCoord;
	local[1] *= -1.0;
	return pointCloudToPanorama(local, r, resolution);
}

static Eigen::Vector2d voxelSpaceToPanorama(const Eigen::Vector3d & voxelCoord,
	const cv::Size & resolution) {
	double tmp;
	return voxelSpaceToPanorama(voxelCoord, tmp, resolution);
}

static Eigen::Vector3d panoramaToPointCloud(const Eigen::Vector2d & panoCoord,
	const double r, const cv::Size & resolution) {
	const int offset = 1.005*resolution.width/2.0;
	double theta = (2.0*panoCoord[0]/(resolution.width - 1.0) - 1.0)*PI;
	double phi = panoCoord[1]*maxPhi/(resolution.height - 1.0);
	return polarToCartesian(Eigen::Vector3d (r, theta, phi));
}

static Eigen::Vector3d panoramaToVoxelSpace(const Eigen::Vector2d & panoCoord,
	const double r, const cv::Size & resolution) {
	auto voxel =  panoramaToPointCloud(panoCoord, r, resolution);
	voxel[1] *= -1.0;
	return voxel;
}

static double aveDepth(const Eigen::RowMatrixXf & depthMap,
	const int x, const int y) {
	int count = 0;
	double ave = 0;
	constexpr int limit = NCCSize/2;
	static_assert(limit*2 + 1 == NCCSize, "limit is wrong");

	for (int j = -limit; j <= limit ; ++j) {
		for (int i = -limit; i <= limit; ++i) {
			if (x + i < 0 || x + i >= depthMap.cols())
				continue;
			if (y + j < 0 || y + j >= depthMap.rows())
				continue;
			if (!depthMap(y + j, x + i))
				continue;
			ave += depthMap(y + j, x + i);
			++count;
		}
	}
	return count > 0 ? ave/count : 0;
}

double pano::compareSIFT(const cv::Mat & panoA, const cv::Mat & panoB,
	const std::vector<Eigen::Vector3d> & pointsInA,
	const std::vector<Eigen::Vector3d> & pointsInB) {

	cv::Mat aVis (panoA.size(), CV_8UC3);
	panoA.copyTo(aVis);
	cv::Mat_<cv::Vec3b> _aVis = aVis;

	cv::Mat bVis (panoB.size(), CV_8UC3);
	panoB.copyTo(bVis);
	cv::Mat_<cv::Vec3b> _bVis = bVis;

	double score = 0.0;
	int count = 0;

	#pragma omp parallel shared (panoA, panoB, pointsInA, pointsInB)
	{
		cv::Ptr<cv::Feature2D> SIFT = cv::xfeatures2d::SIFT::create();
		double r;

		#pragma omp for reduction(+: score, count)
		for (int k = 0; k < pointsInA.size(); ++k) {

			/*std::cout << "A: " << pointsInA[k] << std::endl << std::endl;
			std::cout << "B: " << pointsInB[k] << std::endl << std::endl;*/

			auto panoCoordA = pointCloudToPanorama(pointsInA[k], panoA.size());
			auto panoCoordB = pointCloudToPanorama(pointsInB[k], panoB.size());

			if (panoCoordA[0] - 10 < 0 || panoCoordA[0] + 10 >= panoA.cols)
				continue;
			if (panoCoordA[1] - 10 < 0 || panoCoordA[1] + 10 >= panoA.rows)
				continue;
			if (panoCoordB[0] - 10 < 0 || panoCoordB[0] + 10 >= panoB.cols)
				continue;
			if (panoCoordB[1] - 10 < 0 || panoCoordB[1] + 10 >= panoB.rows)
				continue;


			double radiusA = pointsInA[k].norm();
			double radiusB = pointsInB[k].norm();

			/*std::cout << "RA: " << radiusA << std::endl;
			std::cout << "RB: " << radiusB << std::endl;*/

			cv::Mat scaledA = panoA, scaledB = panoB;
			double level = 0;
			while (radiusA/radiusB > 2.0 || radiusB/radiusA > 2.0) {
				double scaleFactor = pow(2, -(++level));
				if (radiusA/radiusB > 2.0) {
					scaledA = cv::Mat();
					cv::resize(panoA, scaledA, cv::Size(), scaleFactor, scaleFactor, CV_INTER_AREA);
					panoCoordA /= 2.0;
					radiusA /= 4.0;
				} else if (radiusB/radiusA > 2.0) {
					scaledB = cv::Mat();
					cv::resize(panoB, scaledB, cv::Size(), scaleFactor, scaleFactor, CV_INTER_AREA);
					panoCoordB /= 2.0;
					radiusB /= 4.0;
				}
			}

			cv::Mat grayA, grayB;
			cv::cvtColor(scaledA, grayA, CV_RGB2GRAY);
			cv::cvtColor(scaledB, grayB, CV_RGB2GRAY);

			std::vector<cv::KeyPoint> KPA, KPB;
			cv::KeyPoint::convert(std::vector<cv::Point2f> (1, cv::Point2f (panoCoordA[0],
				panoCoordA[1])), KPA);
			cv::KeyPoint::convert(std::vector<cv::Point2f> (1, cv::Point2f (panoCoordB[0],
				panoCoordB[1])), KPB);

			cv::Mat descripA, descripB;
			SIFT->compute(grayA, KPA, descripA);
			SIFT->compute(grayB, KPB, descripB);
			score += cv::norm(descripA, descripB, cv::NORM_L2);
			++count;

			if (false) {
				for (int j = -15; j <= 15; ++j) {
					int row = panoCoordA[1]*pow(2, level) + j;
					for (int i = -15; i <= 15; ++i) {
						int col = panoCoordA[0]*pow(2, level) + i;
						if (col < 0 || col >= aVis.cols)
							continue;
						if (row < 0 || row >= aVis.rows)
							continue;
						_aVis(row, col)[0] = 0;
						_aVis(row, col)[1] = 0;
						_aVis(row, col)[2] = 255;
					}
				}

				for (int j = -15; j <= 15; ++j) {
					int row = panoCoordB[1]*pow(2, level) + j;
					for (int i = -15; i <= 15; ++i) {
						int col = panoCoordB[0]*pow(2, level) + i;
						if (col < 0 || col >= bVis.cols)
							continue;
						if (row < 0 || row >= bVis.rows)
							continue;
						_bVis(row, col)[0] = 0;
						_bVis(row, col)[1] = 0;
						_bVis(row, col)[2] = 255;
					}
				}
				cvNamedWindow("A", CV_WINDOW_NORMAL);
				cvNamedWindow("B", CV_WINDOW_NORMAL);
				cv::imshow("A", aVis);
				cv::imshow("B", bVis);
				cv::waitKey(0);

			}
		}
	}


	score /= count*1000.0;
	return count > 0 ? -1.0*score : 0;
}

#pragma omp declare reduction \
  (merge : std::vector<cv::Point2f> : omp_out.insert(omp_out.end(), omp_in.begin(),\
    omp_in.end()))

#pragma omp declare reduction \
  (merge : std::vector<int> : omp_out.insert(omp_out.end(), omp_in.begin(),\
    omp_in.end()))

#pragma omp declare reduction \
  (merge : std::vector<cv::KeyPoint> : omp_out.insert(omp_out.end(), omp_in.begin(),\
    omp_in.end()))

static inline double getAngle(const auto & a,
	const auto & b) {
	if (a.norm() == 0 || b.norm() == 0) return PI;
	return std::acos(static_cast<double>(a.dot(b))/(a.norm()*b.norm()));
}

static void SIFTHeatMap(place::Panorama & panoA,
	place::Panorama & panoB, const Eigen::Matrix3d & RA,
	const Eigen::Matrix3d & RB, const Eigen::Vector3d & aToB,
	const Eigen::Vector3d & bToA) {

	const Eigen::Matrix3d aToBRotMat = RB*RA.inverse();
	const Eigen::Vector3d aToBTrans = RB*aToB;

	const Eigen::Matrix3d bToARotMat = RA*RB.inverse();
	const Eigen::Vector3d bToATrans = RA*bToA;

	auto & imgA = panoA[0];
	auto & imgB = panoB[0];

	const int rows = std::min(imgA.rows,
		imgB.rows);
	const int cols = std::min(imgA.cols, imgB.cols);
	Eigen::RowMatrixXd heatMap (rows, cols);
	std::vector<cv::Point2f> ptA, ptB;
	#pragma omp parallel
	{
		std::vector<cv::Point2f> privateptA, privateptB;
		#pragma omp for nowait
		for (int j = 0; j < rows; ++j) {
			for (int i = 0; i < cols; ++i) {
				Eigen::Vector2d coordA (i, j);
				const double r = panoA.rMap(j, i);
				auto aVoxelSpace = panoramaToVoxelSpace(coordA, r, imgA.size());
				auto bVoxelSpace = aToBRotMat*aVoxelSpace + aToBTrans;
				auto coordB = voxelSpaceToPanorama(bVoxelSpace, imgB.size());

				privateptA.emplace_back(coordA[0], coordA[1]);
				privateptB.emplace_back(coordB[0], coordB[1]);
			}
		}
		#pragma omp for schedule(static) ordered
		for (int i = 0; i < omp_get_num_threads(); ++i) {
		  #pragma omp ordered
		  {
		  	ptA.insert(ptA.end(), privateptA.begin(), privateptA.end());
		  	ptB.insert(ptB.end(), privateptB.begin(), privateptB.end());
		  }
		}
	}
	std::vector<cv::KeyPoint> kpA, kpB;
	cv::KeyPoint::convert(ptA, kpA);
	cv::KeyPoint::convert(ptB, kpB);

	cv::Mat grayA, grayB;
	cv::cvtColor(imgA, grayA, CV_RGB2GRAY);
	cv::cvtColor(imgB, grayB, CV_RGB2GRAY);

	cv::imwrite("A.png", grayA);
	cv::imwrite("B.png", grayB);

	std::ofstream outWriter ("data.dat", std::ios::out | std::ios::binary);
	int numKP = kpA.size();
	outWriter.write(reinterpret_cast<const char *>(&numKP), sizeof(int));
	for (auto & kp : kpA)
		outWriter.write(reinterpret_cast<const char *>(&kp), sizeof(cv::KeyPoint));

	for (auto & kp : kpB)
		outWriter.write(reinterpret_cast<const char *>(&kp), sizeof(cv::KeyPoint));

	std::cout << "saved" << std::endl;
	exit(1);
}

/*double pano::compareSIFT2(place::Panorama & panoA,
	place::Panorama & panoB, const Eigen::Matrix3d & RA,
	const Eigen::Matrix3d & RB, const Eigen::Vector3d & aToB,
	const Eigen::Vector3d & bToA) {

	SIFTHeatMap(panoB, panoB, RB, RA, bToA, aToB);

	constexpr double cutoffAngle = degreesToRadians(45);
	constexpr double maxRadiusRatio = 2.0;

	const Eigen::Matrix3d aToBRotMat = RB*RA.inverse();
	const Eigen::Vector3d aToBTrans = RB*aToB;

	const Eigen::Matrix3d bToARotMat = RA*RB.inverse();
	const Eigen::Vector3d bToATrans = RA*bToA;

	double score = 0.0;
	int count = 0;

	const std::vector<cv::Point2f> & truePointsInA = panoA.keypoints;
	const std::vector<cv::Point2f> & truePointsInB = panoB.keypoints;

	const Eigen::RowMatrixXf & rMapA = panoA.rMap;
	const Eigen::RowMatrixXf & rMapB = panoB.rMap;

	const cv::Mat & trueADescrip = panoA.descriptors;
	const cv::Mat & trueBDescrip = panoB.descriptors;

	#pragma omp parallel reduction(+:count, score)
	{
		cv::Ptr<cv::Feature2D> s = cv::xfeatures2d::SIFT::create();
		#pragma omp for nowait
		for (int i = 0; i < truePointsInA.size(); ++i) {
			const cv::Point2f & kp = truePointsInA[i];
			Eigen::Vector2d coordA (kp.x, kp.y);
			const double r = rMapA(coordA[1], coordA[0]);
			auto aVoxelSpace = panoramaToVoxelSpace(coordA, r, panoA[0].size());
			auto bVoxelSpace = aToBRotMat*aVoxelSpace + aToBTrans;
			auto coordB = voxelSpaceToPanorama(bVoxelSpace, panoB[0].size());

			if (coordB[0] < 0 || coordB[0] >= std::min(panoB[0].cols, (int)rMapB.cols()))
				continue;
			if (coordB[1] < 0 || coordB[1] >= std::min(panoB[0].rows, (int)rMapB.rows()))
				continue;

			const double radiusA = aVoxelSpace.norm();
			const double radiusB = bVoxelSpace.norm();
			const double trueRadiusB = rMapB(coordB[1], coordB[0]);
			const double aveDepthB = aveDepth(rMapB, coordB[0], coordB[1]);
			const double aveDepthA = aveDepth(rMapA, coordA[0], coordA[1]);
			const double angle = getAngle(aVoxelSpace, bVoxelSpace);
			if (angle > cutoffAngle || std::abs(radiusB - trueRadiusB) > 0.2
				|| std::abs(aveDepthA - r) > 0.2 || std::abs(trueRadiusB - aveDepthB) > 0.2)
				continue;

			int aLevel = floor(radiusA/radiusB/2.0);
			int bLevel = floor(radiusB/radiusA/2.0);

			auto a = voxelSpaceToPanorama(aVoxelSpace, panoA[aLevel].size());
			auto b = voxelSpaceToPanorama(bVoxelSpace, panoB[bLevel].size());

			std::vector<cv::KeyPoint> kpB, kpA;
			cv::KeyPoint::convert({cv::Point2f(b[0], b[1])}, kpB);
			cv::KeyPoint::convert({cv::Point2f(a[0], a[1])}, kpA);

			cv::Mat aDes, bDes;
			s->compute(panoB[bLevel], kpB, bDes);
			s->compute(panoA[aLevel], kpA, aDes);

			const double norm = cv::norm(aDes, bDes, cv::NORM_L2);
			if (Eigen::numext::isfinite(norm)) {
				score += norm;
				++count;
			}
		}
		#pragma omp for nowait
		for (int i = 0; i < truePointsInB.size(); ++i) {
			const cv::Point2f & kp = truePointsInB[i];
			Eigen::Vector2d coordB (kp.x, kp.y);
			const double r = rMapB(coordB[1], coordB[0]);
			auto bVoxelSpace = panoramaToVoxelSpace(coordB, r, panoB[0].size());
			auto aVoxelSpace = bToARotMat*bVoxelSpace + bToATrans;
			auto coordA = voxelSpaceToPanorama(aVoxelSpace, panoA[0].size());

			if (coordA[0] < 0 || coordA[0] >= std::min(panoA[0].cols, (int)rMapA.cols()))
				continue;
			if (coordA[1] < 0 || coordA[1] >= std::min(panoA[0].rows, (int)rMapA.rows()))
				continue;

			const double radiusA = aVoxelSpace.norm();
			const double radiusB = bVoxelSpace.norm();
			const double trueRadiusA = rMapA(coordA[1], coordA[0]);
			const double aveDepthB = aveDepth(rMapB, coordB[0], coordB[1]);
			const double aveDepthA = aveDepth(rMapA, coordA[0], coordA[1]);
			const double angle = getAngle(aVoxelSpace, bVoxelSpace);
			if (angle > cutoffAngle || std::abs(radiusA - trueRadiusA) > 0.2
				|| std::abs(aveDepthA - trueRadiusA) > 0.2 || std::abs(r- aveDepthB) > 0.2)
				continue;

			int aLevel = floor(radiusA/radiusB/2.0);
			int bLevel = floor(radiusB/radiusA/2.0);

			auto a = voxelSpaceToPanorama(aVoxelSpace, panoA[aLevel].size());
			auto b = voxelSpaceToPanorama(bVoxelSpace, panoB[bLevel].size());

			std::vector<cv::KeyPoint> kpB, kpA;
			cv::KeyPoint::convert({cv::Point2f(b[0], b[1])}, kpB);
			cv::KeyPoint::convert({cv::Point2f(a[0], a[1])}, kpA);

			cv::Mat aDes, bDes;
			s->compute(panoB[bLevel], kpB, bDes);
			s->compute(panoA[aLevel], kpA, aDes);

			const double norm = cv::norm(aDes, bDes, cv::NORM_L2);
			if (Eigen::numext::isfinite(norm)) {
				score += norm;
				++count;
			}
		}
	}
	score /= count*1000.0;
	score = 1.0 - score;

	return count > 0 && Eigen::numext::isfinite(score) ? score : 0;
}*/

double NCC(const cv::Mat_<cv::Vec3b> & a, const cv::Mat_<cv::Vec3b> & b) {
	assert(a.rows == b.rows);
	assert(a.cols == b.cols);
	assert(a.channels() == b.channels());

	Eigen::VectorXd aveA = Eigen::VectorXd::Zero(a.channels()),
		aveB = Eigen::VectorXd::Zero(b.channels());

	for (int j = 0; j < a.rows; ++j) {
		for (int i = 0; i < a.cols; ++i) {
			for (int c = 0; c < a.channels(); ++c) {
				aveA[c] += a(j, i)[c];
				aveB[c] += b(j, i)[c];
			}
		}
	}

	aveA /= a.rows*a.cols;
	aveB /= a.rows*a.cols;


	double AB = 0.0, AA = 0.0, BB = 0.0;
	for (int j = 0; j < a.rows; ++j) {
		for (int i = 0; i < a.cols; ++i) {
			for (int c = 0; c < a.channels(); ++c) {
				AA += (a(j, i)[c] - aveA[c])*(a(j, i)[c] - aveA[c]);
				BB += (b(j, i)[c] - aveB[c])*(b(j, i)[c] - aveB[c]);
				AB += (b(j, i)[c] - aveB[c])*(a(j, i)[c] - aveA[c]);
			}
		}
	}
	const double score = AB/sqrt(AA*BB);
	return score;
}

constexpr bool viz = false;

double pano::compare(const cv::Mat & panoA, const cv::Mat & panoB,
		const Eigen::Matrix3d & RA, const Eigen::Matrix3d & RB,
		const std::vector<Eigen::Vector3d> & pointsInA,
		const Eigen::Vector3d & translation,
		const std::vector<Eigen::Vector3d> & pointsInB,
		const Eigen::RowMatrixXd & rMap) {

	std::cout << translation << std::endl << std::endl;

	const Eigen::Matrix3d deltaRot = RB*RA.inverse();
	const Eigen::Vector3d rotatedTranslation = RB*translation;

	const cv::Mat_<cv::Vec3b> _panoA = panoA;
	const cv::Mat_<cv::Vec3b> _panoB = panoB;

	int count = 0;
	double score = 0.0;
	#pragma omp parallel for reduction(+: score, count) if(!viz)
	for (int i = 0; i < pointsInA.size(); ++i) {

		auto panoCoordA = pointCloudToPanorama(pointsInA[i], panoA.size());
		const double r = rMap(panoCoordA[1], panoCoordA[0]);

		Eigen::Vector2d NCCSquareA [NCCSize][NCCSize];
		Eigen::Vector2d NCCSquareB [NCCSize][NCCSize];
		Eigen::Vector2d offset (floor(NCCSize/2), floor(NCCSize/2));
		for (int j = 0; j < NCCSize; ++j) {
			for (int i = 0; i < NCCSize; ++i) {
				NCCSquareA[j][i] = panoCoordA - offset + Eigen::Vector2d(j,i);
			}
		}

		for (int j = 0; j < NCCSize; ++j) {
			for (int i = 0; i < NCCSize; ++i) {
				auto aVoxelSpace = panoramaToVoxelSpace(NCCSquareA[j][i], r, panoA.size());
				auto bVoxelSpace = deltaRot*aVoxelSpace + rotatedTranslation;
				NCCSquareB[j][i] = voxelSpaceToPanorama(bVoxelSpace, panoB.size());
			}
		}

		cv::Mat_<cv::Vec3b> _aPatch (NCCSize, NCCSize, cv::Vec3b (0, 0, 0));
		cv::Mat_<cv::Vec3b> _bPatch (NCCSize, NCCSize, cv::Vec3b (0, 0, 0));
		bool skip = false;

		cv::Mat outA, outB;
		cv::Mat_<cv::Vec3b> _outA, _outB;
		if (viz) {
			outA = cv::Mat (panoA.rows, panoA.cols, CV_8UC3);
			_outA = outA;
			panoA.copyTo(outA);

			outB = cv::Mat (panoB.rows, panoB.cols, CV_8UC3);
			_outB = outB;
			panoB.copyTo(outB);
		}


		for (int j = 0; j < NCCSize; ++j) {
			for (int i = 0; i < NCCSize; ++i) {
				auto & A = NCCSquareA[j][i];
				auto & B = NCCSquareB[j][i];

				if(A[1] >= 0 && A[1] < panoA.rows
					&& A[0] >= 0 && A[0] < panoA.cols) {
					_aPatch(j,i) = _panoA(A[1], A[0]);

					if (viz) {
						_outA(A[1], A[0])[0] = 0;
						_outA(A[1], A[0])[1] = 0;
						_outA(A[1], A[0])[2] = 255;
					}

				} else
					skip = true;

				if(B[1] >= 0 && B[1] < panoB.rows
					&& B[0] >= 0 && B[0] < panoB.cols) {

					_bPatch(j,i) = _panoB(B[1], B[0]);

					if (viz) {
						_outB(B[1], B[0])[0] = 0;
						_outB(B[1], B[0])[1] = 0;
						_outB(B[1], B[0])[2] = 255;
					}
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

		if (viz) {
			cvNamedWindow("PanoA", CV_WINDOW_NORMAL);
			cvNamedWindow("PanoB", CV_WINDOW_NORMAL);
			cv::imshow("PanoA", outA);
			cv::imshow("PanoB", outB);
			cv::waitKey(0);
		}
	}
	score /= count;

	return count > 0 ? score : 0;
}

void pano::voxelGridToWorld(std::vector<Eigen::Vector3d> & points,
	const Eigen::Matrix3d & R, const Eigen::Vector3i & zeroZero,
	const double metersPerVoxel) {

	for (auto & p : points) {
		p = R*(p - zeroZero.cast<double>())*metersPerVoxel;
		p[1] *= -1.0;
	}
}

double pano::compareNCC2(place::Panorama & panoA,
	place::Panorama & panoB, const Eigen::Matrix3d & RA,
	const Eigen::Matrix3d & RB, const Eigen::Vector3d & aToB,
	const Eigen::Vector3d & bToA) {

	constexpr double cutoffAngle = degreesToRadians(25);
	constexpr double maxDiff = 0.3;
	constexpr double occulisionCutoff = 0.3;
	constexpr double roundingOffset = 0.5;
	constexpr int offset = NCCSize/2;
	static_assert(offset*2 + 1 == NCCSize, "offset isn't correct");

	const Eigen::Matrix3d aToBRotMat = RB*RA.inverse();
	const Eigen::Vector3d aToBTrans = RB*aToB;

	const Eigen::Matrix3d bToARotMat = RA*RB.inverse();
	const Eigen::Vector3d bToATrans = RA*bToA;

	double score = 0.0;
	int count = 0;

	const std::vector<cv::Point2f> & truePointsInA = panoA.keypoints;
	const std::vector<cv::Point2f> & truePointsInB = panoB.keypoints;

	const Eigen::RowMatrixXf & rMapA = panoA.rMap;
	const Eigen::RowMatrixXf & rMapB = panoB.rMap;

	const Eigen::Vector2i aMaxes (std::min(panoA[0].cols, (int)rMapA.cols()), std::min(panoA[0].rows, (int)rMapA.rows())),
		bMaxes (std::min(panoB[0].cols, (int)rMapB.cols()), std::min(panoB[0].rows, (int)rMapB.rows()));

	#pragma omp parallel reduction(+:count, score) if (!viz)
	{
		cv::Mat_<cv::Vec3b> NCCA, NCCB;
		#pragma omp for nowait
		for (int i = 0; i < truePointsInA.size(); ++i) {
			const cv::Point2f & kp = truePointsInA[i];
			Eigen::Vector2d coordA (kp.x, kp.y);
			const double r = rMapA(coordA[1], coordA[0]);
			if (r < 0.5) continue;
			auto aVoxelSpace = panoramaToVoxelSpace(coordA, r, panoA[0].size());
			auto bVoxelSpace = aToBRotMat*aVoxelSpace + aToBTrans;
			auto coordB = voxelSpaceToPanorama(bVoxelSpace, panoB[0].size());

			if (coordB[0] < 0 || coordB[0] >= bMaxes[0])
				continue;
			if (coordB[1] < 0 || coordB[1] >= bMaxes[1])
				continue;

			const double radiusA = aVoxelSpace.norm();
			const double radiusB = bVoxelSpace.norm();
			const double trueRadiusB = rMapB(coordB[1], coordB[0]);
			const double aveDepthB = aveDepth(rMapB, coordB[0], coordB[1]);
			const double aveDepthA = aveDepth(rMapA, coordA[0], coordA[1]);
			const double angle = getAngle(aVoxelSpace, bVoxelSpace);
			const double normalAngle = getAngle(panoA.surfaceNormals(coordA[1], coordA[0]),
				panoB.surfaceNormals(coordB[1], coordB[0]));
			if (angle > cutoffAngle || std::abs(radiusB - trueRadiusB) > occulisionCutoff
				|| std::abs(aveDepthA - r) > maxDiff || std::abs(trueRadiusB - aveDepthB) > maxDiff
				|| normalAngle > cutoffAngle)
				continue;

			int aLevel = floor(radiusB/radiusA/2.0 + roundingOffset);
			int bLevel = floor(radiusA/radiusB/2.0 + roundingOffset);

			auto a = voxelSpaceToPanorama(aVoxelSpace, panoA[aLevel].size());
			auto b = voxelSpaceToPanorama(bVoxelSpace, panoB[bLevel].size());

			const cv::Mat_<cv::Vec3b> aLvlImg = panoA[aLevel];
			const cv::Mat_<cv::Vec3b> bLvlImg = panoB[bLevel];

			if (a[1] - offset < 0 || a[1] + offset >= aLvlImg.rows
				|| a[0] - offset < 0 || a[0] + offset >= aLvlImg.cols)
				continue;

			if (b[1] - offset < 0 || b[1] + offset >= bLvlImg.rows
				|| b[0] - offset < 0 || b[0] + offset >= bLvlImg.cols)
				continue;

			NCCB = bLvlImg(cv::Range(b[1] - offset, b[1] + offset + 1),
				cv::Range(b[0] - offset, b[0] + offset + 1));
			NCCA = aLvlImg(cv::Range(a[1] - offset, a[1] + offset + 1),
				cv::Range(a[0] - offset, a[0] + offset + 1));

			auto mean = cv::mean(NCCA);
			if (mean.val[0] > 240 && mean.val[1] > 240 && mean.val[2] > 240)
				continue;

			mean = cv::mean(NCCB);
			if (mean.val[0] > 240 && mean.val[1] > 240 && mean.val[2] > 240)
				continue;

			const double ncc = NCC(NCCA, NCCB);
			if (!Eigen::numext::isfinite(ncc)) continue;
			score += ncc;
			++count;

			if (viz) {
				cv::Mat_<cv::Vec3b> out1 (aLvlImg.size());
				aLvlImg.copyTo(out1);
				cv::Mat_<cv::Vec3b> out2 (bLvlImg.size());
				bLvlImg.copyTo(out2);

				for (int j = 0; j < NCCSize; ++j) {
					for (int i = 0; i < NCCSize; ++i) {
						out1(j - offset + a[1], i - offset + a[0])[0] = 0;
						out1(j - offset + a[1], i - offset + a[0])[1] = 0;
						out1(j - offset + a[1], i - offset + a[0])[2] = 255;

						out2(j - offset + b[1], i - offset + b[0])[0] = 0;
						out2(j - offset + b[1], i - offset + b[0])[1] = 0;
						out2(j - offset + b[1], i - offset + b[0])[2] = 255;
					}
				}
				std::cout << aLevel << "  " << bLevel << std::endl;
				std::cout << ncc << std::endl;
				cvNamedWindow("A", CV_WINDOW_NORMAL);
				cv::imshow("A", out1);

				cvNamedWindow("B", CV_WINDOW_NORMAL);
				cv::imshow("B", out2);

				cvNamedWindow("NCCA", CV_WINDOW_NORMAL);
				cv::imshow("NCCA", NCCA);

				cvNamedWindow("NCCB", CV_WINDOW_NORMAL);
				cv::imshow("NCCB", NCCB);

				cv::waitKey(0);
			}
		}
		#pragma omp for nowait
		for (int i = 0; i < truePointsInB.size(); ++i) {
			const cv::Point2f & kp = truePointsInB[i];
			Eigen::Vector2d coordB (kp.x, kp.y);
			const double r = rMapB(coordB[1], coordB[0]);
			if (r < 0.5) continue;
			auto bVoxelSpace = panoramaToVoxelSpace(coordB, r, panoB[0].size());
			auto aVoxelSpace = bToARotMat*bVoxelSpace + bToATrans;
			auto coordA = voxelSpaceToPanorama(aVoxelSpace, panoA[0].size());

			if (coordA[0] < 0 || coordA[0] >= aMaxes[0])
				continue;
			if (coordA[1] < 0 || coordA[1] >= aMaxes[1])
				continue;

			const double radiusA = aVoxelSpace.norm();
			const double radiusB = bVoxelSpace.norm();
			const double trueRadiusA = rMapA(coordA[1], coordA[0]);
			const double aveDepthB = aveDepth(rMapB, coordB[0], coordB[1]);
			const double aveDepthA = aveDepth(rMapA, coordA[0], coordA[1]);
			const double angle = getAngle(aVoxelSpace, bVoxelSpace);
			const double normalAngle = getAngle(panoA.surfaceNormals(coordA[1], coordA[0]),
				panoB.surfaceNormals(coordB[1], coordB[0]));
			if (angle > cutoffAngle || std::abs(radiusA - trueRadiusA) > occulisionCutoff
				|| std::abs(aveDepthA - trueRadiusA) > maxDiff || std::abs(r- aveDepthB) > maxDiff
				|| normalAngle > cutoffAngle)
				continue;

			int aLevel = floor(radiusB/radiusA/2.0 + roundingOffset);
			int bLevel = floor(radiusA/radiusB/2.0 + roundingOffset);

			auto a = voxelSpaceToPanorama(aVoxelSpace, panoA[aLevel].size());
			auto b = voxelSpaceToPanorama(bVoxelSpace, panoB[bLevel].size());

			const cv::Mat_<cv::Vec3b> aLvlImg = panoA[aLevel];
			const cv::Mat_<cv::Vec3b> bLvlImg = panoB[bLevel];

			if (a[1] - offset < 0 || a[1] + offset >= aLvlImg.rows
				|| a[0] - offset < 0 || a[0] + offset >= aLvlImg.cols)
				continue;

			if (b[1] - offset < 0 || b[1] + offset >= bLvlImg.rows
				|| b[0] - offset < 0 || b[0] + offset >= bLvlImg.cols)
				continue;

			NCCB = bLvlImg(cv::Range(b[1] - offset, b[1] + offset + 1),
				cv::Range(b[0] - offset, b[0] + offset + 1));
			NCCA = aLvlImg(cv::Range(a[1] - offset, a[1] + offset + 1),
				cv::Range(a[0] - offset, a[0] + offset + 1));

			const double ncc = NCC(NCCA, NCCB);
			if (!Eigen::numext::isfinite(ncc)) continue;
			score += ncc;
			++count;

			if (viz) {
				cv::Mat_<cv::Vec3b> out1 (aLvlImg.rows, aLvlImg.cols);
				aLvlImg.copyTo(out1);
				cv::Mat_<cv::Vec3b> out2 (bLvlImg.rows, bLvlImg.cols);
				bLvlImg.copyTo(out2);

				for (int j = 0; j < NCCSize; ++j) {
					for (int i = 0; i < NCCSize; ++i) {
						out1(j - offset + a[1], i - offset + a[0])[0] = 0;
						out1(j - offset + a[1], i - offset + a[0])[1] = 0;
						out1(j - offset + a[1], i - offset + a[0])[2] = 255;

						out2(j - offset + b[1], i - offset + b[0])[0] = 0;
						out2(j - offset + b[1], i - offset + b[0])[1] = 0;
						out2(j - offset + b[1], i - offset + b[0])[2] = 255;
					}
				}
				std::cout << aLevel << "  " << bLevel << std::endl;
				std::cout << ncc << std::endl;
				cvNamedWindow("A", CV_WINDOW_NORMAL);
				cv::imshow("A", out1);

				cvNamedWindow("B", CV_WINDOW_NORMAL);
				cv::imshow("B", out2);

				cvNamedWindow("NCCA", CV_WINDOW_NORMAL);
				cv::imshow("NCCA", NCCA);

				cvNamedWindow("NCCB", CV_WINDOW_NORMAL);
				cv::imshow("NCCB", NCCB);

				cv::waitKey(0);
			}
		}
	}
	score /= count;

	if (count < 0.005*(panoA.keypoints.size() + panoB.keypoints.size()))
		count = 0;

	return count > 0 && Eigen::numext::isfinite(score) ? score : 0;
}
