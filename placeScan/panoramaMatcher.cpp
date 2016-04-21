#include "placeScan_panoramaMatcher.h"

#include <iostream>
#include "opencv2/xfeatures2d.hpp"

static const int NCCSize = 20;

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

static Eigen::Vector2d voxelSpaceToPanorama(const Eigen::Vector3d & voxelCoord,
	double & r, const cv::Size & resolution) {
	Eigen::Vector3d local = voxelCoord;
	local[1] *= -1.0;
	return pointCloudToPanorama(local, r, resolution);
}

static Eigen::Vector2d pointCloudToPanorama(const Eigen::Vector3d & worldCoord,
	const cv::Size & resolution) {
	auto polar = cartesianToPolar(worldCoord);
	double x = (polar[1]/PI + 1.0)*(resolution.width - 1.0)/2.0;
	double y = polar[2]/maxPhi*(resolution.height - 1.0);
	return Eigen::Vector2d (x, y);
}

static Eigen::Vector2d voxelSpaceToPanorama(const Eigen::Vector3d & voxelCoord,
	const cv::Size & resolution) {
	Eigen::Vector3d local = voxelCoord;
	local[1] *= -1.0;
	return pointCloudToPanorama(local, resolution);
}

static Eigen::Vector3d panoramaToPointCloud(const Eigen::Vector2d & panoCoord,
	const double r, const cv::Size & resolution) {
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

static inline double getAngle(const Eigen::Vector3d & a,
	const Eigen::Vector3d & b) {
	return std::acos(a.dot(b)/(a.norm()*b.norm()));
}

double pano::compareSIFT2(const place::Panorama & panoA,
	const place::Panorama & panoB, const Eigen::Matrix3d & RA,
	const Eigen::Matrix3d & RB, const Eigen::Vector3d & aToB,
	const Eigen::Vector3d & bToA) {

	constexpr double cutoffAngle = PI/4.5;

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

	const cv::Mat & imgA = panoA.img;
	const cv::Mat & imgB = panoB.img;

	#pragma omp parallel sections num_threads(2)\
		reduction(+: score, count) if (false)
	{
		#pragma omp section
		{
			std::vector<int> indiciesInA;
			std::vector<cv::Point2f> falsePointsInB;
			cv::Mat falseBDescrip;
			for (int i = 0; i < truePointsInA.size(); ++i) {
				const cv::Point2f & kp = truePointsInA[i];
				Eigen::Vector2d coordA (kp.x, kp.y);
				const double r = rMapA(coordA[1], coordA[0]);
				auto aVoxelSpace = panoramaToVoxelSpace(coordA, r, imgA.size());
				auto bVoxelSpace = aToBRotMat*aVoxelSpace + aToBTrans;
				auto coordB = voxelSpaceToPanorama(bVoxelSpace, imgB.size());

				if (coordB[0] < 0 || coordB[0] >= std::min(imgB.cols, (int)rMapB.cols()))
					continue;
				if (coordB[1] < 0 || coordB[1] >= std::min(imgB.rows, (int)rMapB.rows()))
					continue;

				const double radiusA = aVoxelSpace.norm();
				const double radiusB = bVoxelSpace.norm();
				const double trueRadiusB = rMapB(coordB[1], coordB[0]);
				const double angle = getAngle(aVoxelSpace, bVoxelSpace);
				if (radiusA/radiusB >= 4.0 || radiusB/radiusA >= 4.0
					|| angle > cutoffAngle || std::abs(radiusB - trueRadiusB) > 0.2)
					continue;

				falsePointsInB.emplace_back(coordB[0], coordB[1]);
				indiciesInA.push_back(i);
			}

			std::vector<cv::KeyPoint> falseKeypointsInB;
			cv::KeyPoint::convert(falsePointsInB, falseKeypointsInB);
			cv::Ptr<cv::Feature2D> s = cv::xfeatures2d::SIFT::create();
			s->compute(imgB, falseKeypointsInB, falseBDescrip);

			for (int i = 0; i < indiciesInA.size(); ++i) {
				const int trueIndex = indiciesInA[i];
				cv::Mat t = trueADescrip.row(trueIndex);
				cv::Mat f = falseBDescrip.row(i);
				score += cv::norm(t, f, cv::NORM_L2);
				++count;
			}
		}
		#pragma omp section
		{
			std::vector<int> indiciesInB;
			std::vector<cv::Point2f> falsePointsInA;
			cv::Mat falseADescrip;
			for (int i = 0; i < truePointsInB.size(); ++i) {
				const cv::Point2f & kp = truePointsInB[i];
				Eigen::Vector2d coordB (kp.x, kp.y);
				const double r = rMapB(coordB[1], coordB[0]);
				auto bVoxelSpace = panoramaToVoxelSpace(coordB, r, imgB.size());
				auto aVoxelSpace = bToARotMat*bVoxelSpace + bToATrans;
				auto coordA = voxelSpaceToPanorama(aVoxelSpace, imgA.size());

				if (coordA[0] < 0 || coordA[0] >= std::min(imgA.cols, (int)rMapA.cols()))
					continue;
				if (coordA[1] < 0 || coordA[1] >= std::min(imgA.rows, (int)rMapA.rows()))
					continue;

				const double radiusA = aVoxelSpace.norm();
				const double radiusB = bVoxelSpace.norm();
				const double trueRadiusA = rMapA(coordA[1], coordA[0]);
				const double angle = getAngle(aVoxelSpace, bVoxelSpace);
				if (radiusA/radiusB >= 4.0 || radiusB/radiusA >= 4.0
					|| angle > cutoffAngle || std::abs(radiusA - trueRadiusA) > 0.2)
					continue;

				falsePointsInA.emplace_back(coordA[0], coordA[1]);
				indiciesInB.push_back(i);
			}

			std::vector<cv::KeyPoint> falseKeypointsInA;
			cv::KeyPoint::convert(falsePointsInA, falseKeypointsInA);
			cv::Ptr<cv::Feature2D> s = cv::xfeatures2d::SIFT::create();
			s->compute(imgA, falseKeypointsInA, falseADescrip);

			for (int i = 0; i < indiciesInB.size(); ++i) {
				const int trueIndex = indiciesInB[i];
				cv::Mat t = trueBDescrip.row(trueIndex);
				cv::Mat f = falseADescrip.row(i);
				score += cv::norm(t, f, cv::NORM_L2);
				++count;
			}
		}
	}
	score /= count*1000.0;
	/*if (false) {
		std::cout << score << std::endl;
		cv::Mat out1, out2, out3, out4;

		cv::drawKeypoints(imgA, trueKeypointsInA, out1);
		cvNamedWindow("TrueA", CV_WINDOW_NORMAL);
		cv::imshow("TrueA", out1);

		cv::drawKeypoints(imgA, falseKeypointsInA, out2);
		cvNamedWindow("FalseA", CV_WINDOW_NORMAL);
		cv::imshow("FalseA", out2);

		cv::drawKeypoints(imgB, trueKeypointsInB, out3);
		cvNamedWindow("TrueB", CV_WINDOW_NORMAL);
		cv::imshow("TrueB", out3);

		cv::drawKeypoints(imgB, falseKeypointsInB, out4);
		cvNamedWindow("FalseB", CV_WINDOW_NORMAL);
		cv::imshow("FalseB", out4);
		cv::waitKey(0);
	}*/


	return count > 0 ? 1.0 - score : 0;
}

constexpr bool viz = true;

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