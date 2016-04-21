#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <Eigen/Sparse>
#include <iostream>
#include <string>
#include <dirent.h>
#include <fstream>
#include <random>

#include <omp.h>

#include <scan_gflags.h>
#include <scan_typedefs.hpp>

using namespace std;
using namespace Eigen;

void satoshiRansacManhattan1(const std::vector<Eigen::Vector3d> &, Vector3d &);
void satoshiRansacManhattan2(const std::vector<Eigen::Vector3d> &, const Vector3d &,
	Vector3d &, Vector3d &);
void getMajorAngles(const Vector3d &, vector<Matrix3d> &);
Matrix3d getRotationMatrix(const Vector3d &, const Vector3d &);
void analyzeNormals(const string &, const string &);

int main(int argc, char *argv[]) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	FLAGS_normalsFolder = FLAGS_dataPath + FLAGS_normalsFolder;
	FLAGS_rotFolder = FLAGS_dataPath + FLAGS_rotFolder;

	vector<string> normalsNames;

	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (FLAGS_normalsFolder.data())) != NULL) {
	  /* Add all the files and directories to a vector */
		while ((ent = readdir (dir)) != NULL) {
			string fileName = ent->d_name;
			if(fileName != ".." && fileName != ".") {
				normalsNames.push_back(fileName);
			}
		}
		closedir (dir);
	}  else {
	  /* could not open directory */
		perror ("");
		return EXIT_FAILURE;
	}

	sort(normalsNames.begin(), normalsNames.end());

	if (FLAGS_numScans == -1)
		FLAGS_numScans = normalsNames.size() - FLAGS_startIndex;

	for(int i = FLAGS_startIndex; i < FLAGS_numScans + FLAGS_startIndex; ++i) {
		const string normalsFilePath = FLAGS_normalsFolder + normalsNames[i];
		analyzeNormals(normalsFilePath, FLAGS_rotFolder);
	}

	return 0;
}

void analyzeNormals(const string & normalsFileName, const string & outputFolder) {
	vector<Vector3d> normals;
	const string number = normalsFileName.substr(normalsFileName.find(".") - 3, 3);
	const string buildName = normalsFileName.substr(normalsFileName.rfind("/") + 1, 3);
	const string rotOut = outputFolder + buildName + "_rotations_" + number + ".dat";

	cout << rotOut << endl;

	if(!FLAGS_redo) {
		ifstream out (rotOut, ios::in | ios::binary);
		if(out.is_open())
			return;
	}

	ifstream normalsFile (normalsFileName, ios::in | ios::binary);
	size_t count = 0;
	normalsFile.read(reinterpret_cast<char *> (&count), sizeof(size_t));
	normals.reserve(count);
	for (int i = 0; i < count; ++i) {
		Vector3f point;
		normalsFile.read(reinterpret_cast<char *> (point.data()), sizeof(Vector3f));
		normals.push_back(point.cast<double>());
	}
	normalsFile.close();

	if(!FLAGS_quiteMode)
		cout << "N size: " << normals.size() << endl;

	Vector3d d1, d2, d3;
	satoshiRansacManhattan1(normals, d1);
	if (!FLAGS_quiteMode) {
		cout << "D1: " << d1 << endl << endl;
	}
	std::vector<Eigen::Vector3d> N2;
	for (auto & n : normals)
		if(asin(n.cross(d1).norm()) > PI/2.0 - 0.02)
			N2.push_back(n);

	if(!FLAGS_quiteMode)
		cout << "N2 size: " << N2.size() << endl;

	satoshiRansacManhattan2(N2, d1, d2, d3);

	if(!FLAGS_quiteMode) {
		cout << "D2: " << d2 << endl << endl;
		cout << "D3: " << d3 << endl << endl;
	}

	vector<Matrix3d> R (4);

	if(abs(d1[2]) < 0.02)
		getMajorAngles(d1, R);
	else if( abs(d2[2]) < 0.02)
		getMajorAngles(d2, R);
	else
		getMajorAngles(d3, R);

	if (FLAGS_save) {
		ofstream binaryWriter (rotOut, ios::out | ios::binary);
		for(int i = 0; i < R.size(); ++i) {
			binaryWriter.write(reinterpret_cast<const char *> (R[i].data()),
				sizeof(Matrix3d));
		}
		binaryWriter.close();
	}
}


void satoshiRansacManhattan1(const std::vector<Eigen::Vector3d> & N, Vector3d & M) {
	const int m = N.size();

	volatile double maxInliers = 0, K = 1e5;
	volatile int k = 0;

	#pragma omp parallel shared(k, K, maxInliers, N, M)
	{
		std::random_device seed;
		std::mt19937_64 gen(seed());
		std::uniform_int_distribution<int> dist(0, m - 1);
		for (int i = 0; i < 5000; ++i)
			dist(gen);

		while (k < K) {
			// random sampling
			int randomIndex = dist(gen);
			// compute the model parameters
			const Eigen::Vector3d & nest = N[randomIndex];

			// counting inliers and outliers
			double numInliers = 0;
			Eigen::Vector3d average = Vector3d::Zero();
			for(auto & n : N) {
				if (acos(abs(nest.dot(n))) < 0.02) {
					++numInliers;
					if (nest.dot(n) < 0)
						average -= n;
					else
						average += n;
				}
			}

			#pragma omp crtical
			{
				if(numInliers > maxInliers) {
					maxInliers = numInliers;

					M = average/average.norm();

					double w = (numInliers-3)/m;
					double p = max(0.001,pow(w,3));
					K = log(1-0.999)/log(1-p);
				}
				if(k > 10000) k = 10*k;
				++k;
			}
		}
	}
}

void satoshiRansacManhattan2(const std::vector<Eigen::Vector3d> & N, const Vector3d & n1,
	Vector3d & M1, Vector3d & M2) {
	const int m = N.size();

	volatile double maxInliers = 0, K = 1.0e5;
	volatile int k = 0;

	#pragma omp parallel shared (k, K, maxInliers, N, n1, M1, M2)
	{
		std::random_device seed;
		std::mt19937_64 gen(seed());
		std::uniform_int_distribution<int> dist(0, m - 1);
		for (int i = 0; i < 5000; ++i)
			dist(gen);

		while (k < K) {
			// random sampling
			int randomIndex = dist(gen);
			// compute the model parameters
			const Eigen::Vector3d & nest = N[randomIndex];

			const Eigen::Vector3d nest2 = nest.cross(n1);

			// counting inliers and outliers
			double numInliers = 0, numInliers2 = 0;
			Eigen::Vector3d average = Vector3d::Zero(),
				average2 = Vector3d::Zero();
			for(auto & n : N) {
				if(min(acos(abs(nest.dot(n))),acos(abs(nest2.dot(n)))) < 0.02) {
					if(acos(abs(nest.dot(n))) < 0.02) {
						++numInliers;
						if (nest.dot(n) < 0)
							average -= n;
						else
							average += n;
					} else {
						++numInliers2;
						if (nest2.dot(n) < 0)
							average2 -= n;
						else
							average2 += n;
					}
				}
			}

			#pragma omp crtical
			{
				if((numInliers + numInliers2) > maxInliers) {
					maxInliers = numInliers + numInliers2;

					if(numInliers > numInliers2) {
						average /= average.norm();
						M1 = average;
						M2 = average.cross(n1);
					} else {
						average2 /= average2.norm();
						M1 = average2.cross(n1);
						M2 = average2;
					}

					double w = (maxInliers-3)/m;
					double p = max(0.001,pow(w,3));
					K = log(1-0.999)/log(1-p);
				}

				if(k > 10000) k = 10*K;
				++k;
			}
		}
	}
}

void getMajorAngles(const Vector3d & M, vector<Matrix3d> & R) {
	R[0] = getRotationMatrix(Vector3d::UnitX(), M);
	R[1] = getRotationMatrix(Vector3d::UnitY(), M);
	R[2] = getRotationMatrix(-1.0*Vector3d::UnitX(), M);
	R[3] = getRotationMatrix(-1.0*Vector3d::UnitY(), M);
}


static Matrix3d crossProductMatrix(const Vector3d & vector) {
	Matrix3d scratch = Matrix3d::Zero();

	scratch(1,0) = vector[2];
	scratch(2,0) = -vector[1];
	scratch(0,1) = -vector[2];
	scratch(2,1) = vector[0];
	scratch(0,2) = vector[1];
	scratch(1,2) = -vector[0];

	return scratch;
}


Matrix3d getRotationMatrix(const Vector3d & end, const Vector3d & start) {

	if(acos(abs(start.dot(end))) < 0.005) {
		if( start.dot(end) > 0)
			return Matrix3d::Identity();
		if(start.dot(end) < 0)
			return -1.0*Matrix3d::Identity();
	}

	Vector3d v = start.cross(end);
	double s = v.dot(v);
	double c = start.dot(end);

	Matrix3d vx = crossProductMatrix(v);

	Matrix3d out = Matrix3d::Identity() + vx + vx*vx*(1 - c)/s;
	out(2, 2) = 1.0;
	out(0, 2) = 0;
	out(1, 2) = 0;
	out(2, 0) = 0;
	out(2, 1) = 0;

	return out;
}