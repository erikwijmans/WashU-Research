#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <Eigen/Sparse>
#include <stdio.h>
#include <iostream>
#include <string>
#include <dirent.h>
#include <vector>
#include <fstream>
#include <math.h>
#include <time.h>
#include <omp.h>

#include <scan_gflags.h>

using namespace std;
using namespace Eigen;

void satoshiRansacManhattan1(const VectorXd &, Vector3d &);
void satoshiRansacManhattan2(const VectorXd &, const Vector3d &, 
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

	sort(normalsNames.begin(), normalsNames.end(), 
		[](const std::string & a, const std::string & b) {
			int numA = std::stoi(a.substr(a.find(".") - 3, 3));
			int numB = std::stoi(b.substr(b.find(".") - 3, 3));
			return numA < numB;
		});

	//#pragma omp parallel for schedule(dynamic) shared(normalsNames)
	for(int i = FLAGS_startIndex; i < normalsNames.size() - FLAGS_startIndex; ++i) {
		const string normalsFilePath = FLAGS_normalsFolder + normalsNames[i];
		analyzeNormals(normalsFilePath, FLAGS_rotFolder);
	}
	


	return 0;
}

void analyzeNormals(const string & normalsFileName, const string & outputFolder) {
	vector<Vector3d> normals;
	const string number = normalsFileName.substr(normalsFileName.find(".") - 3, 3);
	const string buildName = normalsFileName.substr(normalsFileName.rfind("/") + 1, 3);
	const string rotOut = outputFolder + buildName + "_rotation_" + number + ".dat";

	cout << rotOut << endl;

	if(!FLAGS_redo) {
		ifstream out (rotOut, ios::in | ios::binary);
		if(out.is_open())
			return;
	}


	ifstream normalsFile (normalsFileName, ios::in | ios::binary);
	size_t count = 0;
	normalsFile.read(reinterpret_cast<char *> (&count), sizeof(size_t));
	for (int i = 0; i < count; ++i)
	{
		Vector3f point;
		normalsFile.read(reinterpret_cast<char *> (point.data()), sizeof(Vector3f));
		Vector3d tmp;
		tmp[0] = point[0];
		tmp[1] = point[1];
		tmp[2] = point[2];
		normals.push_back(tmp);
	}
	normalsFile.close();

	VectorXd N (normals.size()*3);
	for (int i = 0; i < normals.size(); ++i)
	{
		N[i*3] = normals[i][0];
		N[i*3 + 1] = normals[i][1];
		N[i*3 + 2] = normals[i][2];
	}
	if(!FLAGS_quiteMode)
		cout << "N size: " << N.size() << endl; 

	Vector3d d1, d2, d3;
	satoshiRansacManhattan1(N, d1);
	d1 /= d1.norm();
	if (!FLAGS_quiteMode) {
		cout << "D1: " << d1 << endl << endl;
	}
	VectorXd N2;
	int index = 0;
	for (auto & normal : normals)
	{
		if(abs(normal.dot(d1)) < 0.02)
		{	
			N2.resize(index*3 + 3);
			N2[index*3] = normal[0];
			N2[index*3 + 1] = normal[1];
			N2[index*3 + 2] = normal[2];
			++index;
		}
	}
	if(!FLAGS_quiteMode)
		cout << "N2 size: " << N2.size() << endl;

	satoshiRansacManhattan2(N2, d1, d2, d3);
	d2 /= d2.norm();
	d3 /= d3.norm();

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

	ofstream binaryWriter (rotOut, ios::out | ios::binary);
	for(int i = 0; i < R.size(); ++i) {
		binaryWriter.write(reinterpret_cast<const char *> (R[i].data()),
			sizeof(Matrix3d));
	}
	binaryWriter.close();
}


void satoshiRansacManhattan1(const VectorXd & N, Vector3d & M) {

	int m = static_cast<int>(N.size()/3.0);
	
	double maxInliers = 0;
	double K = 1.0e5;
	int k=0;

	int randomIndex;
	Vector3d nest;
	Vector3d ndata;

	while(k < K) {
		// random sampling
		randomIndex = rand()%m;
		// compute the model parameters
		nest[0] = N[3*randomIndex+0];
		nest[1] = N[3*randomIndex+1];
		nest[2] = N[3*randomIndex+2];			

		// counting inliers and outliers
		double numInliers = 0;
		Vector3d average = Vector3d::Zero();
		for(int i=0;i<m;i++) {
			ndata[0] = N[3*i+0];
			ndata[1] = N[3*i+1];
			ndata[2] = N[3*i+2];

			if (acos(abs(nest.dot(ndata))) < 0.02) {
				++numInliers;
				if (nest.dot(ndata) < 0)
					average -= ndata;
				else
					average += ndata;
			}
		}
		
		
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

void satoshiRansacManhattan2(const VectorXd & N, const Vector3d & n1, 
	Vector3d & M1, Vector3d & M2) {
	
	int m = static_cast<int>(N.size()/3);
	
	double maxInliers = 0;
	double K = 1.0e5;
	int k=0;

	int randomIndex;
	Vector3d nest;
	Vector3d nest2;
	Vector3d ndata;		

	while(k < K) {
		// random sampling
		randomIndex = rand()%m;
		// compute the model parameters
		nest[0] = N[3*randomIndex+0];
		nest[1] = N[3*randomIndex+1];
		nest[2] = N[3*randomIndex+2];	

		nest2 = nest.cross(n1);

		// counting inliers and outliers
		double numInliers = 0, numInliers2 = 0;
		Vector3d average = Vector3d::Zero(), average2 = Vector3d::Zero();
		for(int i=0;i<m;i++) {
			ndata[0] = N[3*i+0];
			ndata[1] = N[3*i+1];
			ndata[2] = N[3*i+2];

			if(min(acos(abs(nest.dot(ndata))),acos(abs(nest2.dot(ndata)))) < 0.02) {
				if(acos(abs(nest.dot(ndata))) < 0.02) {
					++numInliers;
					if (nest.dot(ndata) < 0)
						average -= ndata;
					else
						average += ndata;
				} else {
					++numInliers2;
					if (nest2.dot(ndata) < 0)
						average2 -= ndata;
					else 
						average2 += ndata;
				}
			}		
		}

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

	return Matrix3d::Identity() + vx + vx*vx*(1 - c)/s;
}