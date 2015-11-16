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

#include <gflags/gflags.h>



using namespace std;
using namespace Eigen;

void satoshiRansacManhattan1(const VectorXd &, Vector3d &);
void satoshiRansacManhattan2(const VectorXd &, const Vector3d &, 
	Vector3d &, Vector3d &);
void getMajorAngles(const Vector3d &, vector<Matrix3d> &);
Matrix3d getRotationMatrix(const Vector3d &, const Vector3d &);
void analyzeNormals(const string &, const string &);

DEFINE_bool(redo, false, "Redo all the cloud_normals");
DEFINE_bool(verbose, false, "Turns on all prints");
DEFINE_string(outFolder, "/home/erik/Projects/3DscanData/DUC/Floor1/densityMaps/rotations/", 
	"Path to binary files");
DEFINE_string(inFolder, "/home/erik/Projects/3DscanData/DUC/Floor1/cloudNormals/",
	"Path to Output");
DEFINE_int32(startIndex, 0 , "bleh");

int main(int argc, char *argv[])
{
	
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	vector<string> normalsNames;

	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (FLAGS_inFolder.data())) != NULL) {
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

	for(int i = 0; i < normalsNames.size(); ++i) {
		const string normalsFilePath = FLAGS_inFolder + normalsNames[i];
		analyzeNormals(normalsFilePath, FLAGS_outFolder);
	}
	


	return 0;
}

void analyzeNormals(const string & normalsFileName, const string & outputFolder) {
	vector<Vector3d> normals;
	const string number = normalsFileName.substr(normalsFileName.find(".") - 3, 3);
	const string rotOut = outputFolder + "DUC_rotation_" + number + ".dat";

	if(FLAGS_verbose)
		cout << number << endl;
	
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
		normalsFile.read(reinterpret_cast<char *> (&point), sizeof(Vector3f));
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

	Vector3d d1, d2, d3;
	satoshiRansacManhattan1(N, d1);

	VectorXd N2;
	int index = 0;
	for (auto & normal : normals)
	{
		if(abs(normal.dot(d1)) <0.01)
		{	
			N2.resize(index*3 + 3);
			N2[index*3] = normal[0];
			N2[index*3 + 1] = normal[1];
			N2[index*3 + 2] = normal[2];
			index++;
		}
	}


	satoshiRansacManhattan2(N2, d1, d2, d3);

	if(FLAGS_verbose) {
		cout << "D1: " << d1 << endl << endl;
		cout << "D2: " << d2 << endl << endl;
		cout << "D3: " << d3 << endl << endl;
	}

	vector<Matrix3d> R (4);

	if(abs(d1[2]) < 0.01)
		getMajorAngles(d1, R);
	else if( abs(d2[2]) < 0.01)
		getMajorAngles(d2, R);
	else
		getMajorAngles(d3, R);

	ofstream binaryWriter (rotOut, ios::out | ios::binary);
	for(int i = 0; i < R.size(); ++i) {
		binaryWriter.write(reinterpret_cast<const char *> (&R[i]),
			sizeof(Matrix3d));
	}
	binaryWriter.close();
}


void satoshiRansacManhattan1(const VectorXd & N, Vector3d & M)
{
	int m = static_cast<int>(N.size()/3.0);
	
	double maxInliers = 0;
	double K = 1.0e5;
	int k=0;
	int randomIndex;
	Vector3d nest;
	Vector3d ndata;
	#pragma omp parallel private(randomIndex, nest, ndata) shared(N, M, maxInliers, K, k, m)
	{
		while(k < K) {

			
			// random sampling
			randomIndex = rand()%m;
			// compute the model parameters
			nest[0] = N[3*randomIndex+0];
			nest[1] = N[3*randomIndex+1];
			nest[2] = N[3*randomIndex+2];			


			// counting inliers and outliers
			double numInliers = 0;
			for(int i=0;i<m;i++) {
				ndata[0] = N[3*i+0];
				ndata[1] = N[3*i+1];
				ndata[2] = N[3*i+2];

				if(acos(abs(nest.dot(ndata))) < 0.02)
					numInliers++;
			}

			
			#pragma omp critical
			{
				k++;
				if(numInliers > maxInliers) {
				
					maxInliers = numInliers;
					M[0] = nest[0];
					M[1] = nest[1];
					M[2] = nest[2];

					double w = (numInliers-3)/m;
					double p = max(0.001,pow(w,3));
					K = log(1-0.999)/log(1-p);
				}
			}
		}
	}
	
}

void satoshiRansacManhattan2(const VectorXd & N, const Vector3d & n1, 
	Vector3d & M1, Vector3d & M2)
{
	int m = static_cast<int>(N.size()/3);
	
	double maxInliers = 0;
	double K = 1.0e5;
	int k=0;
	int randomIndex;
	Vector3d nest;
	Vector3d nest2;
	Vector3d ndata;

	#pragma omp parallel private(randomIndex, nest, nest2, ndata) shared(N, n1, M1, M2, maxInliers, K, k, m)
	{
		while(k < K) {

			
			// random sampling
			randomIndex = rand()%m;
			// compute the model parameters
			nest[0] = N[3*randomIndex+0];
			nest[1] = N[3*randomIndex+1];
			nest[2] = N[3*randomIndex+2];	

			nest2 = nest.cross(n1);

			// counting inliers and outliers
			double numInliers = 0;
			for(int i=0;i<m;i++) {
				ndata[0] = N[3*i+0];
				ndata[1] = N[3*i+1];
				ndata[2] = N[3*i+2];

				if(min(acos(abs(nest.dot(ndata))),acos(abs(nest2.dot(ndata)))) < 0.02)
					numInliers++;
			}

			
			#pragma omp critical
			{
				k++;
				if(numInliers > maxInliers) {
					maxInliers = numInliers;
					M1[0] = nest[0];
					M1[1] = nest[1];
					M1[2] = nest[2];
					M2[0] = nest2[0];
					M2[1] = nest2[1];
					M2[2] = nest2[2];

					double w = (numInliers-3)/m;
					double p = max(0.001,pow(w,3));
					K = log(1-0.999)/log(1-p);	
				}
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
	Matrix3d scratch;

	scratch(1,0) = vector[2];
	scratch(2,0) = -vector[1];
	scratch(0,1) = -vector[2];
	scratch(2,1) = vector[0];
	scratch(0,2) = vector[1];
	scratch(1,2) = -vector[0];

	return scratch;
}

Matrix3d getRotationMatrix(const Vector3d & end, const Vector3d & start) {
	Vector3d v = start.cross(end);
	double s = v.dot(v);
	double c = start.dot(end);

	Matrix3d vx = crossProductMatrix(v);

	return Matrix3d::Identity() + vx + vx*vx*(1 - c)/s;
}