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



using namespace std;
using namespace Eigen;

void satoshiRansacManhattan1(const VectorXd &, Vector3d &);
void satoshiRansacManhattan2(const VectorXd &, const Vector3d &, 
	Vector3d &, Vector3d &);
void getMajorAngles(const Vector3d &, vector<Matrix3d> &);
Matrix3d getRotationMatrix(const Vector3d &, const Vector3d &);
Matrix3d crossProductMatrix(const Vector3d &);
void analyzeNormals(const string &, const string &);

void usage(){
	cout << "Usage:  ./getRotations <cloudNormalsFolder>/ <rotationOutputFolder>/" << endl;
	exit(-1);
}


int main(int argc, char const *argv[])
{
	if(argc != 3)
		usage();

	vector<string> noramlsNames;
	const char * fileFolderIn = argv[1];
	const string fileFolderOut = argv[2];

	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (fileFolderIn)) != NULL) {
	  /* Add all the files and directories to a vector */
	  while ((ent = readdir (dir)) != NULL) {
	  	string fileName = ent->d_name;
	  	if(fileName != ".." && fileName != "."){
	  		noramlsNames.push_back(fileName);
	  	}
	  }
	  closedir (dir);
	}  else {
	  /* could not open directory */
	  perror ("");
	  return EXIT_FAILURE;
	}

	sort(noramlsNames.begin(), noramlsNames.end());
	for(auto & name : noramlsNames){
		string normalsFilePath = fileFolderIn + name;
		analyzeNormals(normalsFilePath, fileFolderOut);
	}
	


	return 0;
}

void analyzeNormals(const string & normalsFileName, const string & outputFolder){
	vector<Vector3d> normals;
	const string number = normalsFileName.substr(normalsFileName.find(".") - 3, 3);
	const string rotOut = outputFolder + "DUC_rotation_" + number + ".dat";
	cout << rotOut << endl;


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

	vector<Matrix3d> R (4);

	if(abs(d1[2]) < 0.01)
		getMajorAngles(d1, R);
	else if( abs(d2[2]) < 0.01)
		getMajorAngles(d2, R);
	else
		getMajorAngles(d3, R);

	ofstream binaryWriter (rotOut, ios::out | ios::binary);
	for(int i = 0; i < R.size(); ++i){
		binaryWriter.write(reinterpret_cast<const char *> (&R[i]),
			sizeof(Matrix3d));
	}
	binaryWriter.close();
}


void satoshiRansacManhattan1(const VectorXd & N, Vector3d & M)
{
  
	
	int m = static_cast<int>(N.size()/3.0);
	
	int id;
	double x,y,z;
	double param[4];
	double maxInliers = 0;
	double K = 1.0e5;
	int k=0;
	int randomIndex;
	double theta;
	double phi;
	Vector3d nest;
	Vector3d ndata;
	//mexPrintf("%f %f %f\n",N[0],N[1],N[2]);

	while(k < K){

		k++;
		// random sampling
		randomIndex = rand()%m;
		// compute the model parameters
		nest[0] = N[3*randomIndex+0];
		nest[1] = N[3*randomIndex+1];
		nest[2] = N[3*randomIndex+2];			


		// counting inliers and outliers
		double numInliers = 0;
		for(int i=0;i<m;i++){
			ndata[0] = N[3*i+0];
			ndata[1] = N[3*i+1];
			ndata[2] = N[3*i+2];

			if(acos(abs(nest.dot(ndata))) < 0.03)
				numInliers++;
		}

		double w = (numInliers-3)/m;

		if(numInliers > maxInliers){
			maxInliers = numInliers;
			M[0] = nest[0];
			M[1] = nest[1];
			M[2] = nest[2];

			double p = max(0.001,pow(w,3));
			K = log(1-0.999)/log(1-p);
		}
		if(k>100000) break;
	}
	
}

void satoshiRansacManhattan2(const VectorXd & N, const Vector3d & n1, 
	Vector3d & M1, Vector3d & M2)
{
  
	
	int m = static_cast<int>(N.size()/3);
    
	
	int id;
	double x,y,z;
	double param[4];
	double maxInliers = 0;
	double K = 1.0e5;
	int k=0;
	int randomIndex;
	double theta;
	double phi;
	Vector3d nest;
	Vector3d nest2;
	Vector3d ndata;
	//mexPrintf("%f %f %f\n",N[0],N[1],N[2]);

	while(k < K){

		k++;
		// random sampling
		randomIndex = rand()%m;
		// compute the model parameters
		nest[0] = N[3*randomIndex+0];
		nest[1] = N[3*randomIndex+1];
		nest[2] = N[3*randomIndex+2];	

		nest2 = nest.cross(n1);

		// counting inliers and outliers
		double numInliers = 0;
		for(int i=0;i<m;i++){
			ndata[0] = N[3*i+0];
			ndata[1] = N[3*i+1];
			ndata[2] = N[3*i+2];

			if(min(acos(abs(nest.dot(ndata))),acos(abs(nest2.dot(ndata)))) < 0.03)
				numInliers++;
		}

		double w = (numInliers-3)/m;

		if(numInliers > maxInliers){
			maxInliers = numInliers;
			M1[0] = nest[0];
			M1[1] = nest[1];
			M1[2] = nest[2];
			M2[0] = nest2[0];
			M2[1] = nest2[1];
			M2[2] = nest2[2];

			double p = max(0.001,pow(w,3));
			K = log(1-0.999)/log(1-p);
		}
		if(k>100000) break;
	}
	
}

void getMajorAngles(const Vector3d & M, vector<Matrix3d> & R){
	R[0] = getRotationMatrix(Vector3d::UnitX(), M);
	R[1] = getRotationMatrix(Vector3d::UnitY(), M);
	R[2] = getRotationMatrix(-1.0*Vector3d::UnitX(), M);
	R[3] = getRotationMatrix(-1.0*Vector3d::UnitY(), M);
}


Matrix3d getRotationMatrix(const Vector3d & end, const Vector3d & start){
    Vector3d v = start.cross(end);
    double s = v.dot(v);
    double c = start.dot(end);


    Matrix3d vx = crossProductMatrix(v);
    Matrix3d sq_vx = vx*vx;
  
    return Matrix3d::Identity() + vx + sq_vx*(1 - c)/s;
}


Matrix3d crossProductMatrix(const Vector3d & vector){
    Matrix3d scratch = Matrix3d::Zero();

    scratch(1,0) = vector[2];
    scratch(2,0) = -vector[1];
    scratch(0,1) = -vector[2];
    scratch(2,1) = vector[0];
    scratch(0,2) = vector[1];
    scratch(1,2) = -vector[0];

    return scratch;
}