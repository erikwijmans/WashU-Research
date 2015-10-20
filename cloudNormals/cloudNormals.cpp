#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d_omp.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <dirent.h>
#include <vector>
#include <fstream>
#include <algorithm>

using namespace std;
using namespace Eigen;
using namespace pcl;


void calculateNormals(const string &, const string &);
void createBoundingBox(float *, float*, const vector<Vector3f> &);

int main(int argc, char const *argv[])
{
	if(argc != 3){
		cout << "Usage: dominateDirections.o <pathToPoints>/ <pathToOutput>/" << endl;
		return -1;
	}
	
	std::vector<string> pointClouds;
	const char * fileFolderIn = argv[1];
	const char * fileFolderOut = argv[2];
	
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (fileFolderIn)) != NULL) {
	  /* Add all the files and directories to a vector */
	  while ((ent = readdir (dir)) != NULL) {
	  	string fileName = ent->d_name;
	  	if(fileName != ".." && fileName != "."){
	  		pointClouds.push_back(fileName);
	  	}
	  }
	  closedir (dir);
	}  else {
	  /* could not open directory */
	  perror ("");
	  return EXIT_FAILURE;
	}

	sort(pointClouds.begin(), pointClouds.end());
	for(auto & pointCloud : pointClouds)
	{
		const string pointCloudName = fileFolderIn + pointCloud;
		const string normalCloudName = fileFolderOut + 
			pointCloud.substr(0,pointCloud.find(".")) + ".dat";
		calculateNormals(pointCloudName, normalCloudName);
	}
	
	return 0;
}


void calculateNormals(const string & inFile, const string & outFile){
	ifstream scanFile (inFile, ios::in | ios::binary);
	cout << outFile << endl;

    int columns, rows;
   	scanFile.read(reinterpret_cast<char *> (& columns), sizeof(int));
   	scanFile.read(reinterpret_cast<char *> (& rows), sizeof(int));
    
    vector<Vector3f > points;
    for (int k = 0; k < columns*rows; ++k) {
	    Vector3f point;
		scanFile.read(reinterpret_cast<char *> (&point[0]), sizeof(float)*3);

		if (k% 5 != 0)
			continue;

		if(point[0] == 0 || point[1] == 0 || point[2] == 0)
			continue;
	  	

	    points.push_back(point);
	}

	scanFile.close();

	float pointMin[3], pointMax[3];
	createBoundingBox(pointMin, pointMax, points);


	PointCloud<PointXYZ>::Ptr cloud (new PointCloud<PointXYZ>);

	for (int i = 0; i < points.size(); ++i)
	{
		if(points[i][0] < pointMin[0] || points[i][0] > pointMax[0])
			continue;
		if(points[i][1] < pointMin[1] || points[i][1] > pointMax[1])
			continue;
		if(points[i][2] < pointMin[2] || points[i][2] > pointMax[2])
			continue;
		PointXYZ point (points[i][0], points[i][1], points[i][2]);
		cloud->points.push_back(point);
	}

	// Create the normal estimation class, and pass the input dataset to it
	NormalEstimationOMP<PointXYZ, Normal> ne;
	ne.setInputCloud (cloud);

	// Create an empty kdtree representation, and pass it to the normal estimation object.
	// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree 
		(new pcl::search::KdTree<pcl::PointXYZ> ());
	ne.setSearchMethod (tree);

	// Output datasets
	PointCloud<Normal>::Ptr cloud_normals (new PointCloud<Normal>);

	// Use all neighbors in a sphere of radius 3cm
	ne.setRadiusSearch (0.03);

	// Compute the features
	ne.compute (*cloud_normals);


	cout << "Cloud size: " << cloud->points.size() << endl;
	cout << "Normal size: " << cloud_normals->points.size() << endl;

	ofstream binaryFile (outFile, ios::out | ios::binary);
	size_t size = cloud_normals->points.size();
	binaryFile.write(reinterpret_cast<const char *> 
		(& size), sizeof(size_t));


	for (int i = 0; i < cloud_normals->points.size(); ++i)
	{
		Vector3f normals;
		normals[0] = cloud_normals->points[i].normal_x;
		normals[1] = cloud_normals->points[i].normal_y;
		normals[2] = cloud_normals->points[i].normal_z;
		
		binaryFile.write(reinterpret_cast<const char *> (&normals),
			sizeof(Vector3f));
	}

	binaryFile.close();
}

void createBoundingBox(float * pointMin, float * pointMax,
	const vector<Vector3f > & points){
	double averageX, averageY, sigmaX, sigmaY, averageZ, sigmaZ;
	averageX = averageY = sigmaX = sigmaY = averageZ = sigmaZ = 0;

	for (auto & point : points)
	{
		averageX += point[0];
		averageY += point[1];
		averageZ += point[2];
	}
	averageX = averageX/points.size();
	averageY = averageY/points.size();
	averageZ = averageZ/points.size();

	for (auto & point : points)
	{
		sigmaX += (point[0] - averageX)*(point[0] - averageX);
		sigmaY += (point[1] - averageY)*(point[1] - averageY);
		sigmaZ += (point[2] - averageZ)*(point[2] - averageZ);
	}
	sigmaX = sigmaX/(points.size()-1);
	sigmaY = sigmaY/(points.size()-1);
	sigmaZ = sigmaZ/(points.size()-1);
	sigmaX = sqrt(sigmaX);
	sigmaY = sqrt(sigmaY);
	sigmaZ = sqrt(sigmaZ);

	double dX = 1.1*5*sigmaX;
	double dY = 1.1*5*sigmaX;
	double dZ = 1.1*5*sigmaZ;

    pointMax[0] = pointMax[1] = pointMax[2] 
    	= pointMin[0] = pointMin[1] = pointMin[2] = 0;
	for(auto & point : points){
		if(point[0] <= (averageX - dX/2) || (point[0] >= averageX + dX/2) ||
			(point[1] <= averageY - dY/2) || (point[1] >= averageY + dY/2) ||
			(point[2] <= averageZ - dZ/2 || point[2] >= averageZ + dZ/2))
			continue;

		for (int a = 0; a < 3; ++a) {
	  		pointMax[a] = max(pointMax[a], point[a]);
	  		pointMin[a] = min(pointMin[a], point[a]);
	  	}

			
	}

}