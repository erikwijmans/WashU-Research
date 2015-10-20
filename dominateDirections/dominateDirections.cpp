#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d_omp.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
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
using namespace cv;
using namespace pcl;


int main(int argc, char const *argv[])
{
	if(argc != 2){
		cout << "Usage: dominateDirections.o <pathToPoints>.dat" << endl;
		return -1;
	}
	PointCloud<PointXYZ>::Ptr cloud (new PointCloud<PointXYZ>);


	const char * fileName = argv[1];
	

	ifstream scanFile (fileName, ios::in | ios::binary);

    int columns, rows;
   	scanFile.read(reinterpret_cast<char *> (& columns), sizeof(int));
   	scanFile.read(reinterpret_cast<char *> (& rows), sizeof(int));
    
    cout << "columns: " << columns << endl << "rows: " << rows << endl;
 
    vector<Vector3f > points;
    for (int k = 0; k < columns*rows; ++k) {
	    Vector3f point;
		scanFile.read(reinterpret_cast<char *> (&point[0]), sizeof(float)*3);

		if((k % 2000000) == 0){
	    	cout << "Working..." << k << endl;
	    }
		
		if(point[0] == 0 || point[1] == 0 || point[2] == 0)
			continue;
	  	

	    points.push_back(point);
	}
	scanFile.close();
	
	cloud->points.resize(points.size());
	for (int i = 0; i < points.size(); ++i)
	{
		cloud->points[i].x = points[i][0];
		cloud->points[i].y = points[i][1];
		cloud->points[i].z = points[i][2];
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
	ne.setRadiusSearch (0.01);

	// Compute the features
	ne.compute (*cloud_normals);



	cout << "Cloud size: " << cloud->points.size() << endl;
	cout << "Normal size: " << cloud_normals->points.size() << endl;

	ofstream binaryFile ("Output.dat", ios::out | ios::binary);
	size_t size = cloud_normals->points.size();
	binaryFile.write(reinterpret_cast<const char *> 
		(& size), sizeof(size_t));


	for (int i = 0; i < cloud_normals->points.size(); ++i)
	{
		float normals[3];
		normals[0] = cloud_normals->points[i].normal_x;
		normals[1] = cloud_normals->points[i].normal_y;
		normals[2] = cloud_normals->points[i].normal_z;
		
		binaryFile.write(reinterpret_cast<const char *> (&normals[0]),
			sizeof(float)*3);
	}

	binaryFile.close();






	
	return 0;
}