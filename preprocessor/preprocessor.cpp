#include <scan_typedefs.hpp>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/StdVector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <gflags/gflags.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <dirent.h>
#include <vector>
#include <fstream>
#include <math.h>
#include <time.h>
#include <algorithm>



#include <pcl/point_types.h>
#include <pcl/common/io.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/keypoints/impl/sift_keypoint.hpp>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/visualization/pcl_visualizer.h>

DEFINE_bool(save, true, "save the output");
DEFINE_bool(redo, false, "Redo all the binaryFiles and panoramas");
DEFINE_string(dataPath, "/home/erik/Projects/3DscanData/DUC/Floor1/", "Path to data");
DEFINE_string(outFolder, "binaryFiles/", 
	"Path to binary files");
DEFINE_string(inFolder, "PTXFiles/",
	"Path to Output");
DEFINE_string(panoFolder, "panoramas/", "Path to write panoramas");
DEFINE_string(normalsFolder, "cloudNormals/", "name of normalsFolder");
DEFINE_string(descriptorsFolder, "descriptors/", "name of descriptorsFolder");


typedef pcl::PointXYZRGB PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT1344 DescriptorType;

float descr_rad_ (0.02f);

void convertToBinary(const std::string & fileNameIn,
	const std::string &,
	std::vector<scan::PointXYZRGB> & pointCloud);
void createPanorama(const std::vector<scan::PointXYZRGB> & pointCloud,
	const std::string & outName);
void boundingBox(const std::vector<scan::PointXYZRGB> & points,
	Eigen::Vector3f & pointMin, Eigen::Vector3f & pointMax);
void createPCLPointCloud(const std::vector<scan::PointXYZRGB> & points, 
	pcl::PointCloud<PointType>::Ptr & cloud);
void getNormals(const pcl::PointCloud<PointType>::Ptr & cloud,
	pcl::PointCloud<NormalType>::Ptr & cloud_normals,
	std::vector<int> & index);
void saveNormals(const pcl::PointCloud<NormalType>::Ptr & cloud_normals,
	const std::string & outName);
void getDescriptors(const pcl::PointCloud<PointType>::Ptr & cloud,
	const std::vector<int> & index,
	const pcl::PointCloud<NormalType>::Ptr & cloud_normals,
	pcl::PointCloud<DescriptorType>::Ptr & cloud_descriptors);
void saveDescriptors(const pcl::PointCloud<PointType>::Ptr & cloud,
	const pcl::PointCloud<DescriptorType>::Ptr & cloud_descriptors,
	const std::string & outName);

int main(int argc, char *argv[]) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	FLAGS_outFolder = FLAGS_dataPath + FLAGS_outFolder;
	FLAGS_inFolder = FLAGS_dataPath + FLAGS_inFolder;
	FLAGS_panoFolder = FLAGS_dataPath + FLAGS_panoFolder;
	FLAGS_normalsFolder = FLAGS_dataPath + FLAGS_normalsFolder;
	FLAGS_descriptorsFolder = FLAGS_dataPath + FLAGS_descriptorsFolder;

	std::vector<std::string> csvFileNames;
	
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (FLAGS_inFolder.data())) != NULL) {
	  /* Add all the files and directories to a vector */
		while ((ent = readdir (dir)) != NULL) {
			std::string fileName = ent->d_name;
			if(fileName != ".." && fileName != "."){
				csvFileNames.push_back(fileName);
			}
		}
		closedir (dir);
	}  else {
	  /* could not open directory */
		perror ("");
		return EXIT_FAILURE;
	} 

	sort(csvFileNames.begin(), csvFileNames.end(), 
		[](const std::string & a, const std::string & b) {
				int numA = std::stoi(a.substr(a.find(".") - 3, 3));
				int numB = std::stoi(b.substr(b.find(".") - 3, 3));
				return numA < numB;
		}
	);

	for (int i = 0; i < csvFileNames.size(); ++i) {
		const std::string csvFileName = FLAGS_inFolder + csvFileNames[i];
		const std::string binaryFileName = FLAGS_outFolder + 
		csvFileNames[i].substr(0,csvFileNames[i].find(".")) + ".dat";
		const std::string outFileName = csvFileNames[i].substr(0,csvFileNames[i].find(".")) + ".dat";

		std::vector<scan::PointXYZRGB> pointCloud;
	
		convertToBinary(csvFileName, binaryFileName, pointCloud);

		pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType>);
		createPCLPointCloud(pointCloud, cloud);

		std::vector<int> index;
		pcl::PointCloud<NormalType>::Ptr cloud_normals (new pcl::PointCloud<NormalType>);
		getNormals(cloud, cloud_normals, index);

		const std::string normalsName = FLAGS_normalsFolder + outFileName;
		if (FLAGS_save)
			saveNormals(cloud_normals, normalsName);

		pcl::PointCloud<DescriptorType>::Ptr cloud_descriptors (new pcl::PointCloud<DescriptorType>);
		getDescriptors(cloud, index, cloud_normals, cloud_descriptors);

		const std::string descriptorsName = FLAGS_descriptorsFolder + outFileName;
		if (FLAGS_save)
			saveDescriptors(cloud, cloud_descriptors, descriptorsName);

		/* const std::string buildName = csvFileNames[i].substr(0, 3);
		const std::string scanNumber = csvFileNames[i].substr(
			csvFileNames[i].find(".") - 3, 3);
		const std::string outName = buildName + "_panorama_" 
			+ scanNumber + ".png";
		if(FLAGS_redo)
			createPanorama(pointCloud, outName); */
	}

	return 0;
}

void convertToBinary(const std::string & fileNameIn,
	const std::string & outName,
	std::vector<scan::PointXYZRGB> & pointCloud) {
	
	std::cout << fileNameIn << std::endl;

	std::ifstream in (outName, std::ios::in | std::ios::binary);
	
	if (FLAGS_redo || !in.is_open()) {
		std::ifstream scanFile (fileNameIn, ios::in);
		int columns, rows;
		scanFile >> columns >> rows;
		std::string line;

		for (int i = 0; i < 9; ++i) {
			getline(scanFile, line);
		}
		pointCloud.reserve(columns*rows);
		for (int k = 0; k < columns * rows; ++k) {
			scan::PointXYZRGB in;
			scanFile >> in.point[0] >> in.point[1] >> in.point[2];

			double itmp;
			scanFile >> itmp;
			char rgb [3];
			for (int j = 0; j < 3; ++j)
				scanFile >> in.rgb[j];

			if (itmp < 0.2) continue;

			pointCloud.push_back(in);
		}
		int numPoints = pointCloud.size();
		int mult = 1;
		std::ofstream out (outName, std::ios::out | std::ios::binary);
		out.write(reinterpret_cast<const char *>(&numPoints), sizeof(numPoints));
		out.write(reinterpret_cast<const char *>(&mult), sizeof(mult));
		for (auto & p : pointCloud)
			out.write(reinterpret_cast<const char *>(&p), sizeof(scan::PointXYZRGB));
		
	} else {
		int numPoints, mult;
		in.read(reinterpret_cast<char *>(&numPoints), sizeof(numPoints));
		in.read(reinterpret_cast<char *>(&mult), sizeof(mult));

		pointCloud.resize(numPoints*mult);
		for (auto & p : pointCloud)
			in.read(reinterpret_cast<char *>(&p), sizeof(scan::PointXYZRGB));
	}
}


const int panoResolution = 500;
const double PI = 3.14159265358979323846;

static Eigen::Vector3f cartesianToPolar(const Eigen::Vector3f & coords) {
	float r = coords.norm();
	float theta = atan2(coords[1], coords[0]);
	float phi = acos(coords[2]/r);
	return Eigen::Vector3f (r, theta, phi);
}

static Eigen::Vector3f polarToCartesian(const Eigen::Vector3f & coords) {
	float x = coords[0]*cos(coords[1])*sin(coords[2]);
	float y = coords[0]*sin(coords[1])*sin(coords[2]);
	float z = coords[0]*cos(coords[2]);
	return Eigen::Vector3f (x,y,z);
}

const double phinot = PI/2;
const double thetanot = 0;

static Eigen::Vector3d pointCloudToPanorama(const Eigen::Vector3f & worldCoord) {
	auto polar = cartesianToPolar(worldCoord);
	/*int x = 2.0*sin(polar[1] - thetanot)/
		(1.0 + sin(phinot)*tan(polar[2]) + cos(phinot)*cos(polar[1] - thetanot)) * panoResolution;
	int y = 2*tan(phinot)*(cos(phinot) - sin(phinot)/tan(polar[2])*cos(polar[1] - thetanot))/
		(1 + sin(phinot)*tan(polar[2]) + cos(phinot)*cos(polar[1] - thetanot)) * panoResolution;*/
	int x = (polar[1]/PI + 1.0)*panoResolution;
	int y = polar[2]*panoResolution/PI;
	return Eigen::Vector3d(polar[0], x, y);
}

void createPanorama(const std::vector<scan::PointXYZRGB> & pointCloud,
	const std::string & outName) {
	cv::Mat panorama (panoResolution, 2*panoResolution, CV_8UC3, cv::Scalar::all(0));
	cv::Mat_<cv::Vec3b> _panorama = panorama;

	Eigen::MatrixXd zMap = Eigen::MatrixXd::Zero (panoResolution, 2*panoResolution);

	for(auto & element : pointCloud) {
		auto panoCoord = pointCloudToPanorama(element.point);
		int row = panoCoord[2];
		int col = panoCoord[1];
		double z = panoCoord[0];
		if (row < 0 || row >= panorama.rows)
			continue;
		if (col < 0 || col >= panorama.cols)
			continue;
		if(_panorama(row, col)[0] == _panorama(row, col)[1] 
			== _panorama(row, col)[2] == 0) {
			_panorama(row, col)[0] = element.rgb[2];
			_panorama(row, col)[1] = element.rgb[1];
			_panorama(row, col)[2] = element.rgb[0];
		} else {
			_panorama(row, col)[0] = (_panorama(row, col)[0] + element.rgb[2])/2.0;
			_panorama(row, col)[1] = (_panorama(row, col)[1] + element.rgb[1])/2.0;
			_panorama(row, col)[2] = (_panorama(row, col)[2] + element.rgb[0])/2.0;
		}

		if (zMap(row,col))
			zMap(row, col) = z;
		else
			zMap(row, col) = (zMap(row, col) + z)/2.0;
	}
	const std::string imgName = FLAGS_panoFolder + "images/" + outName;
	cv::imwrite(imgName, panorama);

	cvNamedWindow("Preview", CV_WINDOW_NORMAL);
	cv::imshow("Preview", panorama);
	cv::waitKey(0);
}

void createPCLPointCloud(const std::vector<scan::PointXYZRGB> & points, 
	pcl::PointCloud<PointType>::Ptr & cloud) {

	Eigen::Vector3f pointMin, pointMax;
	boundingBox(points, pointMin, pointMax);
	int subSample = 0;
	int subSampleSize = 15;
	for (auto & p : points) {
			bool in = true;
			for (int i = 0; i < 3; ++i)
				if (p.point[i] < pointMin[i] || p.point[i] > pointMax[i])
					in = false;
			
			if (!in) continue;

			if((subSample++)%subSampleSize) continue;

			auto & point = p.point;
			auto & rgb = p.rgb;
			PointType tmp;
			tmp.x = point[0];
			tmp.y = point[1];
			tmp.z = point[2];
			tmp.r = static_cast<uint8_t>(rgb[0]);
			tmp.g = static_cast<uint8_t>(rgb[1]);
			tmp.b = static_cast<uint8_t>(rgb[2]);
			cloud->push_back(tmp);
	}
}

void getNormals(const pcl::PointCloud<PointType>::Ptr & cloud,
	pcl::PointCloud<NormalType>::Ptr & cloud_normals,
	std::vector<int> & index) {

	pcl::NormalEstimationOMP<PointType, NormalType> ne;
ne.setInputCloud (cloud);

	pcl::search::KdTree<PointType>::Ptr tree 
		(new pcl::search::KdTree<PointType> ());
	ne.setSearchMethod (tree);

	// Use all neighbors in a sphere of radius 3cm
	ne.setRadiusSearch (0.03);

	// Compute the features
	ne.compute (*cloud_normals);
	
	pcl::removeNaNNormalsFromPointCloud(*cloud_normals, *cloud_normals, index);
}

void saveNormals(const pcl::PointCloud<NormalType>::Ptr & cloud_normals,
	const std::string & outName) {
	std::ofstream out (outName, std::ios::out | std::ios::binary);

	size_t size = cloud_normals->points.size();
	out.write(reinterpret_cast<const char *> 
		(& size), sizeof(size_t));


	for (int i = 0; i < cloud_normals->points.size(); ++i) {
		Eigen::Vector3f normals;
		normals[0] = cloud_normals->points[i].normal_x;
		normals[1] = cloud_normals->points[i].normal_y;
		normals[2] = cloud_normals->points[i].normal_z;
		
		out.write(reinterpret_cast<const char *> (normals.data()),
			sizeof(normals));
	}

	out.close();
}

void getDescriptors(const pcl::PointCloud<PointType>::Ptr & cloud,
	const std::vector<int> & index,
	const pcl::PointCloud<NormalType>::Ptr & cloud_normals,
	pcl::PointCloud<DescriptorType>::Ptr & cloud_descriptors) {

	pcl::PointCloud<PointType>::Ptr filtered_cloud (new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr NaN_removed_cloud (new pcl::PointCloud<PointType>);
	for (auto & i : index)
		NaN_removed_cloud->push_back(cloud->points[i]);

	pcl::PointCloud<int> sampled_indices; 
	float subSampleSize = 0.01f;
	pcl::UniformSampling<PointType> uniform_sampling;
	uniform_sampling.setInputCloud(NaN_removed_cloud);
  uniform_sampling.setRadiusSearch(subSampleSize);
  uniform_sampling.compute(sampled_indices);
  pcl::copyPointCloud(*NaN_removed_cloud, 
  	sampled_indices.points, *filtered_cloud); 

	pcl::SHOTColorEstimationOMP<PointType, NormalType, DescriptorType> shot_est;
	shot_est.setRadiusSearch (descr_rad_);

	shot_est.setInputCloud (filtered_cloud);
	shot_est.setInputNormals (cloud_normals);
	shot_est.setSearchSurface (cloud);
	shot_est.compute (*cloud_descriptors);
}

void saveDescriptors(const pcl::PointCloud<PointType>::Ptr & cloud,
	const pcl::PointCloud<DescriptorType>::Ptr & cloud_descriptors,
	const std::string & outName) {

	std::ofstream out (outName, std::ios::out | std::ios::binary);
	size_t num = cloud_descriptors->size();
	out.write(reinterpret_cast<const char *>(&num), sizeof(num));
	for (int i = 0; i < num; ++i) {
		SHOT1344WithXYZ out;
		out.position = Eigen::Vector3d 
			(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
		
		for (int i = 0; i < out.descriptor.size(); ++i) {
			out.descriptor[i] = cloud_descriptors->points[i].descriptor[i];
		}
	}
}

void boundingBox(const std::vector<scan::PointXYZRGB> & points,
	Eigen::Vector3f & pointMin, Eigen::Vector3f & pointMax) {
	Eigen::Vector3f average = Eigen::Vector3f::Zero();
	Eigen::Vector3f sigma = Eigen::Vector3f::Zero();
	for (auto & point : points) {
		average += point.point;
	}
	average /= points.size();

	for (auto & point : points)
		for (int i = 0; i < 3; ++i)
			sigma[i] += (point.point[i] - average[i])*(point.point[i] - average[i]);
		
	
	sigma /= points.size() - 1;
	for (int i = 0; i < 3; ++i)
		sigma[i] = sqrt(sigma[i]);

	Eigen::Vector3f range (10, 10, 6);
	Eigen::Vector3f delta;
	for (int i = 0; i < delta.size(); ++i)
	  delta[i] = 1.1*range[i]*sigma[i];

	pointMin = average - delta/2.0;
	pointMax = average + delta/2.0;
}

void SIFT(const std::vector<scan::PointXYZRGB> & points) {
	Eigen::Vector3f pointMin, pointMax;
	boundingBox(points, pointMin, pointMax);
	int subSample = 0;
	pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType>);
	for (auto & p : points) {
			bool in = true;
			for (int i = 0; i < 3; ++i)
				if (p.point[i] < pointMin[i] || p.point[i] > pointMax[i])
					in = false;
			
			if (!in) continue;

			if ((subSample++)%20) continue;

			auto & point = p.point;
			auto & rgb = p.rgb;
			PointType tmp;
			tmp.x = point[0];
			tmp.y = point[1];
			tmp.z = point[2];
			tmp.r = static_cast<uint8_t>(rgb[0]);
			tmp.g = static_cast<uint8_t>(rgb[1]);
			tmp.b = static_cast<uint8_t>(rgb[2]);
			cloud->push_back(tmp);
	}
	std::cout << cloud->size() << std::endl;

	// Parameters for sift computation
	const float min_scale = 0.1f;
	const int n_octaves = 6;
	const int n_scales_per_octave = 10;
	const float min_contrast = 0.5f;

	pcl::SIFTKeypoint<PointType, pcl::PointWithScale> sift;
	pcl::PointCloud<pcl::PointWithScale> result;
	pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType> ());
	sift.setSearchMethod(tree);
	sift.setScales(min_scale, n_octaves, n_scales_per_octave);
	sift.setMinimumContrast(min_contrast);
  sift.setInputCloud(cloud);
  sift.compute(result);


  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp (new pcl::PointCloud<pcl::PointXYZ>);
  copyPointCloud(result, *cloud_temp);

  if (true) {
  	// Visualization of keypoints along with the original cloud
  	  pcl::visualization::PCLVisualizer viewer("PCL Viewer");
  	  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints_color_handler (cloud_temp, 0, 255, 0);
  	  pcl::visualization::PointCloudColorHandlerCustom<PointType> cloud_color_handler (cloud, 255, 255, 0);
  	  viewer.setBackgroundColor( 0.0, 0.0, 0.0 );
  	  viewer.addPointCloud(cloud, "cloud");
  	  viewer.addPointCloud(cloud_temp, keypoints_color_handler, "keypoints");
  	  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "keypoints");
  	  
  	  while (!viewer.wasStopped ()) {
  	  	viewer.spinOnce ();
  	  }
  }
  
}