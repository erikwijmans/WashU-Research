#include <scan_typedefs.hpp>
#include <scan_gflags.h>


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
#include <pcl/filters/statistical_outlier_removal.h>


DEFINE_bool(ptx, false, "tells the preprocessor to read a ptx file and convert it to binary.  Otherwise, a binary file will be read");

typedef pcl::PointXYZRGB PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT1344 DescriptorType;

float search_rad (0.05f);

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
	std::vector<int> & indices);
void saveNormals(const pcl::PointCloud<NormalType>::Ptr & cloud_normals,
	const std::string & outName);
void getDescriptors(const pcl::PointCloud<PointType>::Ptr & cloud,
	std::vector<int> & indices,
	const pcl::PointCloud<NormalType>::Ptr & cloud_normals,
	pcl::PointCloud<DescriptorType>::Ptr & cloud_descriptors,
	pcl::PointCloud<PointType>::Ptr & filtered_cloud);
void saveDescriptors(const pcl::PointCloud<PointType>::Ptr & cloud,
	const pcl::PointCloud<DescriptorType>::Ptr & cloud_descriptors,
	const std::string & outName);

int main(int argc, char *argv[]) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	FLAGS_binaryFolder = FLAGS_dataPath + FLAGS_binaryFolder;
	FLAGS_PTXFolder = FLAGS_dataPath + FLAGS_PTXFolder;
	FLAGS_panoFolder = FLAGS_dataPath + FLAGS_panoFolder;
	FLAGS_normalsFolder = FLAGS_dataPath + FLAGS_normalsFolder;
	FLAGS_descriptorsFolder = FLAGS_dataPath + FLAGS_descriptorsFolder;

	std::vector<std::string> csvFileNames;

	std::string inFolder;
	if (FLAGS_ptx)
		inFolder = FLAGS_PTXFolder;
	else
		inFolder = FLAGS_binaryFolder;
	

	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (inFolder.data())) != NULL) {
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

	if (FLAGS_numScans == -1 )
    FLAGS_numScans = csvFileNames.size() - FLAGS_startIndex;

	for (int i = FLAGS_startIndex; i < FLAGS_numScans + FLAGS_startIndex; ++i) {
		const std::string csvFileName = FLAGS_PTXFolder + csvFileNames[i];
		const std::string binaryFileName = FLAGS_binaryFolder + 
		csvFileNames[i].substr(0,csvFileNames[i].find(".")) + ".dat";
		const std::string outFileName = csvFileNames[i].substr(0,csvFileNames[i].find(".")) + ".dat";

		std::vector<scan::PointXYZRGB> pointCloud;
	
		convertToBinary(csvFileName, binaryFileName, pointCloud);
		
		pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType>);
		createPCLPointCloud(pointCloud, cloud);

		std::cout << "Calculating Normls" << std::endl;
		std::vector<int> indices;
		pcl::PointCloud<NormalType>::Ptr cloud_normals (new pcl::PointCloud<NormalType>);
		getNormals(cloud, cloud_normals, indices);

		const std::string normalsName = FLAGS_normalsFolder + outFileName;
		
		saveNormals(cloud_normals, normalsName);

		std::cout << "Getting descriptors" << std::endl;
		pcl::PointCloud<DescriptorType>::Ptr cloud_descriptors (new pcl::PointCloud<DescriptorType>);
		pcl::PointCloud<PointType>::Ptr filtered_cloud (new pcl::PointCloud<PointType>);
		getDescriptors(cloud, indices, cloud_normals, cloud_descriptors, filtered_cloud);

		const std::string descriptorsName = FLAGS_descriptorsFolder + outFileName;
		
		saveDescriptors(filtered_cloud, cloud_descriptors, descriptorsName);
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
			
			double itmp [4];
			scanFile >> itmp[0];

			for (int j = 0; j < 3; ++j) {
				scanFile >> itmp[1+j];
				in.rgb[j] = itmp[1+j];
			}
			
			if (itmp[0] < 0.2) continue;
			if (!(in.point[0] || in.point[1] || in.point[2]))
				continue;

			pointCloud.push_back(in);
		}
		int numPoints = pointCloud.size();
		int mult = 1;
		std::ofstream out (outName, std::ios::out | std::ios::binary);
		out.write(reinterpret_cast<const char *>(&numPoints), sizeof(numPoints));
		out.write(reinterpret_cast<const char *>(&mult), sizeof(mult));
		for (auto & p : pointCloud) {
			p.writeToFile(out);
		}

		out.close();
		
	} else {
		
		int numPoints, mult;
		in.read(reinterpret_cast<char *>(&numPoints), sizeof(numPoints));
		in.read(reinterpret_cast<char *>(&mult), sizeof(mult));

		pointCloud.resize(numPoints*mult);
		for (auto & p : pointCloud)
			p.loadFromFile(in);
		
		in.close();
	}
}

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

	pcl::PointCloud<PointType>::Ptr local (new pcl::PointCloud<PointType>);
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
		local->push_back(tmp);
	}
	std::cout << "Removing outliers" << std::endl;
	pcl::StatisticalOutlierRemoval<PointType> outlierRemover;

	outlierRemover.setInputCloud(local);
	outlierRemover.setMeanK (50);
  outlierRemover.setStddevMulThresh (1.0);
	outlierRemover.filter(*cloud);

	std::cout << local->size() - cloud->size() << " outliers removed" << std::endl;
}

void getNormals(const pcl::PointCloud<PointType>::Ptr & cloud,
	pcl::PointCloud<NormalType>::Ptr & cloud_normals,
	std::vector<int> & indices) {

	pcl::NormalEstimationOMP<PointType, NormalType> ne;
	ne.setInputCloud (cloud);

	pcl::search::KdTree<PointType>::Ptr tree 
		(new pcl::search::KdTree<PointType> ());
	ne.setSearchMethod (tree);

	// Use all neighbors in a sphere of radius 3cm
	ne.setRadiusSearch(search_rad);

	// ne.setKSearch(20);

	// Compute the features
	ne.compute (*cloud_normals);
	size_t startSize = cloud_normals->size();
	pcl::removeNaNNormalsFromPointCloud(*cloud_normals, *cloud_normals, indices);
	std::cout << startSize - cloud_normals->size() << " NaN removed" << std::endl;
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
	std::vector<int> & indices,
	const pcl::PointCloud<NormalType>::Ptr & cloud_normals,
	pcl::PointCloud<DescriptorType>::Ptr & cloud_descriptors,
	pcl::PointCloud<PointType>::Ptr & filtered_cloud) {

	pcl::PointCloud<PointType>::Ptr NaN_removed_cloud (new pcl::PointCloud<PointType>);
	
	pcl::copyPointCloud(*cloud, indices, *NaN_removed_cloud);

	pcl::PointCloud<int> sampled_indices; 
	pcl::UniformSampling<PointType> uniform_sampling;

	uniform_sampling.setInputCloud(NaN_removed_cloud);
  uniform_sampling.setRadiusSearch(0.03f);
  uniform_sampling.compute(sampled_indices);

  pcl::copyPointCloud(*NaN_removed_cloud, 
  	sampled_indices.points, *filtered_cloud); 

  std::cout << "Down sampled to: " << filtered_cloud->size() << std::endl;

	pcl::SHOTColorEstimationOMP<PointType, NormalType, DescriptorType> shot_est;

	shot_est.setRadiusSearch (search_rad);
	shot_est.setInputCloud (filtered_cloud);
	shot_est.setInputNormals (cloud_normals);
	shot_est.setSearchSurface (NaN_removed_cloud);
	shot_est.compute (*cloud_descriptors);

	std::cout << cloud_descriptors->size() << " descriptors found" << std::endl;
}

static double norm1344 (float * vec) {
	double norm = 0;
	#pragma omp simd
	for (int i = 0; i < 1344; ++i) {
		norm += vec[i]*vec[i];
	}
	return sqrt(norm);
}

void saveDescriptors(const pcl::PointCloud<PointType>::Ptr & cloud,
	const pcl::PointCloud<DescriptorType>::Ptr & cloud_descriptors,
	const std::string & outName) {

	std::vector<int> finiteIndicies;
	for (int i = 0; i < cloud_descriptors->size(); ++i)
		if (pcl_isfinite (cloud_descriptors->at(i).descriptor[0]) && 
			norm1344(cloud_descriptors->at(i).descriptor))
			finiteIndicies.push_back(i);
	
	std::cout << finiteIndicies.size() << " well formed descriptors found" << std::endl;

	std::ofstream out (outName, std::ios::out | std::ios::binary);
	int num = finiteIndicies.size();
	out.write(reinterpret_cast<const char *>(&num), sizeof(num));
	for (int i = 0; i < num; ++i) {
		int index = finiteIndicies[i];
		SHOT1344WithXYZ SHOTout;
		SHOTout.position = Eigen::Vector3d 
			(cloud->at(index).x, cloud->at(index).y, cloud->at(index).z);
		for (int k = 0; k < SHOTout.descriptor->size(); ++k) {
			(*SHOTout.descriptor)[k] = cloud_descriptors->at(index).descriptor[k];
		}
		SHOTout.writeToFile(out);
	}
	out.close();
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