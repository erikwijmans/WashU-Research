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

int PTXrows, PTXcols;

void convertToBinary(const std::string & fileNameIn,
	const std::string &,
	std::vector<scan::PointXYZRGBA> & pointCloud);
void createPanorama(const std::vector<scan::PointXYZRGBA> & pointCloud,
	const std::string & outName);
void boundingBox(const std::vector<scan::PointXYZRGBA> & points,
	Eigen::Vector3f & pointMin, Eigen::Vector3f & pointMax);
void createPCLPointCloud(const std::vector<scan::PointXYZRGBA> & points, 
	pcl::PointCloud<PointType>::Ptr & cloud);
void getNormals(const pcl::PointCloud<PointType>::Ptr & cloud,
	pcl::PointCloud<NormalType>::Ptr & cloud_normals,
	pcl::PointCloud<PointType>::Ptr & normals_points);
void saveNormals(const pcl::PointCloud<NormalType>::Ptr & cloud_normals,
	const std::string & outName);
void getDescriptors(const pcl::PointCloud<PointType>::Ptr & cloud,
	const pcl::PointCloud<NormalType>::Ptr & cloud_normals,
	const pcl::PointCloud<PointType>::Ptr & normals_points,
	pcl::PointCloud<DescriptorType>::Ptr & cloud_descriptors,
	pcl::PointCloud<PointType>::Ptr & filtered_cloud);
void saveDescriptors(const pcl::PointCloud<PointType>::Ptr & cloud,
	const pcl::PointCloud<DescriptorType>::Ptr & cloud_descriptors,
	const std::string & outName);
void SIFT(const pcl::PointCloud<PointType>::Ptr & cloud,
	const std::string & outName);

int main(int argc, char *argv[]) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	preappendDataPath();

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

		std::vector<scan::PointXYZRGBA> pointCloud;
	
		convertToBinary(csvFileName, binaryFileName, pointCloud);
		continue;
		
		pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType>);
		createPCLPointCloud(pointCloud, cloud);

		const std::string SIFTName = FLAGS_SIFTFolder + outFileName;
		SIFT(cloud, SIFTName);

		std::cout << "Calculating Normals" << std::endl;
		pcl::PointCloud<NormalType>::Ptr cloud_normals (new pcl::PointCloud<NormalType>);
		pcl::PointCloud<PointType>::Ptr normals_points (new pcl::PointCloud<PointType>);
		getNormals(cloud, cloud_normals, normals_points);

		const std::string normalsName = FLAGS_normalsFolder + outFileName;
		
		saveNormals(cloud_normals, normalsName);

		std::cout << "Getting descriptors" << std::endl;
		pcl::PointCloud<DescriptorType>::Ptr cloud_descriptors (new pcl::PointCloud<DescriptorType>);
		pcl::PointCloud<PointType>::Ptr filtered_cloud (new pcl::PointCloud<PointType>);
		getDescriptors(cloud, cloud_normals, normals_points, cloud_descriptors, filtered_cloud);

		const std::string descriptorsName = FLAGS_descriptorsFolder + outFileName;
		
		saveDescriptors(filtered_cloud, cloud_descriptors, descriptorsName);
	}
	return 0;
}

void convertToBinary(const std::string & fileNameIn,
	const std::string & outName,
	std::vector<scan::PointXYZRGBA> & pointCloud) {
	
	std::cout << outName << std::endl;

	std::ifstream in (outName, std::ios::in | std::ios::binary);
	
	if (FLAGS_redo || !in.is_open()) {
		
		std::ifstream scanFile (fileNameIn, ios::in);
		int columns, rows;
		scanFile >> columns >> rows;
		std::ofstream out (outName, std::ios::out | std::ios::binary);
		out.write(reinterpret_cast<const char *>(&columns), sizeof(columns));
		out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));

		std::string line;

		for (int i = 0; i < 9; ++i) {
			getline(scanFile, line);
		}
		pointCloud.reserve(columns*rows);
		for (int k = 0; k < columns * rows; ++k) {
			scan::PointXYZRGBA in;
			scanFile >> in.point[0] >> in.point[1] >> in.point[2];
			
			scanFile >> in.intensity;
			
			double itmp [3];
			for (int j = 0; j < 3; ++j) {
				scanFile >> itmp[j];
				in.rgb[j] = itmp[j];
			}
			
			in.writeToFile(out);
		}
		
		out.close();
		in.open(outName, std::ios::out | std::ios::binary);	
	}
	return;
	int columns, rows;
	in.read(reinterpret_cast<char *>(&columns), sizeof(columns));
	in.read(reinterpret_cast<char *>(&rows), sizeof(rows));

	pointCloud.resize(columns*rows);
	for (auto & p : pointCloud)
		p.loadFromFile(in);
		
	in.close();

	PTXcols = columns;
	PTXrows = rows;
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
	int x = (polar[1]/PI + 1.0)*panoResolution;
	int y = polar[2]*panoResolution/PI;
	return Eigen::Vector3d(polar[0], x, y);
}

void createPanorama(const std::vector<scan::PointXYZRGBA> & pointCloud,
	const std::string & outName) {

	cv::Mat trackingPanorama (panoResolution, 2*panoResolution, CV_8UC3, cv::Scalar::all(0)),
		PTXPanorama(panoResolution, 2*panoResolution, CV_8UC3);
	cv::Mat_<cv::Vec3b> _trackingPanorama = trackingPanorama,
		_PTXPanorama = PTXPanorama;

	int count = 0;
	for(auto & element : pointCloud) {
		int row = floor(count/PTXcols);
		int col = count % PTXcols;
		++count;
		_PTXPanorama(row, col)[0] = element.rgb[2];
		_PTXPanorama(row, col)[1] = element.rgb[1];
		_PTXPanorama(row, col)[2] = element.rgb[0];

		auto panoCoord = pointCloudToPanorama(element.point);
		row = panoCoord[2];
		col = panoCoord[1];
		double z = panoCoord[0];
		if (row < 0 || row >= trackingPanorama.rows)
			continue;
		if (col < 0 || col >= trackingPanorama.cols)
			continue;
		if(_trackingPanorama(row, col)[0] == _trackingPanorama(row, col)[1] 
			== _trackingPanorama(row, col)[2] == 0) {
			_trackingPanorama(row, col)[0] = element.rgb[2];
			_trackingPanorama(row, col)[1] = element.rgb[1];
			_trackingPanorama(row, col)[2] = element.rgb[0];
		} else {
			_trackingPanorama(row, col)[0] = (_trackingPanorama(row, col)[0] + element.rgb[2])/2.0;
			_trackingPanorama(row, col)[1] = (_trackingPanorama(row, col)[1] + element.rgb[1])/2.0;
			_trackingPanorama(row, col)[2] = (_trackingPanorama(row, col)[2] + element.rgb[0])/2.0;
		}
	}

	for (int j = 0; j < trackingPanorama.rows; ++j) {
		uchar * dst = trackingPanorama.ptr<uchar>(j);
		const uchar * src = PTXPanorama.ptr<uchar>(j);
		for (int i = 0; i < trackingPanorama.cols*trackingPanorama.channels(); i+=3) {
			if (dst[i + 0] == 0 && dst[i + 1] == 0 && dst[i + 2] == 0) {
				dst[i + 0] = src[i + 0];
				dst[i + 1] = src[i + 1];
				dst[i + 2] = src[i + 2];
			}
		}
	}

	const std::string imgName = FLAGS_panoFolder + "images/" + outName;
	cv::imwrite(imgName, trackingPanorama);

	cvNamedWindow("Tracking", CV_WINDOW_NORMAL);
	cv::imshow("Tracking", trackingPanorama);
	cvNamedWindow("PTX", CV_WINDOW_NORMAL);
	cv::imshow("PTX", PTXPanorama);
	cv::waitKey(0);
}

void createPCLPointCloud(const std::vector<scan::PointXYZRGBA> & points, 
	pcl::PointCloud<PointType>::Ptr & cloud) {

	Eigen::Vector3f pointMin, pointMax;
	boundingBox(points, pointMin, pointMax);

	for (auto & p : points) {
		bool in = true;
		for (int i = 0; i < 3; ++i)
			if (p.point[i] < pointMin[i] || p.point[i] > pointMax[i])
				in = false;
		
		if (!in) continue;

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
	pcl::PointCloud<PointType>::Ptr & normals_points) {

	pcl::PointCloud<int> sampled_indices;
	pcl::PointCloud<PointType>::Ptr filtered_cloud (new pcl::PointCloud<PointType>);
	pcl::UniformSampling<PointType> uniform_sampling;

	uniform_sampling.setInputCloud(cloud);
	// 85 mm
  uniform_sampling.setRadiusSearch(0.0085f);
  uniform_sampling.compute(sampled_indices);

  pcl::copyPointCloud(*cloud, 
  	sampled_indices.points, *filtered_cloud);

  std::cout << "Num normals to find: " << filtered_cloud->size() << std::endl;

	pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
	pcl::search::KdTree<PointType>::Ptr tree 
		(new pcl::search::KdTree<PointType> ());

	norm_est.setInputCloud (filtered_cloud);
	norm_est.setSearchSurface(cloud);
	norm_est.setSearchMethod (tree);
	norm_est.setRadiusSearch(0.03);
	norm_est.compute (*cloud_normals);

	size_t startSize = cloud_normals->size();
	std::vector<int> indices;
	pcl::removeNaNNormalsFromPointCloud(*cloud_normals, *cloud_normals, indices);
	pcl::copyPointCloud(*filtered_cloud, indices, *normals_points);

	std::cout << startSize -  cloud_normals->size() << " NaN removed" << std::endl;
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
	const pcl::PointCloud<NormalType>::Ptr & cloud_normals,
	const pcl::PointCloud<PointType>::Ptr & normals_points,
	pcl::PointCloud<DescriptorType>::Ptr & cloud_descriptors,
	pcl::PointCloud<PointType>::Ptr & filtered_cloud) {

	pcl::PointCloud<int> sampled_indices; 
	pcl::UniformSampling<PointType> uniform_sampling;

	uniform_sampling.setInputCloud(cloud);
  uniform_sampling.setRadiusSearch(0.05);
  uniform_sampling.compute(sampled_indices);

  pcl::copyPointCloud(*cloud, 
  	sampled_indices.points, *filtered_cloud); 

  std::cout << "Down sampled to: " << filtered_cloud->size() << std::endl;

	pcl::SHOTColorEstimationOMP<PointType, NormalType, DescriptorType> shot_est;

	shot_est.setRadiusSearch (0.03);
	shot_est.setInputCloud (filtered_cloud);
	shot_est.setInputNormals (cloud_normals);
	shot_est.setSearchSurface (normals_points);
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

static bool finite(float * vec) {
	for (int i = 0; i < 1344; ++i) {
		if (!pcl_isfinite(vec[i])) return false;
	}
	return true;
}

void saveDescriptors(const pcl::PointCloud<PointType>::Ptr & cloud,
	const pcl::PointCloud<DescriptorType>::Ptr & cloud_descriptors,
	const std::string & outName) {

	std::vector<int> wellFormedIndicies;
	for (int i = 0; i < cloud_descriptors->size(); ++i)
		if (finite(cloud_descriptors->at(i).descriptor) && 
			norm1344(cloud_descriptors->at(i).descriptor))
			wellFormedIndicies.push_back(i);
	
	std::cout << wellFormedIndicies.size() << " well formed descriptors found" << std::endl;

	std::ofstream out (outName, std::ios::out | std::ios::binary);
	int num = wellFormedIndicies.size();
	out.write(reinterpret_cast<const char *>(&num), sizeof(num));
	for (int i = 0; i < num; ++i) {
		int index = wellFormedIndicies[i];
		SPARSE1344WithXYZ SHOTout;
		SHOTout.position = Eigen::Vector3d 
			(cloud->at(index).x, cloud->at(index).y, cloud->at(index).z);
		for (int k = 0; k < SHOTout.descriptor->size(); ++k) {
			float value = cloud_descriptors->at(index).descriptor[k];
			if (value)
				SHOTout.descriptor->coeffRef(k) = value;
		}
		SHOTout.writeToFile(out);
	}
	out.close();
}

void boundingBox(const std::vector<scan::PointXYZRGBA> & points,
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

void saveSIFT(const pcl::PointCloud<pcl::PointXYZ>::Ptr & SIFT, 
	const std::string & outName) {
	std::ofstream out (outName, std::ios::out | std::ios::binary);
	int size = SIFT->size();
	out.write(reinterpret_cast<const char *>(&size), sizeof(size));

	for (auto & F : *SIFT) {
		Eigen::Vector3f tmp (F.x, F.y, F.z);
		out.write(reinterpret_cast<const char *>(tmp.data()), sizeof(tmp));
	}
	out.close();
}

void SIFT(const pcl::PointCloud<PointType>::Ptr & cloud, 
	const std::string & outName) {

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

  pcl::PointCloud<pcl::PointXYZ>::Ptr SIFT (new pcl::PointCloud<pcl::PointXYZ>);
  copyPointCloud(result, *SIFT);

  if (true) {
  	pcl::PointCloud<PointType>::Ptr sub_sampled (new pcl::PointCloud<PointType>);
  	int subSample = 0;
  	const int subSampleSize = 10;
  	for (auto & p : *cloud)
  		if (subSample++ % subSampleSize == 0)
  			sub_sampled->push_back(p);
  	// Visualization of keypoints along with the original cloud
  	  pcl::visualization::PCLVisualizer viewer("PCL Viewer");
  	  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints_color_handler (SIFT, 0, 255, 0);
  	  pcl::visualization::PointCloudColorHandlerCustom<PointType> cloud_color_handler (cloud, 255, 255, 0);
  	  viewer.setBackgroundColor( 0.0, 0.0, 0.0 );
  	  viewer.addPointCloud(sub_sampled, "cloud");
  	  viewer.addPointCloud(SIFT, keypoints_color_handler, "keypoints");
  	  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "keypoints");
  	  
  	  while (!viewer.wasStopped ()) {
  	  	viewer.spinOnce ();
  	  }
  }
  saveSIFT(SIFT, outName);
  exit(1);
}