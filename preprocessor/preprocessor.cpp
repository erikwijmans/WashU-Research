#include "getRotations.h"
#include "preprocessor.h"

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/StdVector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>

#include "opencv2/xfeatures2d.hpp"

#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/keypoints/impl/sift_keypoint.hpp>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <dirent.h>

DEFINE_bool(ptx, false, "Tells the preprocessor to read from ptx file and convert it to binary.  Otherwise, a binary file will be read");

int PTXrows, PTXcols;

void convertToBinary(const std::string & fileNameIn,
	const std::string &,
	std::vector<scan::PointXYZRGBA> & pointCloud);
void createPanorama(const std::vector<scan::PointXYZRGBA> & pointCloud,
	const std::string & panoName,
	const std::string & dataName);
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

	std::string * inFolder;
	if (FLAGS_ptx)
		inFolder = &FLAGS_PTXFolder;
	else
		inFolder = &FLAGS_binaryFolder;

	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (inFolder->data())) != NULL) {
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
		const std::string number = csvFileNames[i].substr(csvFileNames[i].find(".") - 3, 3);
		const std::string buildName = csvFileNames[i].substr(csvFileNames[i].rfind("/") + 1, 3);
		const std::string csvFileName = FLAGS_PTXFolder + csvFileNames[i];
		const std::string binaryFileName = FLAGS_binaryFolder +
			buildName + "_binary_" + number + ".dat";

		std::vector<scan::PointXYZRGBA> pointCloud;
		convertToBinary(csvFileName, binaryFileName, pointCloud);

		const std::string panoName = FLAGS_panoFolder + "images/"
			+ buildName + "_panorama_" + number + ".png";
			const std::string dataName = FLAGS_panoFolder + "data/"
			+ buildName + "_data_" + number + ".dat";
		createPanorama(pointCloud, panoName, dataName);
		continue;

		pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType>);
		createPCLPointCloud(pointCloud, cloud);

		std::cout << "Calculating Normals" << std::endl;
		pcl::PointCloud<NormalType>::Ptr cloud_normals (new pcl::PointCloud<NormalType>);
		pcl::PointCloud<PointType>::Ptr normals_points (new pcl::PointCloud<PointType>);
		getNormals(cloud, cloud_normals, normals_points);

		const std::string normalsName = FLAGS_normalsFolder +
			buildName + "_normals_" + number + ".dat";
		saveNormals(cloud_normals, normalsName);

		const std::string rotName = FLAGS_rotFolder +
			buildName + "_rotations_" + number + ".dat";
		getRotations(cloud_normals, rotName);

		std::cout << "Getting descriptors" << std::endl;
		pcl::PointCloud<DescriptorType>::Ptr cloud_descriptors (new pcl::PointCloud<DescriptorType>);
		pcl::PointCloud<PointType>::Ptr filtered_cloud (new pcl::PointCloud<PointType>);
		getDescriptors(cloud, cloud_normals, normals_points, cloud_descriptors, filtered_cloud);

		const std::string descriptorsName = FLAGS_descriptorsFolder +
			buildName + "_descriptors_" + number + ".dat";

		saveDescriptors(filtered_cloud, cloud_descriptors, descriptorsName);
	}
	return 0;
}

void convertToBinary(const std::string & fileNameIn,
	const std::string & outName,
	std::vector<scan::PointXYZRGBA> & pointCloud) {

	std::cout << outName << std::endl;

	std::ifstream in (outName, std::ios::in | std::ios::binary);

	if (FLAGS_ptx || !in.is_open()) {
		std::ifstream scanFile (fileNameIn, std::ios::in);
		int columns, rows;
		scanFile >> columns >> rows;
		std::ofstream out (outName, std::ios::out | std::ios::binary);
		out.write(reinterpret_cast<const char *>(&columns), sizeof(columns));
		out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
		PTXcols = columns;
		PTXrows = rows;
		std::string line;
		pointCloud.reserve(columns*rows);
		for (int i = 0; i < 9; ++i) {
			getline(scanFile, line);
		}

		for (int k = 0; k < columns * rows; ++k) {
			scan::PointXYZRGBA tmp;
			scanFile >> tmp.point[0] >> tmp.point[1] >> tmp.point[2];

			scanFile >> tmp.intensity;

			double itmp [3];
			for (int j = 0; j < 3; ++j) {
				scanFile >> itmp[j];
				tmp.rgb[j] = cv::saturate_cast<uchar>(itmp[j]);
			}

			tmp.writeToFile(out);
			pointCloud.push_back(tmp);
		}

		out.close();

	} else {
		int columns, rows;
		in.read(reinterpret_cast<char *>(&columns), sizeof(columns));
		in.read(reinterpret_cast<char *>(&rows), sizeof(rows));

		PTXcols = columns;
		PTXrows = rows;

		pointCloud.resize(columns*rows);
		for (auto & p : pointCloud)
			p.loadFromFile(in);

		in.close();
	}
}

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

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


static Eigen::Vector3d pointCloudToPanorama(const Eigen::Vector3f & worldCoord,
	const cv::Size & resolution) {
	auto polar = cartesianToPolar(worldCoord.cast<double>());
	int x = (polar[1]/PI + 1.0)*(resolution.width - 1)/2.0;
	int y = polar[2]*(resolution.height - 1)/maxPhi;
	return Eigen::Vector3d(polar[0], x, y);
}

void createPanorama(const std::vector<scan::PointXYZRGBA> & pointCloud,
	const std::string & panoName,
	const std::string & dataName) {

	cv::Mat trackingPanorama (PTXrows, PTXcols, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat_<cv::Vec3b> _trackingPanorama = trackingPanorama;

	cv::Mat PTXPanorama(PTXrows, PTXcols, CV_8UC3);
	cv::Mat_<cv::Vec3b>	_PTXPanorama = PTXPanorama;
	Eigen::RowMatrixXd rMap = Eigen::RowMatrixXd::Zero(PTXrows, PTXcols);

	int row = PTXrows - 1;
	int col = PTXcols - 1;
	Eigen::RowMatrixXb touched = Eigen::RowMatrixXb::Zero(PTXrows, PTXcols);

	for(auto element : pointCloud) {
		if (row >= 0 && row < PTXPanorama.rows
			&& col >= 0 && col < PTXPanorama.cols) {
			_PTXPanorama(row, col)[0] = element.rgb[2];
			_PTXPanorama(row, col)[1] = element.rgb[1];
			_PTXPanorama(row, col)[2] = element.rgb[0];

		} else
		 std::cout << "Uhoh" << std::endl;

		if (row == 0) {
			row = PTXrows - 1;
			col = col == 0 ? PTXcols - 1 : col - 1;
		} else
			--row;

		auto panoCoord = pointCloudToPanorama(element.point, trackingPanorama.size());
		int trackedRow = panoCoord[2];
		int trackedCol = panoCoord[1];
		double r = panoCoord[0];
		if (trackedRow < 0 || trackedRow >= trackingPanorama.rows)
			continue;
		if (trackedCol < 0 || trackedCol >= trackingPanorama.cols)
			continue;
		if (!(element.rgb[2] || element.rgb[1] || element.rgb[0]))
			continue;

		_trackingPanorama(trackedRow, trackedCol)[0] = element.rgb[2];
		_trackingPanorama(trackedRow, trackedCol)[1] = element.rgb[1];
		_trackingPanorama(trackedRow, trackedCol)[2] = element.rgb[0];

		touched(trackedRow, trackedCol) = 1;
		rMap(trackedRow, trackedCol) = r;
	}

	double currentR = 0;
	double * dataPtr = rMap.data();
	for (int i = 0; i < rMap.size(); ++i) {
		if (*(dataPtr + i)) {
			currentR = *(dataPtr + i);
			break;
		}
	}
	for (int i = 0; i < rMap.size(); ++i) {
		if (*(dataPtr + i))
			currentR = *(dataPtr + i);
		else
			*(dataPtr + i) = currentR;
	}

	const int offset = 5156 + 20;
	for (int j = 0; j < touched.rows(); ++j) {
		const int panoRow = j;
		uchar * dst = trackingPanorama.ptr<uchar>(panoRow);
		const uchar * src = PTXPanorama.ptr<uchar>(panoRow);
		for (int i = 0; i < touched.cols(); ++i) {
			if (!touched(j, i)) {
				const int panoCol = i;
				const int tCol = 3*panoCol;
				const int pCol = 3*((panoCol + offset) % PTXPanorama.cols);
				dst[tCol + 0] = src[pCol + 0];
				dst[tCol + 1] = src[pCol + 1];
				dst[tCol + 2] = src[pCol + 2];
			}
		}
	}

	constexpr double scale = pow(2, -1);
	cv::Mat scaledTracking, scaledPTX;
	cv::resize(trackingPanorama, scaledTracking, cv::Size(), scale, scale, CV_INTER_AREA);
	cv::resize(PTXPanorama, scaledPTX, cv::Size(), scale, scale, CV_INTER_AREA);

	Eigen::RowMatrixXd scaledRMap ((int)(PTXrows*scale), (int)(PTXcols*scale));
	for (int j = 0; j < scaledRMap.rows(); ++j) {
		for (int i = 0; i < scaledRMap.cols(); ++i) {
			int row = floor(j/scale);
			int col = floor(i/scale);
			double average = 0;
			for (int y = 0; y < 1.0/scale; ++y) {
				for (int x = 0; x < 1.0/scale; ++x) {
					average += rMap(row + y, col + x);
				}
			}
			average *= scale*scale;
			scaledRMap(j, i) = average;
		}
	}
	std::vector<cv::KeyPoint> keypoints;
	cv::Ptr<cv::Feature2D> SIFT = cv::xfeatures2d::SIFT::create();

	place::Panorama pano;
	pano.img = scaledTracking;
	pano.rMap = scaledRMap.cast<float>();

	SIFT->detect(pano.img, keypoints);
	SIFT->compute(pano.img, keypoints, pano.descriptors);
	std::cout << keypoints.size() << std::endl;

	for (auto & kp : keypoints)
		pano.keypoints.push_back(kp.pt);

	pano.writeToFile(panoName, dataName);

	if (FLAGS_preview) {
		cvNamedWindow("Tracking", CV_WINDOW_NORMAL);
		cv::imshow("Tracking", scaledTracking);
		cvNamedWindow("PTX", CV_WINDOW_NORMAL);
		cv::imshow("PTX", scaledPTX);
		cv::Mat out;
		cv::drawKeypoints(pano.img, keypoints, out);
		cvNamedWindow("KP", CV_WINDOW_NORMAL);
		cv::imshow("KP", out);
		std::cout << type2str(pano.descriptors.type()) << std::endl;
		cv::waitKey(0);
	}
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
		if (p.intensity < 0.2) continue;

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

	for (auto & n : *cloud_normals) {
		Eigen::Vector3f normals (n.normal_x, n.normal_y, n.normal_z);

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

	pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> shot_est;

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

static bool finite1344(float * vec) {
	for (int i = 0; i < 1344; ++i) {
		if (!pcl_isfinite(vec[i])) return false;
	}
	return true;
}

static double norm352 (float * vec) {
	double norm = 0;
	#pragma omp simd
	for (int i = 0; i < 352; ++i) {
		norm += vec[i]*vec[i];
	}
	return sqrt(norm);
}

static bool finite352(float * vec) {
	for (int i = 0; i < 352; ++i) {
		if (!pcl_isfinite(vec[i])) return false;
	}
	return true;
}

void saveDescriptors(const pcl::PointCloud<PointType>::Ptr & cloud,
	const pcl::PointCloud<DescriptorType>::Ptr & cloud_descriptors,
	const std::string & outName) {

	std::vector<int> wellFormedIndicies;
	for (int i = 0; i < cloud_descriptors->size(); ++i)
		if (finite352(cloud_descriptors->at(i).descriptor) &&
			norm352(cloud_descriptors->at(i).descriptor))
			wellFormedIndicies.push_back(i);

	std::cout << wellFormedIndicies.size() << " well formed descriptors found" << std::endl;

	std::ofstream out (outName, std::ios::out | std::ios::binary);
	int num = wellFormedIndicies.size();
	out.write(reinterpret_cast<const char *>(&num), sizeof(num));
	for (int i = 0; i < num; ++i) {
		int index = wellFormedIndicies[i];
		SPARSE352WithXYZ SHOTout;
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

  if (false) {
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