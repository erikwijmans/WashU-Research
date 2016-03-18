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

#include <omp.h>

DEFINE_bool(redo, false, "Redo all the binaryFiles and panoramas");
DEFINE_string(dataPath, "/home/erik/Projects/3DscanData/DUC/Floor1/", "Path to data");
DEFINE_string(outFolder, "binaryFiles/", 
	"Path to binary files");
DEFINE_string(inFolder, "PTXFiles/",
	"Path to Output");
DEFINE_string(panoFolder, "panoramas/", "Path to write panoramas");

typedef struct {
	Eigen::Vector3f point;
	Eigen::Vector3i rgb;
} PointXYZRGB;

using namespace std;
void csvToBinary(const string & fileNameIn, const string& fileNameOut,
	std::vector<PointXYZRGB> & pointCloud);
void createPanorama(const std::vector<PointXYZRGB> & pointCloud,
	const std::string & outName);

const double PI = 3.14159265358979323846;

int main(int argc, char *argv[]) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	FLAGS_outFolder = FLAGS_dataPath + FLAGS_outFolder;
	FLAGS_inFolder = FLAGS_dataPath + FLAGS_inFolder;
	FLAGS_panoFolder = FLAGS_dataPath + FLAGS_panoFolder;

	std::vector<string> csvFileNames;
	
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (FLAGS_inFolder.data())) != NULL) {
	  /* Add all the files and directories to a vector */
		while ((ent = readdir (dir)) != NULL) {
			string fileName = ent->d_name;
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
		const string csvFileName = FLAGS_inFolder + csvFileNames[i];
		const string binaryFileName = FLAGS_outFolder + 
		csvFileNames[i].substr(0,csvFileNames[i].find(".")) + ".dat";
		std::vector<PointXYZRGB> pointCloud;
		csvToBinary(csvFileName, binaryFileName, pointCloud);

		const std::string buildName = csvFileNames[i].substr(0, 3);
		const std::string scanNumber = csvFileNames[i].substr(
			csvFileNames[i].find(".") - 3, 3);
		const std::string outName = FLAGS_panoFolder + buildName + "_panorama_" 
			+ scanNumber + ".png";
		if(FLAGS_redo)
			createPanorama(pointCloud, outName);
	}


	return 0;
}

void csvToBinary(const string & fileNameIn, const string& fileNameOut,
	std::vector<PointXYZRGB> & pointCloud) {
	if(!FLAGS_redo) {
		ifstream out (fileNameOut, ios::in | ios::binary);
		if(out.is_open())
			return;
	}

	std::cout << fileNameIn << std::endl;

	ifstream scanFile (fileNameIn, ios::in);
	ofstream binaryFile (fileNameOut, ios::out | ios::binary);


	int columns, rows;
	scanFile >> columns >> rows;
	string line;

	binaryFile.write(reinterpret_cast<const char *> (& columns), sizeof(int));
	binaryFile.write(reinterpret_cast<const char *> (& rows), sizeof(int));


	for (int i = 0; i < 9; ++i) {
		getline(scanFile, line);
	}

	for (int k = 0; k < columns * rows; ++k) {
		Eigen::Vector3f point;
		scanFile >> point[0] >> point[1] >> point[2];

		double itmp[4];
		for (int j = 0; j < 4; ++j)
			scanFile >> itmp[j];

		if(itmp[0] < 0.2){
			float temp = 0;
			binaryFile.write(reinterpret_cast<const char *> (& temp), 
				sizeof(float));
			binaryFile.write(reinterpret_cast<const char *> (& temp), 
				sizeof(float));
			binaryFile.write(reinterpret_cast<const char *> (& temp), 
				sizeof(float));
		} else{
				binaryFile.write(reinterpret_cast<const char *> (point.data()), 
					sizeof(Eigen::Vector3f));

				Eigen::Vector3i rgb (itmp[1], itmp[2], itmp[3]);
				pointCloud.push_back({point, rgb});
		}
		
	}
	scanFile.close();
	binaryFile.close();
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

void createPanorama(const std::vector<PointXYZRGB> & pointCloud,
	const std::string & outName) {
	cv::Mat panorama (1000, 2000, CV_8UC3, cv::Scalar::all(0));
	cv::Mat_<cv::Vec3b> _panorama = panorama;
	for(auto & element : pointCloud) {
		auto polarCoords = cartesianToPolar(element.point);
		int row = polarCoords[2]/PI*panorama.rows;
		int col = panorama.cols - (polarCoords[1]/PI + 1.0)*panorama.cols/2.0;
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
	}
	cvNamedWindow("Preview", CV_WINDOW_NORMAL);
	cv::imshow("Preview", panorama);
	cv::waitKey(0);
	cv::imwrite(outName, panorama);
}

/*void reserveEngineer(const std::vector<PointXYZRGB> & pointCloud) {
	std::ofstream longitude ("longitude.txt");
	std::ofstream latitude ("latitude.txt");
	const int numThetaSteps = 1000;
	const int numPhiSteps = 500;
	for (float theta = 0; theta < 2*PI; theta += (2.0*PI)/numThetaSteps) {
		for (int phi = 0; phi < PI; ++phi) {
			Eigen::Vector3f panoCoord (1.0, theta, phi);
			auto cartCoord = polarToCartesian(panoCoord);
			cartCoord /= cartCoord.norm();
			for(auto & element : pointCloud) {
				auto point = element.point;
				point /= point.norm();
				if(point.dot(cartCoord) > 0.95) {
					auto polarCoords = cartesianToPolar(point);
					longitude << theta << polarCoords
				}
			}
		}
	}

}*/