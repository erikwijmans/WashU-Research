/*Scanner units are proabaly in meters */

#include "scanDensity_scanDensity.h"

#include <omp.h>
#include <locale> 


#include <sstream>
#include <map>


/*#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>*/

DEFINE_bool(pe, false, "Tells the program to only examine point evidence");
DEFINE_bool(fe, false, "Tells the program to only examine free space evidence");
DEFINE_bool(quiteMode, true, "Turns of all extrenous statements");
DEFINE_bool(preview, false, "Turns on previews of the output");
DEFINE_bool(redo, false, "Recreates the density map even if it already exists");
DEFINE_bool(3D, false, "writes out 3D voxelGrids");
DEFINE_bool(2D, false, "Creates 2D density maps");
DEFINE_string(inFolder, "binaryFiles/",
	"Path to binary files");
DEFINE_string(outFolder, "densityMaps/",
	"Path to output folder");
DEFINE_string(zerosFolder, "densityMaps/zeros/",
	"Path to folder where the pixel cordinates of (0,0) will be written to");
DEFINE_string(voxelFolder, "voxelGrids/",
	"Path to the folder where the voxelGrids are saved to.");
DEFINE_string(rotFolder, "densityMaps/rotations/",
	"Path to folder containing the dominate direction rotations");
DEFINE_string(dataPath, "/home/erik/Projects/3DscanData/DUC/Floor1/",
	"Path to location where program will search for the folders it needs");
DEFINE_double(scale, -1, "Scale used to size the density maps.  If -1, it will be looked up");
DEFINE_int32(startIndex, 0, "Number to start with");
DEFINE_int32(numScans, -1, "Number to process, -1 or default implies all scans");

static std::map<std::string, double> buildingToScale = {{"duc", 73.5}, {"cse", 98.0}};



DensityMapsManager::DensityMapsManager(const std::string & commandLine) {
	std::vector<std::string> v;
	std::istringstream is (commandLine);
	std::string tmp;
	while(is >> tmp)
		v.push_back(tmp);
	char ** argv = new char * [v.size() + 1];
	for (int i = 0; i < v.size(); ++i) {
		argv[i] = &v[i][0];
	}
	argv[v.size()] = NULL;
	DensityMapsManager(v.size(), argv);
}

DensityMapsManager::DensityMapsManager(int argc, char * argv[]) {
	this->resetFlags(argc, argv);
}


void DensityMapsManager::resetFlags(const std::string & commandLine) {
	std::vector<std::string> v;
	std::istringstream is (commandLine);
	std::string tmp;
	while(is >> tmp)
		v.push_back(tmp);
	char ** argv = new char * [v.size() + 1];
	for (int i = 0; i < v.size(); ++i) {
		argv[i] = &v[i][0];
	}
	argv[v.size()] = NULL;
	resetFlags(v.size(), argv);
}

void DensityMapsManager::resetFlags(int argc, char * argv[]) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);
		FLAGS_inFolder = FLAGS_dataPath + FLAGS_inFolder;
		FLAGS_outFolder = FLAGS_dataPath + FLAGS_outFolder;
		FLAGS_zerosFolder = FLAGS_dataPath + FLAGS_zerosFolder;
		FLAGS_voxelFolder = FLAGS_dataPath + FLAGS_voxelFolder;
		FLAGS_rotFolder = FLAGS_dataPath + FLAGS_rotFolder;

		if (!FLAGS_2D && !FLAGS_3D) 
			FLAGS_2D = FLAGS_3D = true;

		if(!FLAGS_pe && !FLAGS_fe)
			FLAGS_pe = FLAGS_fe = true;
		
		DIR *dir;
		struct dirent *ent;
		if ((dir = opendir (FLAGS_inFolder.data())) != NULL) {
		  /* Add all the files and directories to a std::vector */
		  while ((ent = readdir (dir)) != NULL) {
		  	std::string fileName = ent->d_name;
		  	if (fileName != ".." && fileName != "."){
		  		binaryNames.push_back(fileName);
		  	}
		  }
		  closedir (dir);
		}  else {
		  /* could not open directory */
		  perror ("");
		  exit(EXIT_FAILURE);
		}

		sort(binaryNames.begin(), binaryNames.end(), 
			[](const std::string & a, const std::string & b) {
					int numA = std::stoi(a.substr(a.find(".") - 3, 3));
					int numB = std::stoi(b.substr(b.find(".") - 3, 3));
					return numA < numB;
			});

		if ((dir = opendir (FLAGS_rotFolder.data())) != NULL) {
		  /* Add all the files and directories to a std::vector */
		  while ((ent = readdir (dir)) != NULL) {
		  	std::string fileName = ent->d_name;
		  	if (fileName != ".." && fileName != "."){
		  		rotationsFiles.push_back(fileName);
		  	}
		  }
		  closedir (dir);
		}  else {
		  /* could not open directory */
		  perror ("");
		  exit(EXIT_FAILURE);
		}
		sort(rotationsFiles.begin(), rotationsFiles.end());

		if (binaryNames.size() != rotationsFiles.size())
			std::cout << "Not the same number of binaryFiles as rotationsFiles" << std::endl;

		std::string buildName = rotationsFiles[0].substr(0, 3);
		std::locale loc;
		for (int i = 0; i < buildName.length(); ++i) {
			buildName[i] = std::tolower(buildName[i], loc);
		}
		if (FLAGS_scale == -1)
			FLAGS_scale = buildingToScale.find(buildName)->second;

		this->current = 0;
}

void DensityMapsManager::run() {
	rotationFile = FLAGS_rotFolder + rotationsFiles[current]; 
	fileName = FLAGS_inFolder + binaryNames[current];

	scanNumber = fileName.substr(fileName.find(".") - 3, 3);
	buildName = fileName.substr(fileName.rfind("/") + 1, 3);

	std::cout << scanNumber << std::endl;

	std::ifstream binaryReader (rotationFile, std::ios::in | std::ios::binary);
	R.assign(NUM_ROTS, Eigen::Matrix3d());
	for (int i = 0; i < R.size(); ++i) {
	  binaryReader.read(reinterpret_cast<char *>(R[i].data()),
	    sizeof(Eigen::Matrix3d));
	}
	binaryReader.close();

  binaryReader.open(fileName, std::ios::in | std::ios::binary);

  int columns, rows;
 	binaryReader.read(reinterpret_cast<char *> (& columns), sizeof(int));
 	binaryReader.read(reinterpret_cast<char *> (& rows), sizeof(int));
 	pointsWithCenter.clear();
  pointsWithCenter.reserve(columns*rows);
  pointsNoCenter.clear();
  pointsNoCenter.reserve(columns*rows);

  for (int k = 0; k < columns * rows; ++k) {
    Eigen::Vector3f point;
		binaryReader.read(reinterpret_cast<char *> (point.data()), sizeof(point));

		point[1] *= -1.0;

		if (!(point[0] || point[1] || point[2]))
			continue;

		pointsWithCenter.push_back(point);

		if (point[0]*point[0] + point[1]*point[1] > 0.8)
			pointsNoCenter.push_back(pointsWithCenter.back());
	}
	binaryReader.close();
}

bool DensityMapsManager::hasNext() {
	return current < binaryNames.size();
}

void DensityMapsManager::setNext() {
	++current;
}

void DensityMapsManager::get2DPointNames(std::vector<std::string> & names) {
	for (int r = 0; r < NUM_ROTS; ++r) {
	  names.push_back(FLAGS_outFolder + "R" + std::to_string(r)
	  	+ "/" + buildName + "_point_" + scanNumber + ".png");
	}
}

void DensityMapsManager::get3DPointNames(std::vector<std::string> & names) {
	for (int r = 0; r < NUM_ROTS; ++r) {
	  names.push_back(FLAGS_voxelFolder + "R" + std::to_string(r)
	  	+ "/" + buildName + "_point_" + scanNumber + ".dat");
	}
}

void DensityMapsManager::get2DFreeNames(std::vector<std::string> & names) {
	for (int r = 0; r < NUM_ROTS; ++r) {
	  names.push_back(FLAGS_outFolder + "R" + std::to_string(r)
	  	+ "/" + buildName + "_freeSpace_" + scanNumber + ".png");
	}
}

void DensityMapsManager::get3DFreeNames(std::vector<std::string> & names) {
	for (int r = 0; r < NUM_ROTS; ++r) {
	  names.push_back(FLAGS_voxelFolder + "R" + std::to_string(r)
	  	+ "/" + buildName + "_freeSpace_" + scanNumber + ".dat");
	}
}

std::string DensityMapsManager::getZerosName() {
	return FLAGS_zerosFolder + buildName + "zeros" + scanNumber + ".dat";
}

std::string DensityMapsManager::getMetaDataName() {
	return FLAGS_voxelFolder + "metaData/" + buildName + "_scan_"
		+ scanNumber + ".dat";
}

static bool fexists(const std::string & name) {
	std::ifstream in (name, std::ios::in);
	return in.is_open();
}

bool DensityMapsManager::exists2D() {
	std::vector<std::string> names;
	if (!FLAGS_pe) {
		get2DFreeNames(names);
		bool exists = true;
		for (auto & n : names)
			exists = exists && fexists(n);
		return exists;
	}
	if (!FLAGS_fe) {
		get2DPointNames(names);
		bool exists = true;
		for (auto & n : names)
			exists = exists && fexists(n);
		return exists;
	}

	get2DPointNames(names);
	get2DFreeNames(names);
	bool exists = true;
	for (auto & n : names)
		exists = exists && fexists(n);
	return exists;
}

bool DensityMapsManager::exists3D() {
	std::vector<std::string> names;
	if (!FLAGS_pe) {
		get3DFreeNames(names);
		bool exists = true;
		for (auto & n : names)
			exists = exists && fexists(n);
		return exists;
	}
	if (!FLAGS_fe) {
		get3DPointNames(names);
		bool exists = true;
		for (auto & n : names)
			exists = exists && fexists(n);
		return exists;
	}

	get3DPointNames(names);
	get3DFreeNames(names);
	bool exists = true;
	for (auto & n : names)
		exists = exists && fexists(n);
	return exists;
}

BoundingBox::BoundingBox(const std::vector<Eigen::Vector3f> * points,
	Eigen::Vector3f && range) {
	this->points = points;
	this->range = range;
}

BoundingBox::BoundingBox(const std::vector<Eigen::Vector3f> * points,
	Eigen::Vector3f & range) {
	this->points = points;
	this->range = range;
}

void BoundingBox::run() {
	this->average = Eigen::Vector3f::Zero();
	this->sigma = Eigen::Vector3f::Zero();
	for (auto & point : *points) {
		average += point;
	}
	average /= points->size();

	for (auto & point : *points)
		for (int i = 0; i < 3; ++i)
			sigma[i] += (point[i] - average[i])*(point[i] - average[i]);
		
	
	sigma /= points->size() - 1;
	for (int i = 0; i < 3; ++i)
		sigma[i] = sqrt(sigma[i]);
}

void BoundingBox::setRange(Eigen::Vector3f && range) {
	this->range = range;
}

void BoundingBox::setRange(Eigen::Vector3f & range) {
	this->range = range;
}

void BoundingBox::getBoundingBox(Eigen::Vector3f & min, 
	Eigen::Vector3f & max) const {

	Eigen::Vector3f delta;
	for (int i = 0; i < delta.size(); ++i)
	  delta[i] = 1.1*range[i]*sigma[i];
	
	min = average - delta/2.0;
	max = average + delta/2.0;
}


CloudAnalyzer2D::CloudAnalyzer2D(const std::vector<Eigen::Vector3f> * points,
	const std::vector<Eigen::Matrix3d> * R,
	const BoundingBox * bBox) {
	this->bBox = bBox;
	this->points = points;
	this->R = R;
}

void CloudAnalyzer2D::initalize(double scale) {
	bBox->getBoundingBox(pointMin, pointMax);

	zScale = (float)numZ/(pointMax[2] - pointMin[2]);

	numX = scale * (pointMax[0] - pointMin[0]);
	numY = scale * (pointMax[1] - pointMin[1]);


	pointsPerVoxel.assign(numY, Eigen::MatrixXi::Zero(numZ, numX));


	for (auto & point : *points){
	 	const int x = scale*(point[0] - pointMin[0]);
		const int y = scale*(point[1] - pointMin[1]);
		const int z = zScale*(point[2] - pointMin[2]);
		   
		if (x < 0 || x >= numX)
			continue;
		if (y < 0 || y >= numY)
			continue; 
		if ( z < 0 || z >= numZ)
			continue;

	  ++pointsPerVoxel[y](z, x); 
	}

	zeroZero = Eigen::Vector3d(-pointMin[0]*FLAGS_scale, -pointMin[1]*FLAGS_scale, 0);
}

void CloudAnalyzer2D::examinePointEvidence() {
	pointEvidence.clear();
	Eigen::MatrixXf total = Eigen::MatrixXf::Zero (numY, numX);
	for (int i = 0; i < numX; ++i)
		for (int j = 0; j < numY; ++j)
			for (int k = 0; k < numZ; ++k)
				if (pointsPerVoxel[j](k,i))
					++total(j,i);


	double average, sigma;
	average = sigma = 0;
	int count = 0;
	float minV = 1e10;
	float maxV = 0;
	const float * dataPtr = total.data();
	for (int i = 0; i < total.size(); ++i) {
		if (*(dataPtr+ i)) {
			++count;
			average+= *(dataPtr + i);
			minV = std::min(minV, *(dataPtr+i));
			maxV = std::max(maxV, *(dataPtr + i));
		}
	}

	average = average/count;

	for (int i = 0; i < total.size(); ++i) {
		if (*(dataPtr + i) !=0)
			sigma += (*(dataPtr + i) - average)*(*(dataPtr + i)- average);
	}

	sigma = sigma/(count-1);
	sigma = sqrt(sigma);


	if (!FLAGS_quiteMode) {
		std::cout << "Average     Sigma" << std::endl << average << "     " << sigma << std::endl;
		std::cout << "Max     Min" << std::endl << maxV << "      " << minV << std::endl;
	}


	int newRows = std::max(total.rows(), total.cols());
	int newCols = newRows;
	int dX = (newCols - total.cols())/2.0;
	int dY = (newRows - total.rows())/2.0;
	Eigen::Vector3d newZZ = zeroZero;
	newZZ[0] += dX;
	newZZ[1] += dY;

	imageZeroZero = Eigen::Vector2i(newZZ[0], newZZ[1]);

	for (int r = 0; r < NUM_ROTS; ++r) {
		cv::Mat heatMap  (newRows, newCols, CV_8UC1, cv::Scalar::all(255));
		for (int j = 0; j < heatMap.rows; ++j) {
			uchar * dst = heatMap.ptr<uchar>(j);
			for (int i = 0; i < heatMap.cols; ++i) {
				const Eigen::Vector3d pixel (i, j, 0);
				const Eigen::Vector3d src = R->at(r)*(pixel - newZZ) + zeroZero;
				// const Eigen::Vector3d & src = pixel;

				if (src[0] < 0 || src[0] >= total.cols())
					continue;
				if (src[1] < 0 || src[1] >= total.rows())
					continue;

				const double count = total(src[1], src[0]);
				if (count > 0) {
					const int gray = cv::saturate_cast<uchar>(
						255.0 * (count - average - sigma) 
						 	/ (3.0 * sigma));
					dst[i] = 255 - gray;
				}
			}
		}
		if (FLAGS_preview) {
			cv::imshow("Preview", heatMap);
			cv::waitKey(0);
		}

		pointEvidence.push_back(heatMap);
	}
}

void CloudAnalyzer2D::examineFreeSpaceEvidence() {
	freeSpaceEvidence.clear();
	Eigen::Vector3f cameraCenter = -1.0*pointMin;

	std::vector<Eigen::MatrixXi> numTimesSeen (numX, Eigen::MatrixXi::Zero(numZ, numY));

	for (int i = 0; i < numX; ++i) {
		for (int j = 0; j < numY; ++j) {
			for (int k = 0; k < numZ; ++k) {
				if (pointsPerVoxel[j](k,i) == 0)
					continue;

				float ray[3];
				ray[0] = i - cameraCenter[0]*FLAGS_scale;
				ray[1] = j - cameraCenter[1]*FLAGS_scale;
				ray[2] = k - cameraCenter[2]*zScale;
				float length = sqrt(ray[0]*ray[0] + ray[1]*ray[1] + ray[2]*ray[2]);
				float unitRay[3];
				unitRay[0] = ray[0]/length;
				unitRay[1] = ray[1]/length;
				unitRay[2] = ray[2]/length;
				int voxelHit [3];
				for (int a = 0; a <= ceil(length); ++a) {
			
					voxelHit[0] = floor(cameraCenter[0]*FLAGS_scale + a*unitRay[0]);
					voxelHit[1] = floor(cameraCenter[1]*FLAGS_scale + a*unitRay[1]);
					voxelHit[2] = floor(cameraCenter[2]*zScale + a*unitRay[2]);

					if (voxelHit[0] < 0 || voxelHit[0] >= numX)
						continue;
					if (voxelHit[1] < 0 || voxelHit[1] >= numY)
						continue;
					if (voxelHit[2] < 0 || voxelHit[2] >= numZ)
						continue;
					numTimesSeen[voxelHit[0]](voxelHit[2], voxelHit[1])
						+= pointsPerVoxel[j](k,i);
				}
			}
		}
	}
	
	for (int i = -0.836*FLAGS_scale; i < 0.836*FLAGS_scale; ++i)
		for (int j = -sqrt(0.7*FLAGS_scale*FLAGS_scale - i*i); 
			j < sqrt(0.7*FLAGS_scale*FLAGS_scale - i*i); ++j)
			for (int k = numZ/2; k < numZ; ++k)
				++numTimesSeen[cameraCenter[0]*FLAGS_scale + i]
					(k, cameraCenter[1]*FLAGS_scale + j);


	Eigen::MatrixXd collapsedCount (numY, numX);

	for (int i = 0; i < numX; ++i) {
		for (int j = 0; j < numY; ++j) {
			int count = 0;
			for (int k = 0; k < numZ; ++k) {
				if (numTimesSeen[i](k,j)) {
					count++;
				}
			}
			collapsedCount(j,i) = count;
		}
	}

	double average, sigma;
	average = sigma = 0;
	size_t count = 0;
	const double * vPtr = collapsedCount.data();
	
	for (int i = 0; i < collapsedCount.size(); ++i) {
		if (*(vPtr + i) != 0) {
			average += *(vPtr + i);
			++count;
		}
	}

	average = average/count;

	for (int i = 0; i < collapsedCount.size(); ++i){
		if (*(vPtr + i)!=0)
			sigma += (*(vPtr + i)-average)*(*(vPtr + i)-average);
	}
	sigma = sigma/(count - 1);
	sigma = sqrt(sigma);


	int newRows = std::max(collapsedCount.rows(), collapsedCount.cols());
	int newCols = newRows;
	int dX = (newCols - collapsedCount.cols())/2.0;
	int dY = (newRows - collapsedCount.rows())/2.0;
	Eigen::Vector3d newZZ = zeroZero;
	newZZ[0] += dX;
	newZZ[1] += dY;

	for (int r = 0; r < NUM_ROTS; ++r) {
		cv::Mat heatMap (newRows, newCols, CV_8UC1, cv::Scalar::all(255));
		for (int j = 0; j < heatMap.rows; ++j) {
			uchar * dst = heatMap.ptr<uchar>(j);
			for (int i = 0; i < heatMap.cols; ++i) {
				const Eigen::Vector3d pixel (i, j, 0);
				const Eigen::Vector3d src = R->at(r)*(pixel - newZZ) + zeroZero;

				if (src[0] < 0 || src[0] >= collapsedCount.cols())
					continue;
				if (src[1] < 0 || src[1] >= collapsedCount.rows())
					continue;

				const double count = collapsedCount(src[1], src[0]);
				if (count > 0) {
					const int gray = cv::saturate_cast<uchar>(
						255.0 * ((count - average)/sigma + 1.0 ));
					dst[i] = 255 - gray;
				}
			}				
		}
		if (FLAGS_preview) {
			cv::imshow("Preview", heatMap);
			cv::waitKey(0);
		}

		freeSpaceEvidence.push_back(heatMap);
	}
}


const std::vector<cv::Mat> & CloudAnalyzer2D::getPointEvidence() {
	return pointEvidence;
}

const std::vector<cv::Mat> & CloudAnalyzer2D::getFreeSpaceEvidence() {
	return freeSpaceEvidence;
}

Eigen::Vector2i CloudAnalyzer2D::getImageZeroZero() {
	return imageZeroZero;
}