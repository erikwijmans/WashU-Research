/*Scanner units are proabaly in meters */

#include "scanDensity_scanDensity.h"
#include "scanDensity_3DInfo.h"
#include <omp.h>




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
DEFINE_double(scale, 73.5, "scale used to size the density maps");
DEFINE_int32(startIndex, 0, "Number to start with");
DEFINE_int32(numScans, -1, "Number to process, -1 or default implies all scans");


int main(int argc, char *argv[]) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	FLAGS_inFolder = FLAGS_dataPath + FLAGS_inFolder;
	FLAGS_outFolder = FLAGS_dataPath + FLAGS_outFolder;
	FLAGS_zerosFolder = FLAGS_dataPath + FLAGS_zerosFolder;
	FLAGS_voxelFolder = FLAGS_dataPath + FLAGS_voxelFolder;
	FLAGS_rotFolder = FLAGS_dataPath + FLAGS_rotFolder;

	if(!FLAGS_2D && !FLAGS_3D) 
		FLAGS_2D = FLAGS_3D = true;
	
	cvNamedWindow("Preview", CV_WINDOW_NORMAL);

	std::vector<std::string> binaryNames, rotationsFiles;
	
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (FLAGS_inFolder.data())) != NULL) {
	  /* Add all the files and directories to a std::vector */
	  while ((ent = readdir (dir)) != NULL) {
	  	std::string fileName = ent->d_name;
	  	if(fileName != ".." && fileName != "."){
	  		binaryNames.push_back(fileName);
	  	}
	  }
	  closedir (dir);
	}  else {
	  /* could not open directory */
	  perror ("");
	  return EXIT_FAILURE;
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
	  	if(fileName != ".." && fileName != "."){
	  		rotationsFiles.push_back(fileName);
	  	}
	  }
	  closedir (dir);
	}  else {
	  /* could not open directory */
	  perror ("");
	  return EXIT_FAILURE;
	}
	sort(rotationsFiles.begin(), rotationsFiles.end());
	
	if(FLAGS_numScans == -1)
		FLAGS_numScans = binaryNames.size() - FLAGS_startIndex;

	
	for(int i = FLAGS_startIndex; i < FLAGS_startIndex + FLAGS_numScans; ++i) {
		const std::string binaryFilePath = FLAGS_inFolder + binaryNames[i];
		const std::string rotationFile = FLAGS_rotFolder + rotationsFiles[i];

		if(FLAGS_2D)
			analyzeScan(binaryFilePath, FLAGS_outFolder, rotationFile);
		if(FLAGS_3D)
			voxel::analyzeScan3D(binaryFilePath, rotationFile);

	}
	
	std::cout << "Scan Density Done!" << std::endl;

	return 0;
}


DensityMaps::DensityMaps(int argc, char * argv[]) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	if(!FLAGS_2D && !FLAGS_3D) 
		FLAGS_2D = FLAGS_3D = true;
	
	cvNamedWindow("Preview", CV_WINDOW_NORMAL);

	std::vector<std::string> binaryNames, rotationsFiles;
	
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (FLAGS_inFolder.data())) != NULL) {
	  /* Add all the files and directories to a std::vector */
	  while ((ent = readdir (dir)) != NULL) {
	  	std::string fileName = ent->d_name;
	  	if(fileName != ".." && fileName != "."){
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
	  	if(fileName != ".." && fileName != "."){
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
}

void DensityMaps::run() {
	for(int i =0; i < rotationsFiles.size(); ++i) {
		const std::string binaryFilePath = FLAGS_inFolder + binaryNames[i];
		const std::string rotationFile = FLAGS_rotFolder + rotationsFiles[i];

		analyzeScan(binaryFilePath, FLAGS_outFolder, rotationFile);
	}
}

void DensityMaps::run(int startIndex, int numScans) {
	for(int i = startIndex; i < startIndex + numScans; ++i) {
		const std::string binaryFilePath = FLAGS_inFolder + binaryNames[i];
		const std::string rotationFile = FLAGS_rotFolder + rotationsFiles[i];

		analyzeScan(binaryFilePath, FLAGS_outFolder, rotationFile);
	}
}

void analyzeScan(const std::string & fileName, const std::string & outputFolder,
	const std::string & rotationFile) {
	const std::string scanNumber = fileName.substr(fileName.find(".") - 3, 3);
	const std::string buildName = fileName.substr(fileName.rfind("/") + 1, 3);

	std::cout << scanNumber << std::endl;

	if(FLAGS_pe && !FLAGS_redo) {
		const std::string imageName = outputFolder + "R" + std::to_string(0) + "/"
		 + buildName + "_point_" + scanNumber + ".png";
		cv::Mat img = cv::imread(imageName);
		if(img.data)
			return;
	}
	if(FLAGS_fe && !FLAGS_redo) {
		const std::string imageName = outputFolder + "R" + std::to_string(0) + "/"
			+ buildName + "_freeSpace_" + scanNumber + ".png";
		cv::Mat img = cv::imread(imageName);
		if(img.data)
			return;
	}

	if(!FLAGS_pe && !FLAGS_fe && !FLAGS_redo) {
		const std::string imageName = outputFolder + "R" + std::to_string(0) + "/"
			+ buildName + "_point_" + scanNumber + ".png";
		cv::Mat img = cv::imread(imageName);
		const std::string imageName2 = outputFolder + "R" + std::to_string(0) + "/"
			+ buildName + "_freeSpace_" + scanNumber + ".png";
		cv::Mat img2 = cv::imread(imageName2);
		if(img.data && img2.data)
			return;
	}

	std::ifstream binaryReader (rotationFile, std::ios::in | std::ios::binary);
	std::vector<Eigen::Matrix3d> R (NUM_ROTS);
	for (int i = 0; i < R.size(); ++i) {
	  binaryReader.read(reinterpret_cast<char *>(R[i].data()),
	    sizeof(Eigen::Matrix3d));
	}
	binaryReader.close();

  std::ifstream scanFile (fileName, std::ios::in | std::ios::binary);

  int columns, rows;
 	scanFile.read(reinterpret_cast<char *> (& columns), sizeof(int));
 	scanFile.read(reinterpret_cast<char *> (& rows), sizeof(int));

  std::vector<Eigen::Vector3f> points;
  points.reserve(columns*rows);
  for (int k = 0; k < columns * rows; ++k) {
    Eigen::Vector3f point;
		scanFile.read(reinterpret_cast<char *> (point.data()), sizeof(point));

		point[1] *= -1.0;

		if(point[0]*point[0] + point[1]*point[1] < 0.8)
			continue;
		else if(point[0] || point[1] || point[2])
		  points.push_back(point);
		
	}
	scanFile.close();

	const std::string zeroName = FLAGS_zerosFolder + buildName + "_zeros_" + scanNumber + ".dat";
	std::ofstream zZOut (zeroName, std::ios::out | std::ios::binary);


	float pointMax [3], pointMin [3];
	createBoundingBox(pointMin, pointMax, points);
	
	if(FLAGS_pe || (!FLAGS_pe && !FLAGS_fe))
		examinePointEvidence(points, R, pointMin, pointMax, outputFolder, scanNumber, buildName, zZOut);

	if(FLAGS_fe || (!FLAGS_pe && !FLAGS_fe))
		examineFreeSpaceEvidence(points, R, pointMin, pointMax, outputFolder, scanNumber, buildName);

	zZOut.close();
}

void createBoundingBox(float * pointMin, float * pointMax,
	const std::vector<Eigen::Vector3f> & points){
	Eigen::Vector3f average = Eigen::Vector3f::Zero(), 
		sigma = Eigen::Vector3f::Zero();
	

	/*double averageX, averageY, sigmaX, sigmaY, averageZ, sigmaZ;
	averageX = averageY = sigmaX = sigmaY = averageZ = sigmaZ = 0;*/

	for (auto & point : points) {
		average += point;
	}
	average /= points.size();

	for (auto & point : points) {
		for(int i = 0; i < 3; ++i) {
			sigma[i] += (point[i] - average[i])*(point[i] - average[i]);
		}
	}
	sigma /= points.size() - 1;
	for(int i = 0; i < 3; ++i)
		sigma[i] = sqrt(sigma[i]);
	
	if(!FLAGS_quiteMode) {
		std::cout << "average: " << average << std::endl;
		std::cout << "sigma: " << sigma << std::endl;
	}
	Eigen::Vector3f delta;
	delta[0] = 1.1*9*sigma[0];
	delta[1] = 1.1*9*sigma[1];
	delta[2] = 1.1*6*sigma[2];

	pointMin[0] = average[0] - delta[0]/2;
	pointMin[1] = average[1] - delta[1]/2;
	pointMin[2] = average[2] - delta[2]/2;

	pointMax[0] = average[0] + delta[0]/2;
	pointMax[1] = average[1] + delta[1]/2;
	pointMax[2] = average[2] + delta[2]/2;
} 

void examinePointEvidence(const std::vector<Eigen::Vector3f> & points,
	const std::vector<Eigen::Matrix3d> & R,
	const float* pointMin, const float * pointMax, 
	const std::string & outputFolder, const std::string & scanNumber,
	const std::string & buildName,
	std::ofstream & zZOut){
	const int numZ = 100;
	const float zScale = (float)numZ/(pointMax[2] - pointMin[2]);

	const int numCols = FLAGS_scale * (pointMax[0] - pointMin[0]);
	const int numRows = FLAGS_scale * (pointMax[1] - pointMin[1]);


	std::vector<Eigen::MatrixXi> numTimesSeen3D (numRows, Eigen::MatrixXi::Zero(numZ, numCols));

	// PointCloud<PointXYZ> cloud;
	for(auto & point : points){
	 	const int x = FLAGS_scale*(point[0] - pointMin[0]);
		const int y = FLAGS_scale*(point[1] - pointMin[1]);
		const int z = zScale*(point[2] - pointMin[2]);
		   
		if(x < 0 || x >= numCols)
			continue;
		if(y < 0 || y >= numRows)
			continue; 
		if( z < 0 || z >= numZ)
			continue;

	  ++numTimesSeen3D[y](z, x); 
	}
	// io::savePLYFileBinary("output.ply",cloud);

	Eigen::MatrixXf total = Eigen::MatrixXf::Zero (numRows, numCols);
	for(int i = 0; i < numCols; ++i)
		for (int j = 0; j < numRows; ++j)
			for (int k = 0; k < numZ; ++k)
				if(numTimesSeen3D[j](k,i))
					++total(j,i);

	const Eigen::Vector3d zeroZero (-pointMin[0]*FLAGS_scale, -pointMin[1]*FLAGS_scale, 0);
	displayPointEvenidence(total, R, zeroZero, scanNumber, buildName, 3.0, zZOut);
	
}

void displayPointEvenidence(const Eigen::MatrixXf & numTimesSeen,
	const std::vector<Eigen::Matrix3d> & R, const Eigen::Vector3d & zeroZero, 
	const std::string & scanNumber, const std::string & buildName,
	const int bias, std::ofstream & zZOut){

	
	double average, sigma;
	average = sigma = 0;
	int count = 0;
	float minV = 1e10;
	float maxV = 0;
	const float * dataPtr = numTimesSeen.data();
	for(int i = 0; i < numTimesSeen.size(); ++i) {
		if(*(dataPtr+ i)) {
			++count;
			average+= *(dataPtr + i);
			minV = std::min(minV, *(dataPtr+i));
			maxV = std::max(maxV, *(dataPtr + i));
		}
	}

	average = average/count;

	for(int i = 0; i < numTimesSeen.size(); ++i) {
		if(*(dataPtr + i) !=0)
			sigma += (*(dataPtr + i) - average)*(*(dataPtr + i)- average);
	}

	sigma = sigma/(count-1);
	sigma = sqrt(sigma);


	if(!FLAGS_quiteMode)
	{
		std::cout << "Average     Sigma" << std::endl << average << "     " << sigma << std::endl;
		std::cout << "Max     Min" << std::endl << maxV << "      " << minV << std::endl;
	}


	int newRows = std::max(numTimesSeen.rows(), numTimesSeen.cols());
	int newCols = newRows;
	int dX = (newCols - numTimesSeen.cols())/2.0;
	int dY = (newRows - numTimesSeen.rows())/2.0;
	Eigen::Vector3d newZZ = zeroZero;
	newZZ[0] += dX;
	newZZ[1] += dY;

	for(int r = 0; r < NUM_ROTS; ++r) {
		const std::string imageName = FLAGS_outFolder + "R" + std::to_string(r) + "/"
			+ buildName + "_point_" + scanNumber + ".png";

		cv::Mat heatMap  (newRows, newCols, CV_8UC1, cv::Scalar::all(255));
		for (int j = 0; j < heatMap.rows; ++j) {
			uchar * dst = heatMap.ptr<uchar>(j);
			for (int i = 0; i < heatMap.cols; ++i) {
				const Eigen::Vector3d pixel (i, j, 0);
				const Eigen::Vector3d src = R[r]*(pixel - newZZ) + zeroZero;
				// const Eigen::Vector3d & src = pixel;

				if(src[0] < 0 || src[0] >= numTimesSeen.cols())
					continue;
				if(src[1] < 0 || src[1] >= numTimesSeen.rows())
					continue;

				const double count = numTimesSeen(src[1], src[0]);
				if(count > 0) {
					const int gray = cv::saturate_cast<uchar>(
						255.0 * (count - average - sigma) 
						 	/ (bias * sigma));
					dst[i] = 255 - gray;
				}


				/*if(numTimesSeen(j,i)){
					const int gray = cv::saturate_cast<uchar>(
						255.0 * (numTimesSeen(j,i) - average - 1.5*sigma) 
						 	/ (bias * sigma));
					const Eigen::Vector3d point (i, j, 0);
					const Eigen::Vector3d newPoint = R[r]*(point - zZ) + zZ;

					if(newPoint[1] < 0 || newPoint[1] >= heatMap.rows)
						continue;
					if(newPoint[0] < 0 || newPoint[0] >= heatMap.cols)
						continue;

					heatMap.at<uchar>(newPoint[1], newPoint[0]) = 255 - gray;
					int red, green, blue;
					if (gray < 128) {
						red = 0;
						blue = 2 * gray;
						green = 255 - blue;
					} else {
						blue = 0;
						red = 2 * (gray - 128);
						green = 255 - red;
					}
					dst[j*3] = blue;
					dst[j*3 +1] = green;
					dst[j*3 + 2] = red;
				} */
			} 
		}



		/*for (int y = heatMap.rows/2; y <= heatMap.rows/2+3; ++y)
		{
			for (int x = heatMap.cols/2; x <= heatMap.cols/2+3; ++x)
			{
				heatMap.at<uchar>(y,x) = 0;
			}
		}*/
		
		
		if(FLAGS_preview) {
			cv::imshow("Preview", heatMap);
			cv::waitKey(0);
		}
		
		cv::imwrite(imageName, heatMap);

		Eigen::Vector2i tmp (newZZ[0], newZZ[1]);
		zZOut.write(reinterpret_cast<const char *>(tmp.data()), sizeof(tmp));
	}
}

void examineFreeSpaceEvidence(const std::vector<Eigen::Vector3f> & points, 
	const std::vector<Eigen::Matrix3d> & R,
	const float* pointMin, const float * pointMax,
	const std::string & outputFolder, const std::string & scanNumber,
	const std::string & buildName){

	const int numX = FLAGS_scale * (pointMax[0] - pointMin[0]);
	const int numY = FLAGS_scale * (pointMax[1] - pointMin[1]);
	const int numZ = 100;

	const float zScale = (float)numZ/(pointMax[2] - pointMin[2]);
	float cameraCenter [3];
	cameraCenter[0] = -1*pointMin[0];
	cameraCenter[1] = -1*pointMin[1];
	cameraCenter[2] = -1*pointMin[2];

	std::vector<Eigen::MatrixXi> pointsPerVoxel (numZ, Eigen::MatrixXi::Zero(numY, numX));
	std::vector<Eigen::MatrixXi> numTimesSeen4C (numX, Eigen::MatrixXi::Zero(numZ, numY));

	for(auto & point : points) {
		int x = floor((point[0]- pointMin[0]) * FLAGS_scale);
		int y = floor((point[1] - pointMin[1]) * FLAGS_scale);
		int z = floor((point[2] - pointMin[2]) * zScale);

		if(x < 0 || x >= numX)
			continue;
		if(y < 0 || y >= numY)
			continue;
		if(z < 0 || z >= numZ)
			continue;

		++pointsPerVoxel[z](y,x);
	}

	
	for (int k = 0; k < numZ; ++k) {
		for (int i = 0; i < numX; ++i) {
			for (int j = 0; j < numY; ++j) {
				if(pointsPerVoxel[k](j,i)==0)
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

					if(voxelHit[0] < 0 || voxelHit[0] >= numX)
						continue;
					if(voxelHit[1] < 0 || voxelHit[1] >= numY)
						continue;
					if(voxelHit[2] < 0 || voxelHit[2] >= numZ)
						continue;
					numTimesSeen4C[voxelHit[0]](voxelHit[2], voxelHit[1])
						+= pointsPerVoxel[k](j,i);
				}
			}
		}
	}
	
	for(int i = -0.836*FLAGS_scale; i < 0.836*FLAGS_scale; ++i)
		for(int j = -sqrt(0.7*FLAGS_scale*FLAGS_scale - i*i); 
			j < sqrt(0.7*FLAGS_scale*FLAGS_scale - i*i); ++j)
			for(int k = numZ/2; k < numZ; ++k)
				++numTimesSeen4C[cameraCenter[0]*FLAGS_scale + i]
					(k, cameraCenter[1]*FLAGS_scale + j);
	

	const Eigen::Vector3d zeroZero (-pointMin[0]*FLAGS_scale, -pointMin[1]*FLAGS_scale, 0);
	collapseFreeSpaceEvidence(numTimesSeen4C, R, zeroZero, numZ, numY, numX,
	 	scanNumber, buildName);
}

void showSlices(const Eigen::MatrixXi & currentSlice,
	const int numZ, const int numY, const int numX, const std::string & scanNumber){


	float average, sigma;
	average = sigma = 0;
	size_t count = 0;
	
	for(int i = 0; i < currentSlice.size(); ++i) {
		if(*(currentSlice.data() + i) ==0)
			continue;
		average += *(currentSlice.data() + i);
		count ++;
	}

	average = average/count;

	for (int i = 0; i < currentSlice.size(); ++i) {
		if(*(currentSlice.data() + i) !=0)
			sigma+=(*(currentSlice.data() + i) - average)*
				(*(currentSlice.data() + i) - average);
	}
	sigma = sigma/(count - 1);
	sigma = sqrt(sigma);

	cv::Mat sliceMap (numY, numX, CV_8UC3, cv::Scalar::all(255));

	for (int j = 0; j < sliceMap.rows; ++j) {
		uchar * dst = sliceMap.ptr<uchar>(j);
		
		for (int i = 0; i < sliceMap.cols; ++i) {
			if(currentSlice(j,i) != 0){
				const int gray = cv::saturate_cast<uchar>(255.0 * (currentSlice(j,i)
					  - average) / ((3 * sigma) + 1.0) / 2.0);
				int red, green, blue;
				if (gray < 128) {
					red = 0;
					green = 2 * gray;
					blue = 255 - green;
				} else {
					blue = 0;
					red = 2 * (gray - 128);
					green = 255 - red;
				}
				dst[i*3] = blue;
				dst[i*3 +1] = green;
				dst[i*3 + 2] = red;
			}
		} 
	}
	/*const std::string imageName = FLAGS_outFolder + "DUC_freeSpace_" + scanNumber + ".png";
	cv::imwrite(imageName, sliceMap);*/

	cv::imshow("Preview", sliceMap);
	cv::waitKey(0); 


}


void collapseFreeSpaceEvidence(const std::vector<Eigen::MatrixXi> & numTimesSeen,
	const std::vector<Eigen::Matrix3d> & R, const Eigen::Vector3d & zeroZero, 
	const int numZ, const int numY, const int numX,
	const std::string & scanNumber, const std::string & buildName){

	Eigen::MatrixXd collapsedMean (numY, numX);

	for (int i = 0; i < numX; ++i) {
		for (int j = 0; j < numY; ++j) {
			// double mean = 0;
			int count = 0;
			for (int k = 0; k < numZ; ++k) {
				if(numTimesSeen[i](k,j)) {
					// mean += static_cast<double>(numTimesSeen[i](k,j));
					count++;
				}
			}
			// mean = mean/numZ;
			collapsedMean(j,i) = count;
		}
	}

	displayCollapsed(collapsedMean, R, zeroZero, scanNumber, buildName);
	
	
}

void displayCollapsed(const Eigen::MatrixXd & numTimesSeen,
	const std::vector<Eigen::Matrix3d> & R, const Eigen::Vector3d & zeroZero, 
	const std::string & scanNumber, const std::string & buildName){

	double average, sigma;
	average = sigma = 0;
	size_t count = 0;
	const double * vPtr = numTimesSeen.data();
	
	for(int i = 0; i < numTimesSeen.size(); ++i) {
		if(*(vPtr + i) != 0) {
			average += *(vPtr + i);
			++count;
		}
		
	}

	average = average/count;

	for(int i = 0; i < numTimesSeen.size(); ++i){
		if(*(vPtr + i)!=0)
			sigma += (*(vPtr + i)-average)*(*(vPtr + i)-average);
	}
	sigma = sigma/(count - 1);
	sigma = sqrt(sigma);


	int newRows = std::max(numTimesSeen.rows(), numTimesSeen.cols());
	int newCols = newRows;
	int dX = (newCols - numTimesSeen.cols())/2.0;
	int dY = (newRows - numTimesSeen.rows())/2.0;
	Eigen::Vector3d newZZ = zeroZero;
	newZZ[0] += dX;
	newZZ[1] += dY;

	for(int r = 0; r < NUM_ROTS; ++r) {
			const std::string imageName = FLAGS_outFolder + "R" + std::to_string(r) + "/"
				+ buildName + "_freeSpace_" + scanNumber + ".png";

			cv::Mat heatMap (newRows, newCols, CV_8UC1, cv::Scalar::all(255));
			for (int j = 0; j < heatMap.rows; ++j) {
				uchar * dst = heatMap.ptr<uchar>(j);
				for (int i = 0; i < heatMap.cols; ++i) {
					const Eigen::Vector3d pixel (i, j, 0);
					const Eigen::Vector3d src = R[r]*(pixel - newZZ) + zeroZero;

					if(src[0] < 0 || src[0] >= numTimesSeen.cols())
						continue;
					if(src[1] < 0 || src[1] >= numTimesSeen.rows())
						continue;

					const double count = numTimesSeen(src[1], src[0]);
					if(count > 0) {
						const int gray = cv::saturate_cast<uchar>(
							255.0 * ((count - average)/sigma + 1.0 ));
						dst[i] = 255 - gray;
					}
						

					/*if(numTimesSeen(j,i)){
						const int gray = cv::saturate_cast<uchar>(
							255.0 * ((numTimesSeen(j,i) - average) 
							 	/ (sigma) + 1.0));
						const Eigen::Vector3d point (i, j, 0);
						Eigen::Vector3d newPoint = R[r]*(point - zZ) + zZ;
						
						newPoint[2] = floor(newPoint[2]);


						if(newPoint[1] < 0 || newPoint[1] >= heatMap.rows)
							continue;
						if(newPoint[0] < 0 || newPoint[0] >= heatMap.cols)
							continue;

						heatMap.at<uchar>(newPoint[1], newPoint[0]) = 255 - gray;*/
						/*int red, green, blue;
						if (gray < 128) {
							red = 0;
							blue = 2 * gray;
							green = 255 - blue;
						} else {
							blue = 0;
							red = 2 * (gray - 128);
							green = 255 - red;
						}
						dst[j*3] = blue;
						dst[j*3 +1] = green;
						dst[j*3 + 2] = red;
					} */
				} 
			}



			/*for (int y = heatMap.rows/2; y <= heatMap.rows/2+3; ++y)
			{
				for (int x = heatMap.cols/2; x <= heatMap.cols/2+3; ++x)
				{
					heatMap.at<uchar>(y,x) = 0;
				}
			}*/
			
			
			if(FLAGS_preview) {
				cv::imshow("Preview", heatMap);
				cv::waitKey(0);
			}
			
			cv::imwrite(imageName, heatMap);
		}
}

