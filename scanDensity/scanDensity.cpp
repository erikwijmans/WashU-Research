/*Scanner units are proabaly in meters */
#include "scanDensity_scanDensity.h"
#include "scanDensity_3DInfo.h"


/*#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>*/

static cv::Mat heatMap;

DEFINE_bool(pe, false, "Tells the program to only examine point evidence");
DEFINE_bool(fe, false, "Tells the program to only examine free space evidence");
DEFINE_bool(quiteMode, true, "Turns of all extrenous statements");
DEFINE_bool(preview, false, "Turns on previews of the output");
DEFINE_bool(redo, false, "Recreates the density map even if it already exists");
DEFINE_bool(3D, true, "writes out 3D voxelGrids");
DEFINE_string(inFolder, "/home/erik/Projects/3DscanData/DUC/Floor1/binaryFiles/",
	"Path to binary files");
DEFINE_string(outFolder, "/home/erik/Projects/3DscanData/DUC/Floor1/densityMaps/",
	"Path to output folder");
DEFINE_string(zerosFolder, "/home/erik/Projects/3DscanData/DUC/Floor1/densityMaps/zeros/",
	"Path to folder where the pixel cordinates of (0,0) will be written to");
DEFINE_string(voxelFolder, "/home/erik/Projects/3DscanData/DUC/Floor1/voxelGrids",
	"Path to the folder where the voxelGrids are saved to.");
DEFINE_string(rotFolder, "/home/erik/Projects/3DscanData/DUC/Floor1/densityMaps/rotations/",
	"Path to folder containing the dominate direction rotations");
DEFINE_double(scale, 73.5, "scale used to size the density maps");
DEFINE_int32(startIndex, 0, "Number to start with");
DEFINE_int32(numScans, -1, "Number to process, -1 or default implies all scans");


int main(int argc, char *argv[]) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	
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

	sort(binaryNames.begin(), binaryNames.end());
	if(FLAGS_numScans == -1)
		FLAGS_numScans = binaryNames.size() - FLAGS_startIndex;

	for(int i = FLAGS_startIndex; i < FLAGS_startIndex + FLAGS_numScans; ++i){
		const std::string binaryFilePath = FLAGS_inFolder + binaryNames[i];
		const std::string rotationFile = FLAGS_rotFolder + rotationsFiles[i];
		analyzeScan(binaryFilePath, rotationFile, FLAGS_outFolder);
	}
	
	
	
	std::cout << "Scan Density Done!" << std::endl;

	return 0;
}

void analyzeScan(const std::string & fileName, 
	const std::string & rotationFile, const std::string & outputFolder) {
	const std::string scanNumber = fileName.substr(fileName.find(".") - 3, 3);
	if(!FLAGS_quiteMode)
		std::cout << scanNumber << std::endl;

	if(FLAGS_pe && !FLAGS_redo) {
		const std::string imageName = outputFolder + "DUC_point_" + scanNumber + ".png";
		cv::Mat img = cv::imread(imageName);
		if(img.data)
			return;
	}
	if(FLAGS_fe && !FLAGS_redo) {
		const std::string imageName = outputFolder + "DUC_freeSpace_" + scanNumber + ".png";
		cv::Mat img = cv::imread(imageName);
		if(img.data)
			return;
	}

	if(!FLAGS_pe && !FLAGS_fe && !FLAGS_redo) {
		const std::string imageName = outputFolder + "DUC_point_" + scanNumber + ".png";
		cv::Mat img = cv::imread(imageName);
		const std::string imageName2 = outputFolder + "DUC_freeSpace_" + scanNumber + ".png";
		cv::Mat img2 = cv::imread(imageName2);
		if(img.data && img2.data)
			return;
	}

  std::ifstream scanFile (fileName, std::ios::in | std::ios::binary);

  int columns, rows;
 	scanFile.read(reinterpret_cast<char *> (& columns), sizeof(int));
 	scanFile.read(reinterpret_cast<char *> (& rows), sizeof(int));

  
  float pointMax [3], pointMin[3];
  
  std::vector<Eigen::Vector3f> points;
  int numCenter = 0;
  for (int k = 0; k < columns * rows; ++k) {
    Eigen::Vector3f point;
		scanFile.read(reinterpret_cast<char *> (&point[0]), sizeof(point));

		if(point[0]*point[0] + point[1]*point[1] < 1) {
			if(numCenter%1000 == 0)
				points.push_back(point);
			++numCenter;
		} else
			if(point[0] || point[1] || point[2])
		  	points.push_back(point);
		
	}

	scanFile.close();

	createBoundingBox(pointMin, pointMax, points);

	if(FLAGS_pe || (!FLAGS_pe && !FLAGS_fe))
		examinePointEvidence(points, pointMin, pointMax, outputFolder, scanNumber);

	if(FLAGS_fe || (!FLAGS_pe && !FLAGS_fe))
		examineFreeSpaceEvidence(points, pointMin, pointMax, outputFolder, scanNumber);

	if(FLAGS_3D)
		voxel::createVoxelGrids(points, pointMin, pointMax, rotationFile, scanNumber);
}



void createBoundingBox(float * pointMin, float * pointMax,
	const std::vector<Eigen::Vector3f> & points){
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

	if(!FLAGS_quiteMode)
	{
		std::cout << "averageX: " << averageX << std::endl;
		std::cout << "averageY: " << averageY << std::endl;
		std::cout << "averageZ: " << averageZ << std::endl;
		std::cout << "sigmaX: " << sigmaX << std::endl;
		std::cout << "sigmaY: " << sigmaY << std::endl;
		std::cout << "sigmaZ: " << sigmaZ << std::endl;
	}

	double dX = 1.1*9*sigmaX;
	double dY = 1.1*9*sigmaY;
	double dZ = 1.1*6*sigmaZ;


	pointMin[0] = averageX - dX/2;
	pointMin[1] = averageY - dY/2;
	pointMin[2] = averageZ - dZ/2;

	pointMax[0] = averageX + dX/2;
	pointMax[1] = averageY + dY/2;
	pointMax[2] = averageZ + dZ/2;
} 

void examinePointEvidence(const std::vector<Eigen::Vector3f> & points,
	const float* pointMin, const float * pointMax, 
	const std::string & outputFolder, const std::string & scanNumber){
	const int numZ = 100;
	const float zScale = (float)numZ/(pointMax[2] - pointMin[2]);

	const int numCols = FLAGS_scale * (pointMax[0] - pointMin[0]);
	const int numRows = FLAGS_scale * (pointMax[1] - pointMin[1]);

	heatMap = cv::Mat (numRows, numCols, CV_8UC1, cv::Scalar::all(255));

	std::vector<Eigen::MatrixXi> numTimesSeen3D (heatMap.rows, Eigen::MatrixXi::Zero(heatMap.cols, numZ));

	Eigen::Vector2i zeroZero (-pointMin[0], -pointMin[1]);
	zeroZero *= FLAGS_scale;
	const std::string zeroName = FLAGS_zerosFolder + "DUC_point_" + scanNumber + ".dat";
	std::ofstream out (zeroName, std::ios::out | std::ios::binary);
	out.write(reinterpret_cast<const char *> (&zeroZero[0]), sizeof(zeroZero));
	out.close();


	// PointCloud<PointXYZ> cloud;
	for(auto & point : points){
	 	const int x = FLAGS_scale*(point[0] - pointMin[0]);
		const int y = FLAGS_scale*(point[1] - pointMin[1]);
		const int z = zScale*(point[2] - pointMin[2]);
		   
		if(x <0 || x >= heatMap.cols)
			continue;
		if(y < 0 || y >= heatMap.rows)
			continue; 
		if( z < 0 || z >= numZ)
			continue;

	  ++numTimesSeen3D[y](x, z); 
	    /*if(y>=heatMap.rows/2 && y<= heatMap.rows/2+20
	    	&& x>=heatMap.cols/2+120 && x<=heatMap.cols/2 + 140)
	    	cloud.push_back(PointXYZ(x,y,z));*/
		
		
	}
	// io::savePLYFileBinary("output.ply",cloud);

	Eigen::MatrixXf total = Eigen::MatrixXf::Zero (heatMap.rows, heatMap.cols);
	for(int i = 0; i < heatMap.rows; ++i)
		for (int j = 0; j < heatMap.cols; ++j)
			for (int k = 0; k < numZ; ++k)
				if(numTimesSeen3D[i](j,k))
					++total(i,j);

	const std::string imageName = outputFolder + "DUC_point_" + scanNumber + ".png";
	displayPointEvenidence(total, imageName, 2.0);

}

void displayPointEvenidence(const Eigen::MatrixXf & numTimesSeen, 
	const std::string & imageName,
	const int bias){
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

	
	for (int i = 0; i < heatMap.rows; ++i) {
		uchar * dst = heatMap.ptr<uchar>(i);
		for (int j = 0; j < heatMap.cols; ++j) {
			if(numTimesSeen(i,j)){
				const int gray = cv::saturate_cast<uchar>(
					255.0 * (numTimesSeen(i,j) - average - 1.5*sigma) 
					 	/ (bias * sigma));
				dst[j] = 255 - gray;
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
				dst[j*3 + 2] = red;*/
			}
		} 
	}



	/*for (int y = heatMap.rows/2; y <= heatMap.rows/2+3; ++y)
	{
		for (int x = heatMap.cols/2; x <= heatMap.cols/2+3; ++x)
		{
			heatMap.at<uchar>(y,x) = 0;
		}
	}*/
	
	
	if(FLAGS_preview)
	{
		cv::imshow("Preview", heatMap);
		cv::waitKey(0);
	}
	
	cv::imwrite(imageName, heatMap);
}

void examineFreeSpaceEvidence(const std::vector<Eigen::Vector3f> & points, 
	const float* pointMin, const float * pointMax,
	const std::string & outputFolder, const std::string & scanNumber){

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
				for (int a = 0; a < floor(length-1); ++a) {
			
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
	
	collapseFreeSpaceEvidence(numTimesSeen4C, numZ, numY, numX,
	 outputFolder, scanNumber);
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
	const int numZ, const int numY, const int numX,
	const std::string & outputFolder, const std::string & scanNumber){

	Eigen::MatrixXd collapsedMean (numY, numX);

	for (int i = 0; i < numX; ++i) {
		for (int j = 0; j < numY; ++j) {
			double mean = 0;
			int count = 0;
			for (int k = 0; k < numZ; ++k) {
				if(numTimesSeen[i](k,j) != 0) {
					mean += static_cast<double>(numTimesSeen[i](k,j));
					count++;
				}
			}
			mean = mean/numZ;
			collapsedMean(j,i) = count;
		}
	}
	const std::string imageName = outputFolder + "DUC_freeSpace_" + scanNumber + ".png";
	displayCollapsed(collapsedMean, numX, numY, imageName);
	
	
}

void displayCollapsed(const Eigen::MatrixXd & numTimesSeen, 
	const int numX, const int numY,
	const std::string & imageName){
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

	cv::Mat collapsedMap (numY, numX, CV_8UC1, cv::Scalar::all(255));

	for (int j = 0; j < collapsedMap.rows; ++j)
	{
		uchar * dst = collapsedMap.ptr<uchar>(j);
		
		for (int i = 0; i < collapsedMap.cols; ++i)
		{
			if(numTimesSeen(j,i) != 0){
				const int gray = cv::saturate_cast<uchar>(255.0 *((numTimesSeen(j,i)
					  - average) / (1.0*sigma) + 1.0));
				dst[i] = 255 - gray;
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
				dst[i*3] = blue;
				dst[i*3 +1] = green;
				dst[i*3 + 2] = red;*/
			}
		} 
	}

	if(FLAGS_preview)
	{
		cv::imshow("Preview", collapsedMap);
		cv::waitKey(0); 
	}
	

	cv::imwrite(imageName, collapsedMap);
}

