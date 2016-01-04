/*Scanner units are proabaly in meters */
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/StdVector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <string>
#include <dirent.h>
#include <vector>
#include <fstream>
#include <math.h>
#include <time.h>
#include <gflags/gflags.h>

/*#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>*/

using namespace cv;
using namespace std;
using namespace Eigen;
// using namespace pcl;



void examinePointEvidence(const vector<Vector3f> &, const float *, const float *,
	const string &, const string &);
void createBoundingBox(float *, float *, const vector<Vector3f> &);
void examineFreeSpaceEvidence(const vector<Vector3f> &, const float*, const float *,
	const string &, const string &);
void showSlices(const MatrixXi & numTimesSeen,
	const int numZ, const int numY, const int numX, const string &);
void collapseFreeSpaceEvidence(const vector<MatrixXi> &, const int, const int,
	const int, const string &, const string &);
void displayCollapsed(const MatrixXd &, const int, const int, const string &);
void displayPointEvenidence(const MatrixXf &, const string &, const int);

void analyzeScan(const string &, const string &);


static Mat heatMap;




DEFINE_bool(pe, false, "Tells the program to only examine point evidence");
DEFINE_bool(fe, false, "Tells the program to only examine free space evidence");
DEFINE_bool(quiteMode, true, "Turns of all extrenous statements");
DEFINE_bool(preview, false, "Turns on previews of the output");
DEFINE_bool(redo, false, "Recreates the density map even if it already exists");
DEFINE_string(inFolder, "/home/erik/Projects/3DscanData/DUC/Floor1/binaryFiles/",
	"Path to binary files");
DEFINE_string(outFolder, "/home/erik/Projects/3DscanData/DUC/Floor1/densityMaps/",
	"Path to output folder");
DEFINE_string(zerosFolder, "/home/erik/Projects/3DscanData/DUC/Floor1/densityMaps/zeros/",
	"Path to folder where the pixel cordinates of (0,0) will be written to");
DEFINE_string(voxelFolder, "/home/erik/Projects/3DscanData/DUC/Floor1/voxelGrids",
	"Path to the folder where the voxelGrids are saved to.");
DEFINE_double(scale, 73.5, "scale used to size the density maps");
DEFINE_int32(startIndex, 0, "Number to start with");
DEFINE_int32(numScans, -1, "Number to process, -1 or default implies all scans");


int main(int argc, char *argv[])
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	
	cvNamedWindow("Preview", WINDOW_NORMAL);

	vector<string> binaryNames;
	
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (FLAGS_inFolder.data())) != NULL) {
	  /* Add all the files and directories to a vector */
	  while ((ent = readdir (dir)) != NULL) {
	  	string fileName = ent->d_name;
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

	sort(binaryNames.begin(), binaryNames.end());
	if(FLAGS_numScans == -1)
		FLAGS_numScans = binaryNames.size() - FLAGS_startIndex;

	for(int i = FLAGS_startIndex; i < FLAGS_startIndex + FLAGS_numScans; ++i){
		const string binaryFilePath = FLAGS_inFolder + binaryNames[i];
		analyzeScan(binaryFilePath, FLAGS_outFolder);
	}
	
	
	
	cout << "Scan Density Done!" << endl;

	return 0;
}

void analyzeScan(const string & fileName, const string & outputFolder){
	const string scanNumber = fileName.substr(fileName.find(".") - 3, 3);
	if(!FLAGS_quiteMode)
		cout << scanNumber << endl;

	if(FLAGS_pe && !FLAGS_redo) {
		const string imageName = outputFolder + "DUC_point_" + scanNumber + ".png";
		cv::Mat img = imread(imageName);
		if(img.data)
			return;
	}
	if(FLAGS_fe && !FLAGS_redo) {
		const string imageName = outputFolder + "DUC_freeSpace_" + scanNumber + ".png";
		cv::Mat img = imread(imageName);
		if(img.data)
			return;
	}

	if(!FLAGS_pe && !FLAGS_fe && !FLAGS_redo) {
		const string imageName = outputFolder + "DUC_point_" + scanNumber + ".png";
		cv::Mat img = imread(imageName);
		const string imageName2 = outputFolder + "DUC_freeSpace_" + scanNumber + ".png";
		cv::Mat img2 = imread(imageName2);
		if(img.data && img2.data)
			return;
	}

  ifstream scanFile (fileName, ios::in | ios::binary);

  int columns, rows;
 	scanFile.read(reinterpret_cast<char *> (& columns), sizeof(int));
 	scanFile.read(reinterpret_cast<char *> (& rows), sizeof(int));

  
  float pointMax [3], pointMin[3];
  pointMax[0] = pointMax[1] = pointMax[2] 
  	= pointMin[0] = pointMin[1] = pointMin[2] = 0;
  
  vector<Vector3f> points;
  int numCenter = 0;
  for (int k = 0; k < columns * rows; ++k) {
    Vector3f point;
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
}



void createBoundingBox(float * pointMin, float * pointMax,
	const vector<Vector3f> & points){
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
		cout << "averageX: " << averageX << endl;
		cout << "averageY: " << averageY << endl;
		cout << "averageZ: " << averageZ << endl;
		cout << "sigmaX: " << sigmaX << endl;
		cout << "sigmaY: " << sigmaY << endl;
		cout << "sigmaZ: " << sigmaZ << endl;
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

void examinePointEvidence(const vector<Vector3f> & points,
	const float* pointMin, const float * pointMax, 
	const string & outputFolder, const string & scanNumber){
	const int numZ = 100;
	const float zScale = (float)numZ/(pointMax[2] - pointMin[2]);

	const int numCols = FLAGS_scale * (pointMax[0] - pointMin[0]);
	const int numRows = FLAGS_scale * (pointMax[1] - pointMin[1]);

	heatMap = Mat (numRows, numCols, CV_8UC1, Scalar::all(255));

	vector<MatrixXi> numTimesSeen3D (heatMap.rows, MatrixXi::Zero(heatMap.cols, numZ));

	Vector2i zeroZero (0,0);
	zeroZero[0] -= pointMin[0];
	zeroZero[1] -= pointMin[1];
	zeroZero *= FLAGS_scale;
	zeroZero[0] = static_cast<int>(zeroZero[0]);
	zeroZero[1] = static_cast<int>(zeroZero[1]);
	const string zeroName = FLAGS_zerosFolder + "DUC_point_" + scanNumber + ".dat";
	ofstream out (zeroName, ios::out | ios::binary);
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

	MatrixXf total = MatrixXf::Zero (heatMap.rows, heatMap.cols);
	for(int i = 0; i < heatMap.rows; ++i)
		for (int j = 0; j < heatMap.cols; ++j)
			for (int k = 0; k < numZ; ++k)
				if(numTimesSeen3D[i](j,k))
					++total(i,j);

	const string imageName = outputFolder + "DUC_point_" + scanNumber + ".png";
	displayPointEvenidence(total, imageName, 2.0);

}

void displayPointEvenidence(const MatrixXf & numTimesSeen, 
	const string & imageName,
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
			minV = min(minV, *(dataPtr+i));
			maxV = max(maxV, *(dataPtr + i));
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
		cout << "Average     Sigma" << endl << average << "     " << sigma << endl;
		cout << "Max     Min" << endl << maxV << "      " << minV << endl;
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
		imshow("Preview", heatMap);
		waitKey(0);
	}
	
	imwrite(imageName, heatMap);
}

void examineFreeSpaceEvidence(const vector<Vector3f> & points, 
	const float* pointMin, const float * pointMax,
	const string & outputFolder, const string & scanNumber){

	const float numZSimga = 2;
	const int numX = FLAGS_scale * (pointMax[0] - pointMin[0]);
	const int numY = FLAGS_scale * (pointMax[1] - pointMin[1]);
	const int numZ = 100;

	const float zScale = (float)numZ/(pointMax[2] - pointMin[2]);
	float cameraCenter [3];
	cameraCenter[0] = -1*pointMin[0];
	cameraCenter[1] = -1*pointMin[1];
	cameraCenter[2] = -1*pointMin[2];

	vector<MatrixXi> pointsPerVoxel (numZ, MatrixXi::Zero(numY, numX));
	vector<MatrixXi> numTimesSeen4C (numX, MatrixXi::Zero(numZ, numY));
	vector<MatrixXi> numTimesSeen (numZ, MatrixXi::Zero(numY, numX));

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

	std::string outName = FLAGS_voxelFolder + 
		"DUC_point_" + scanNumber + ".dat";
	saveVoxelGrid(pointsPerVoxel, outName);


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

					numTimesSeen[voxelHit[2]](voxelHit[1], voxelHit[0])
						+= pointsPerVoxel[k](j,i);

				}
			}
		}
	}
	outName = FLAGS_voxelFolder + "DUC_freeSpace_" + scanNumber + ".dat";
	saveVoxelGrid(numTimesSeen, outFolder);
	
	collapseFreeSpaceEvidence(numTimesSeen4C, numZ, numY, numX,
	 outputFolder, scanNumber);
}

void showSlices(const MatrixXi & currentSlice,
	const int numZ, const int numY, const int numX, const string & scanNumber){


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

	Mat sliceMap (numY, numX, CV_8UC3, Scalar::all(255));

	for (int j = 0; j < sliceMap.rows; ++j) {
		uchar * dst = sliceMap.ptr<uchar>(j);
		
		for (int i = 0; i < sliceMap.cols; ++i) {
			if(currentSlice(j,i) != 0){
				const int gray = max(0, min(255,
					 static_cast<int>(255.0 * (currentSlice(j,i)
					  - average) / ((3 * sigma) + 1.0) / 2.0)));
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
	/*const string imageName = FLAGS_outFolder + "DUC_freeSpace_" + scanNumber + ".png";
	imwrite(imageName, sliceMap);*/

	imshow("Preview", sliceMap);
	waitKey(0); 


}


void collapseFreeSpaceEvidence(const vector<MatrixXi> & numTimesSeen,
	const int numZ, const int numY, const int numX,
	const string & outputFolder, const string & scanNumber){

	MatrixXd collapsedMean (numY, numX);

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
	const string imageName = outputFolder + "DUC_freeSpace_" + scanNumber + ".png";
	displayCollapsed(collapsedMean, numX, numY, imageName);
	
	
}

void displayCollapsed(const MatrixXd & numTimesSeen, 
	const int numX, const int numY,
	const string & imageName){
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

	Mat collapsedMap (numY, numX, CV_8UC1, Scalar::all(255));

	for (int j = 0; j < collapsedMap.rows; ++j)
	{
		uchar * dst = collapsedMap.ptr<uchar>(j);
		
		for (int i = 0; i < collapsedMap.cols; ++i)
		{
			if(numTimesSeen(j,i) != 0){
				const int gray = max(0, min(255,
					 static_cast<int>(255.0 *((numTimesSeen(j,i)
					  - average) / (1.0*sigma) + 1.0))));
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
		imshow("Preview", collapsedMap);
		waitKey(0); 
	}
	

	imwrite(imageName, collapsedMap);
}

void saveVoxelGrid(std::vector<Eigen::MatrixXd> & grid,
	std::string & outName) {
	std::ofstream out (outName, ios::out | ios::binary);

	int x, y, z;
	z = grid.size();
	y = grid[0].rows();
	x = grid[0].cols();

	out.write(reinterpret_cast<const char *>(& z), sizeof(z));
	out.write(reinterpret_cast<const char *>(& y), sizeof(y));
	out.write(reinterpret_cast<const char *>(& x), sizeof(x));

	double average = 0.0;
	int count = 0;

	for(int i = 0; i < z; ++i) {
		const double * dataPtr = grid[i].data();
		for(int j = 0; j < grid[i].outerSize()) {
			const double value = *(dataPtr + i);
			if(value) {
				average += value;
				++count;
			}
		}
	}
	average /= count;
	double simga = 0.0;
	for(int i = 0; i < z; ++i) {
		const double * dataPtr = grid[i].data();
		for(int j = 0; j < grid[i].outerSize()) {
			const double value = *(dataPtr + i);
			if(value)
				sigma += (value - average)*(value - average);
		}
	}
	sigma /= count - 1;
	sigma = sqrt(sigma);

	for(int i = 0; i < z; ++i) {
		const double * valuePtr = grid[i].data();
		for(int j = 0; j < grid[i].size(); ++j) {
			const double normalized = (*(valuePtr + i) - average)/(1.0*sigma);
			const char tmp = normalized > -1.0 ? 1 : 0;
			out.write(&tmp, sizeof(tmp));
		}
	}
	out.close();
}