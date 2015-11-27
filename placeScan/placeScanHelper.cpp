#include "placeScan_placeScanHelper.h"

#include <dirent.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <fstream>


DEFINE_bool(visulization, false, 
	"Turns on all visualization options that do not impact performance");
DEFINE_bool(previewIn, false, "Turns on a preview of the scan before it is placed");
DEFINE_bool(previewOut, true, "Shows a preview of the scans placement before saving");
DEFINE_bool(replace, false, "Forces the program to redo the placement of all scans given");
DEFINE_bool(quiteMode, false, "Very minimal status reports");
DEFINE_bool(tinyPreviewIn, false, "Shows the input scan before any processing");
DEFINE_bool(save, true, "Saves the placements to file");
DEFINE_bool(debugMode, false, 
	"Turns save off, turns replace on, and forces the program to display the correct placement according to the placement files specified by the preDone flag");
DEFINE_string(floorPlan, "/home/erik/Projects/3DscanData/DUC/floorPlans/DUC-floor-1_cropped.png", 
	"Path to the floor plan that the scan should be placed on");
DEFINE_string(dmFolder, "/home/erik/Projects/3DscanData/DUC/Floor1/densityMaps/",
	"Path to folder containing densityMaps");
DEFINE_string(rotFolder, "/home/erik/Projects/3DscanData/DUC/Floor1/densityMaps/rotations/",
	"Path to folder containing the dominate direction rotations");
DEFINE_string(preDone, "/home/erik/Projects/3DscanData/DUC/Floor1/placementOptions/",
	"Path to folder containing previous placements of a scan");
DEFINE_string(zerosFolder, "/home/erik/Projects/3DscanData/DUC/Floor1/densityMaps/zeros/",
	"Path to folder where the pixel coordinates of (0,0) are");
DEFINE_int32(startIndex, 0, "Scan number to start with");
DEFINE_int32(numScans, -1, 
	"Number of scans to place, default or -1 will cause all scans in the folder to placed");
DEFINE_int32(numLevels, 5, "Number of levels in the pyramid");
DEFINE_int32(metricNumber, 3, "Which metric version the algorithm uses for placement");


std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > zeroZero;
std::vector<int> globalMins;
cv::Mat fpColor, floorPlan;
std::vector<Eigen::Vector3i> truePlacement;





void place::parseFolders(std::vector<std::string> & pointFileNames, 
	std::vector<std::string> & rotationFileNames,
	std::vector<std::string> & zerosFileNames,
	std::vector<std::string> * freeFileNames){
	
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (FLAGS_dmFolder.data())) != NULL) {
	  while ((ent = readdir (dir)) != NULL) {
	  	std::string fileName = ent->d_name;
	  	if(fileName != ".." && fileName != "." 
	  		&& fileName.find("point") != std::string::npos){
	  		pointFileNames.push_back(fileName);
	  	} else if (fileName != ".." && fileName != "." 
	  		&& fileName.find("freeSpace") != std::string::npos &&
	  		freeFileNames) {
	  		freeFileNames->push_back(fileName);
	  	}
	  }
	  closedir (dir);
	}  else {
	  /* could not open directory */
	  perror ("");
	  exit(-1);
	}

	if ((dir = opendir (FLAGS_rotFolder.data())) != NULL) {
	  while ((ent = readdir (dir)) != NULL) {
	  	std::string fileName = ent->d_name;
	  	if(fileName != ".." && fileName != "." ){
	  		rotationFileNames.push_back(fileName);
	  	}
	  }
	  closedir (dir);
	}  else {
	  /* could not open directory */
	  perror ("");
	  exit(-1);
	}

	if ((dir = opendir (FLAGS_zerosFolder.data())) != NULL) {
	  while ((ent = readdir (dir)) != NULL) {
	  	std::string fileName = ent->d_name;
	  	if(fileName != ".." && fileName != "." ){
	  		zerosFileNames.push_back(fileName);
	  	}
	  }
	  closedir (dir);
	}  else {
	  /* could not open directory */
	  perror ("");
	  exit(-1);
	}

	if(pointFileNames.size() != rotationFileNames.size()){
		perror("Not the same number of scans as rotations!");
		exit(-1);
	}


	sort(rotationFileNames.begin(), rotationFileNames.end());
	sort(pointFileNames.begin(), pointFileNames.end());
	sort(zerosFileNames.begin(), zerosFileNames.end());
	if(freeFileNames)
		sort(freeFileNames->begin(), freeFileNames->end());
}


void place::loadInScans(const std::string & scanName, const std::string & rotationFile, 
	 const std::string * zerosFile, std::vector<cv::Mat> & rotatedScans){
	std::ifstream binaryReader (rotationFile, std::ios::in | std::ios::binary);
	std::vector<Eigen::Matrix3d> R (4);
	for (int i = 0; i < R.size(); ++i) {
		binaryReader.read(reinterpret_cast<char *>(&R[i]),
			sizeof(Eigen::Matrix3d));
	}
	binaryReader.close();

	if(zerosFile) {
		binaryReader.open(*zerosFile, std::ios::in | std::ios::binary);
		zeroZero.clear();
		zeroZero.resize(4);
		for (int i = 0; i < 4; ++i) {
			binaryReader.read(reinterpret_cast<char *>(&zeroZero[i]),
				sizeof(Eigen::Vector2d));
			binaryReader.seekg(0);
		}
		binaryReader.close();
	}
	
	cv::Mat scan = cv::imread(scanName, 0);
	
	if(!scan.data){
		std::cout << "Error reading scan" << std::endl;
		exit(1);
	}
	if(FLAGS_tinyPreviewIn || FLAGS_visulization) {
		cvNamedWindow("Preview", CV_WINDOW_NORMAL);
		cv::imshow("Preview", scan);
		cv::waitKey(0);
	}

	const int maxDimension = std::max(1.5*scan.rows, 1.5*scan.cols);
	const int colOffset = (maxDimension - scan.cols)/2;
	const int rowOffset = (maxDimension - scan.rows)/2;
	if(zerosFile) {
		for (int i = 0; i < 4; ++i) {
			zeroZero[i][0] += colOffset;
			zeroZero[i][0] += rowOffset;
		}
	}

	cv::Mat widenedScan (maxDimension, maxDimension, CV_8UC1, cv::Scalar::all(255));
	for (int i = 0; i < scan.rows; ++i) {
		uchar * src = scan.ptr<uchar>(i);
		uchar * dst = widenedScan.ptr<uchar>(i + rowOffset);
		for (int j = 0; j < scan.cols; ++j) {
			dst[j+colOffset] = src[j];
		}
	}

	if(FLAGS_tinyPreviewIn || FLAGS_visulization) {
		cvNamedWindow("Preview", CV_WINDOW_NORMAL);
		cv::imshow("Preview", widenedScan);
		cv::waitKey(0);
	}

	const Eigen::Vector3d center (widenedScan.cols/2.0, widenedScan.rows/2.0, 0.0);
	int i = 0;
	for(auto & rot : R) {
		cv::Mat rScan (widenedScan.rows, widenedScan.cols, CV_8UC1, cv::Scalar::all(255));
		for (int i = 0; i < widenedScan.rows; ++i) {
			uchar * dst = rScan.ptr<uchar>(i);
			for (int j = 0; j < widenedScan.cols; ++j)
			{
				const Eigen::Vector3d pixel (j, i, 0.0);
				const Eigen::Vector3d src = rot*(pixel-center) + center;
				if(src[0] < 0 || src[0] >= widenedScan.cols )
					continue;
				if(src[1] < 0 || src[1] >= widenedScan.rows)
					continue;
				dst[j] = widenedScan.at<uchar>(src[0], src[1]);
			}
		}
		rotatedScans.push_back(rScan);

		if(zerosFile) {
			Eigen::Vector3d tmp (zeroZero[i][0], zeroZero[i][1], 0.0);
			tmp = rot*(tmp - center) + center;
			zeroZero[i][0] = tmp[0];
			zeroZero[i][0] = tmp[1];
			i++;
		}
	}

	if(FLAGS_visulization || FLAGS_previewIn) {
		for(auto & scan : rotatedScans){
			cvNamedWindow("Preview", CV_WINDOW_NORMAL);
			cv::imshow("Preview", scan);
			cv::waitKey(0);
		}
	}
}

void place::loadInScansAndMasks(const std::string & scanName, 
	const std::string & rotationFile, const std::string & zerosFile, 
	const std::string & maskName, std::vector<cv::Mat> & rotatedScans,
	std::vector<cv::Mat> & masks){
	place::loadInScans(scanName, rotationFile, &zerosFile, rotatedScans);

	place::loadInScans(maskName, rotationFile, NULL, masks);
}

void place::trimScans(const std::vector<cv::Mat> & toTrim, 
	std::vector<cv::Mat> & trimmedScans){
	int k = 0;
	for(auto scan : toTrim){
		int minRow = scan.rows;
		int minCol = scan.cols;
		int maxRow = 0;
		int maxCol = 0;

		for (int i = 0; i < scan.rows; ++i) {
			const uchar * src = scan.ptr<uchar>(i);
			for (int j = 0; j < scan.cols; ++j)
			{
				if(src[j]!=255){
					minRow = std::min(i, minRow);
					minCol = std::min(j, minCol);
					maxRow = std::max(i, maxRow);
					maxCol = std::max(j, maxCol);
				}
			}
		}

		cv::Mat trimmedScan (maxRow - minRow + 1, maxCol - minCol + 1, CV_8UC1);
		for (int i = minRow; i < maxRow + 1; ++i) {
			const uchar * src = scan.ptr<uchar>(i);
			uchar * dst = trimmedScan.ptr<uchar>(i-minRow);
			for (int j = minCol; j < maxCol + 1; ++j)
			{
				dst[j-minCol] = src[j];
			}
		}

		trimmedScans.push_back(trimmedScan);
		zeroZero[k][0] -= minCol;
		zeroZero[k][1] -= minRow;
		++k;
	
	}
}

void place::savePlacement(const std::vector<posInfo> & scores, const std::vector<int> & localMinima, 
	const std::string & outName){
	std::ofstream out (outName, std::ios::out);
	std::ofstream outB (outName.substr(0, outName.find(".")) + ".dat", std::ios::out | std::ios::binary);
	out << "Score x y rotation" << std::endl;
	const int numLoc = localMinima.size();
	const int numGlob = globalMins.size();
	outB.write(reinterpret_cast<const char *>(&numLoc), sizeof(numLoc));
	outB.write(reinterpret_cast<const char *>(&numGlob), sizeof(numGlob));
	out << "localMinima: " << std::endl;
	for(auto index : localMinima){
		posInfo minScore = scores[index];
		minScore.x += zeroZero[minScore.rotation][0];
		minScore.y += zeroZero[minScore.rotation][1];
		out << minScore.score << " " << minScore.x  << " "
			<< minScore.y << " " << minScore.rotation << std::endl;
		
		outB.write(reinterpret_cast<const char *> (&minScore), sizeof(minScore));
	}
	out << std::endl << "globalMins: " << std::endl;
	for(auto index : globalMins){
		posInfo minScore = scores[index];
		minScore.x += zeroZero[minScore.rotation][0];
		minScore.y += zeroZero[minScore.rotation][1];
		out << minScore.score << " " << minScore.x  << " "
			<< minScore.y  << " " << minScore.rotation << std::endl;
		
		outB.write(reinterpret_cast<const char *> (&minScore), sizeof(minScore));
	}
	out.close();
	outB.close();
}

bool place::reshowPlacement(const std::string & scanName, const std::string & rotationFile,
	const std::string & zerosFile){
	const std::string placementName = FLAGS_preDone + scanName.substr(scanName.find("_")-3, 3) 
	+ "_placement_" + scanName.substr(scanName.find(".")-3, 3) + ".dat";
	if(!FLAGS_quiteMode)
		std::cout << placementName << std::endl;

	std::ifstream in (placementName, std::ios::in | std::ios::binary);
	if(!in.is_open())
		return false;
	if(!FLAGS_previewOut)
		return true;

	
	std::vector<cv::Mat> rotatedScans, toTrim;
	place::loadInScans(scanName, rotationFile, &zerosFile, toTrim);
	place::trimScans(toTrim, rotatedScans);

	int numLoc, numGlob;
	in.read(reinterpret_cast<char *>(&numLoc), sizeof(numLoc));
	in.read(reinterpret_cast<char *>(&numGlob), sizeof(numGlob));

	cvNamedWindow("Preview", CV_WINDOW_NORMAL);


	if(!FLAGS_quiteMode)
		std::cout << "Showing localMinima, " << numLoc << std::endl;
	std::vector<posInfo> scores;
	for (int i = 0; i < numLoc; ++i) {
		posInfo minScore;
		in.read(reinterpret_cast<char *>(&minScore), sizeof(minScore));
		minScore.x -= zeroZero[minScore.rotation][0];
		minScore.y -= zeroZero[minScore.rotation][1];
		
		cv::Mat bestScan = rotatedScans[minScore.rotation];

		const int xOffset = minScore.x;
		const int yOffset = minScore.y;
		cv::Mat output (fpColor.rows, fpColor.cols, CV_8UC3, cv::Scalar::all(255));
		fpColor.copyTo(output);
		
		for (int i = 0; i < bestScan.rows; ++i) {
			uchar * src = bestScan.ptr<uchar>(i);
			uchar * dst = output.ptr<uchar>(i + yOffset);
			for (int j = 0; j < bestScan.cols; ++j)
			{
				if(src[j]!=255){
					dst[j*3 + xOffset*3] = 0;
					dst[j*3 + xOffset*3 + 1] = 0;
					dst[j*3 + xOffset*3 + 2] = 255 - src[j];
				}
			}
		}
		if(!FLAGS_quiteMode) {
			std::cout << minScore.score <<"      " << minScore.x << "      " <<minScore.y << std::endl;
			std::cout << minScore.scanFP << "      " << minScore.fpScan << std::endl;
			std::cout << minScore.scanPixels << "    " << minScore.fpPixels  << std::endl;
			std::cout << "% of scan unexplained: " << minScore.scanFP/minScore.scanPixels << std::endl << std::endl;
		}
		cv::imwrite("Out.png", output);
		cv::imshow("Preview", output);
		cv::waitKey(0);
	}

	if(!FLAGS_quiteMode)
		std::cout << "Showing globalMins, " << numGlob << std::endl;
	for (int i = 0; i < numGlob; ++i) {
		posInfo minScore;
		in.read(reinterpret_cast<char *>(&minScore), sizeof(minScore));
		minScore.x -= zeroZero[minScore.rotation][0];
		minScore.y -= zeroZero[minScore.rotation][1];
		
		cv::Mat bestScan = rotatedScans[minScore.rotation];

		const int xOffset = minScore.x;
		const int yOffset = minScore.y;
		cv::Mat output (fpColor.rows, fpColor.cols, CV_8UC3, cv::Scalar::all(255));
		fpColor.copyTo(output);
		
		for (int i = 0; i < bestScan.rows; ++i) {
			uchar * src = bestScan.ptr<uchar>(i);
			uchar * dst = output.ptr<uchar>(i + yOffset);
			for (int j = 0; j < bestScan.cols; ++j)
			{
				if(src[j]!=255){
					dst[j*3 + xOffset*3] = 0;
					dst[j*3 + xOffset*3 + 1] = 0;
					dst[j*3 + xOffset*3 + 2] = 255 - src[j];
				}
			}
		}
		if(!FLAGS_quiteMode) {
			std::cout << minScore.score <<"      " << minScore.x << "      " <<minScore.y << std::endl;
			std::cout << minScore.scanFP << "      " << minScore.fpScan << std::endl;
			std::cout << minScore.scanPixels << "    " << minScore.fpPixels  << std::endl;
			std::cout << "% of scan unexplained: " << minScore.scanFP/minScore.scanPixels << std::endl << std::endl;
		}

		cv::imwrite("Out.png", output);
		cv::imshow("Preview", output);
		cv::waitKey(0);
	}

	return true;
}

void place::displayOutput(const std::vector<Eigen::SparseMatrix<double> > & rSSparseTrimmed, 
	const std::vector<int> & localMinima, const std::vector<posInfo> & scores){
	if(!FLAGS_quiteMode) {
		std::cout << "Num localMinima: " << localMinima.size() << std::endl;
		std::cout << "Press a key to begin displaying placement options" << std::endl;
	}
	
	cvNamedWindow("Preview", CV_WINDOW_NORMAL);
	cv::imshow("Preview", fpColor);
	cv::waitKey(0);


	for(auto & index : localMinima){
		posInfo minScore = scores[index];
		const int xOffset = minScore.x;
		const int yOffset = minScore.y;
		const Eigen::SparseMatrix<double> & currentScan = rSSparseTrimmed[minScore.rotation]; 
		cv::Mat output (fpColor.rows, fpColor.cols, CV_8UC3, cv::Scalar::all(255));
		fpColor.copyTo(output);

		cv::Mat_<cv::Vec3b> _output = output;

		for (int i = 0; i < currentScan.outerSize(); ++i) {
			for(Eigen::SparseMatrix<double>::InnerIterator it(currentScan, i); it; ++it){
				_output(it.row() + yOffset, it.col() + xOffset)[0]=0;
				_output(it.row() + yOffset, it.col() + xOffset)[1]=0;
				_output(it.row() + yOffset, it.col() + xOffset)[2]=255;

			}
		}
		
		cv::imshow("Preview", output);
		if(!FLAGS_quiteMode)
			std::cout << minScore.score <<"      " << minScore.x << "      " <<minScore.y << std::endl;
			std::cout << minScore.scanFP << "      " << minScore.fpScan << std::endl;
			std::cout << minScore.scanPixels << "    " << minScore.fpPixels << std::endl << std::endl;
		cv::waitKey(0);
		
		
		~output;
	}


	if(!FLAGS_quiteMode)
		std::cout << "Num globalMins: " << globalMins.size() << std::endl;
	for(auto & globMin : globalMins) {
		posInfo minScore = scores[globMin];
		const int xOffset = minScore.x;
		const int yOffset = minScore.y;
		/*if(yOffset == 0)
			continue;*/
		const Eigen::SparseMatrix<double> & currentScan = rSSparseTrimmed[minScore.rotation]; 
		cv::Mat output (fpColor.rows, fpColor.cols, CV_8UC3, cv::Scalar::all(255));
		fpColor.copyTo(output);

		cv::Mat_<cv::Vec3b> _output = output;

		for (int i = 0; i < currentScan.outerSize(); ++i) {
			for(Eigen::SparseMatrix<double>::InnerIterator it(currentScan, i); it; ++it){
				_output(it.row() + yOffset, it.col() + xOffset)[0]=0;
				_output(it.row() + yOffset, it.col() + xOffset)[1]=0;
				_output(it.row() + yOffset, it.col() + xOffset)[2]=255;

			}
		}

		output = _output;
		cv::imshow("Preview", output);
		if(!FLAGS_quiteMode) {
			std::cout << minScore.score <<"      " << minScore.x << "      " <<minScore.y << std::endl;
			std::cout << minScore.scanFP << "      " << minScore.fpScan << std::endl;
			std::cout << minScore.scanPixels << "    " << minScore.fpPixels << std::endl << std::endl;
		}
		
		cv::waitKey(0);
		~output;
	}
}

void place::loadInTruePlacement(const std::string & scanName){
	const std::string placementName = FLAGS_preDone + scanName.substr(scanName.find("_")-3, 3) 
	+ "_placement_" + scanName.substr(scanName.find(".")-3, 3) + ".dat";
	std::ifstream in (placementName, std::ios::in | std::ios::binary);

	int numLoc, numGlob;
	in.read(reinterpret_cast<char *>(&numLoc), sizeof(numLoc));
	in.read(reinterpret_cast<char *>(&numGlob), sizeof(numGlob));

	std::vector<posInfo> tmp (numLoc + numGlob);
	for (int i = 0; i < numLoc + numGlob; ++i) {
		in.read(reinterpret_cast<char *>(&tmp[i]), sizeof(posInfo));
	}

	truePlacement.clear();
	for(auto & s : tmp){
		Eigen::Vector3i tmp2 (s.x - zeroZero[s.rotation][0], 
			s.y - zeroZero[s.rotation][1], s.rotation);
		truePlacement.push_back(tmp2);
	}
}

void place::displayTruePlacement(const std::vector<Eigen::SparseMatrix<double> > & rSSparseTrimmed,
	const std::vector<posInfo> & scores){

	std::vector<int> tmp;
	const int offset = scores.size() - truePlacement.size();
	for (int i = 0; i < truePlacement.size(); ++i) {
		tmp.push_back(offset + i);
	}

	std::cout << "displaying true placement" << std::endl;
	globalMins.clear();
	place::displayOutput(rSSparseTrimmed, tmp, scores);

}

void place::sparseToImage(const Eigen::SparseMatrix<double> & toImage,
	cv::Mat & imageOut){
	imageOut = cv::Mat(toImage.rows(), toImage.cols(), CV_8UC1, cv::Scalar::all(255));

	for (int i = 0; i < toImage.outerSize(); ++i) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(toImage, i); it; ++it) {
			imageOut.at<uchar>(it.row(), it.col()) = 255 - 255*it.value();
		}
	}
}

cv::Mat place::sparseToImage(const Eigen::SparseMatrix<double> & toImage){
	cv::Mat image (toImage.rows(), toImage.cols(), CV_8UC1, cv::Scalar::all(255));

	for (int i = 0; i < toImage.outerSize(); ++i) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(toImage, i); it; ++it) {
			image.at<uchar>(it.row(), it.col()) = 255 - 255*it.value();
		}
	}
	return image;
}

void place::scanToSparse(const cv::Mat & scan, 
	Eigen::SparseMatrix<double> & sparse) {
	std::vector<Eigen::Triplet<double> > tripletList;

	for (int i = 0; i < scan.rows; ++i) {
		const uchar * src = scan.ptr<uchar>(i);
		for (int j = 0; j < scan.cols; ++j) {
			if(src[j] == 255)
				continue;
			double confidence = 1.0 -(double)src[j]/255.0;
			tripletList.push_back(Eigen::Triplet<double> (i,j,confidence));
		}
	}
	sparse = Eigen::SparseMatrix<double>(scan.rows, scan.cols);
	sparse.setFromTriplets(tripletList.begin(), tripletList.end());
	sparse.makeCompressed();
	sparse.prune(1.0);
}

Eigen::SparseMatrix<double> place::scanToSparse(const cv::Mat & scan) {
	std::vector<Eigen::Triplet<double> > tripletList;

	for (int i = 0; i < scan.rows; ++i) {
		const uchar * src = scan.ptr<uchar>(i);
		for (int j = 0; j < scan.cols; ++j) {
			if(src[j] == 255)
				continue;
			double confidence = 1.0 -(double)src[j]/255.0;
			tripletList.push_back(Eigen::Triplet<double> (i,j,confidence));
		}
	}
	Eigen::SparseMatrix<double> sparseTmp (scan.rows, scan.cols);
	sparseTmp.setFromTriplets(tripletList.begin(), tripletList.end());
	sparseTmp.makeCompressed();
	sparseTmp.prune(1.0);

	return sparseTmp;
}

void place::displayMostConfidentScan(const std::string & imageName, 
	const std::string & rotationName, const std::string & zerosFile,
	posInfo& minScore){
	
	std::vector<cv::Mat> rotatedScans;
	loadInScans(imageName, rotationName, &zerosFile, rotatedScans);

	cv::Mat bestScan = rotatedScans[minScore.rotation];
	minScore.x -= zeroZero[minScore.rotation][0];
	minScore.y -= zeroZero[minScore.rotation][1];




	const int xOffset = minScore.x;
	const int yOffset = minScore.y;
	cv::Mat output (fpColor.rows, fpColor.cols, CV_8UC3, cv::Scalar::all(255));
	fpColor.copyTo(output);

	for (int i = 0; i < bestScan.rows; ++i) {
		uchar * src = bestScan.ptr<uchar>(i);
		uchar * dst = output.ptr<uchar>(i + yOffset);
		for (int j = 0; j < bestScan.cols; ++j) {
			if(src[j]!=255){
				dst[j*3 + xOffset*3] = 0;
				dst[j*3 + xOffset*3 + 1] = 0;
				dst[j*3 + xOffset*3 + 2] = 255 - src[j];
				
			}
		}
	}
	if(!FLAGS_quiteMode) {
		std::cout << minScore.score <<"      " << minScore.x << "      " <<minScore.y << std::endl;
		std::cout << minScore.scanFP << "      " << minScore.fpScan << std::endl;
		std::cout << minScore.scanPixels << "    " << minScore.fpPixels  << std::endl;
		std::cout << "% of scan unexplained: " << minScore.scanFP/minScore.scanPixels << std::endl << std::endl;
	}
	cvNamedWindow("Preview", CV_WINDOW_NORMAL);
	cv::imshow("Preview", output);
	cv::waitKey(0);
	
}

void place::displayScanAndMask(const std::vector<std::vector<Eigen::SparseMatrix<double> > > & rSSparsePyramidTrimmed,
	const std::vector<std::vector<Eigen::MatrixXb> > & eMaskPyramidTrimmedNS) {

	for(int i = 0; i < rSSparsePyramidTrimmed.size(); ++i) {
		for(int j = 0; j < rSSparsePyramidTrimmed[i].size(); ++j) {
			const Eigen::SparseMatrix<double> & currentScan = rSSparsePyramidTrimmed[i][j];
			const Eigen::MatrixXb & currentMask = eMaskPyramidTrimmedNS[i][j];
			cv::Mat out (currentScan.rows(), currentScan.cols(), CV_8UC3, cv::Scalar::all(255));

			for(int i = 0; i < out.rows; ++i) {
				uchar * dst = out.ptr<uchar>(i);
				for(int j = 0; j < out.cols; ++j) {
					if(currentMask(i,j) != 0) {
						dst[3*j] = 0;
						dst[3*j+1] = 0;
						dst[3*j+2] = 0;
					}
				}
			}

			cv::Mat_<cv::Vec3b> _out = out;
			for(int i = 0; i < currentScan.outerSize(); ++i) {
				for(Eigen::SparseMatrix<double>::InnerIterator it (currentScan, i); it; ++it) {
					if(it.value() > 0 && _out(it.row(), it.col())[0] == 0) {
						_out(it.row(), it.col())[0] = 0;
						_out(it.row(), it.col())[1] = 255;
						_out(it.row(), it.col())[2] = 0;
					} else if( it.value() > 0) {
						_out(it.row(), it.col())[0] = 0;
						_out(it.row(), it.col())[1] = 0;
						_out(it.row(), it.col())[2] = 255;
					}
				}
			}
			out = _out;
			cvNamedWindow("Preview", CV_WINDOW_NORMAL);
			cv::imshow("Preview", out);
			cv::waitKey(0);
		}
	}
}