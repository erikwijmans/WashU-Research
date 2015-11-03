#include "placeScan.h"


#include <iostream>
#include <dirent.h>
#include <fstream>
#include <math.h>
#include <time.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>



constexpr float fpScale = 86.0*3.28084;  /*pixels per meter */
/*Something like 75 is looking more likely however...Maybe the scanner
isn't in meters?*/
constexpr double PI = 3.14159265;
static cv::Mat floorPlan, fpColor;
static std::vector<Eigen::Vector3i> truePlacement;
static std::vector<int> globalMins;
static int levelNum = 10;
static std::vector<Eigen::Vector2d> zeroZero;


DEFINE_bool(visulization, false, 
	"Turns on all visualization options that do not impact performance");
DEFINE_bool(previewIn, false, "Turns on a preview of the scan before it is placed");
DEFINE_bool(previewOut, true, "Shows a preview of the scans placement before saving");
DEFINE_bool(replace, false, "Forces the program to redo the placement of all scans given");
DEFINE_bool(quiteMode, false, "Very minimal status reports");
DEFINE_bool(tinyPreviewIn, false, "Shows the input scan before any processing");
DEFINE_bool(save, true, "Saves the placements to file");
DEFINE_bool(debugMode, false, 
	"Turns save off, turns on replace, and forces the program to display the correct placement according to the placement files specified by the preDone flag");
DEFINE_string(floorPlan, "/home/erik/Projects/3DscanData/DUC/floorPlans/DUC-floor-1_cropped.png", 
	"Path to the floor plan that the scan should be placed on");
DEFINE_string(dmFolder, "/home/erik/Projects/3DscanData/DUC/densityMaps/",
	"Path to folder containing densityMaps");
DEFINE_string(rotFolder, "/home/erik/Projects/3DscanData/DUC/densityMaps/rotations/",
	"Path to folder containing the dominate direction rotations");
DEFINE_string(preDone, "/home/erik/Projects/3DscanData/DUC/placementOptions/",
	"Path to folder containing previous placements of a scan");
DEFINE_string(zerosFolder, "/home/erik/Projects/3DscanData/DUC/densityMaps/zeros/",
	"Path to folder where the pixel coordinates of (0,0) are");
DEFINE_int32(startIndex, 0, "Scan number to start with");
DEFINE_int32(numScans, -1, 
	"Number of scans to place, default or -1 will cause all scans in the folder to placed");
DEFINE_int32(numLevels, 5, "Number of levels in the pyramid");
DEFINE_int32(metricNumber, 3, "Which metric version the algorithm uses for placement");

int main(int argc, char *argv[])
{

	gflags::ParseCommandLineFlags(&argc, &argv, true);
	if(FLAGS_debugMode)
	{
		FLAGS_save = false;
		FLAGS_replace = true;
	}

	floorPlan = cv::imread(FLAGS_floorPlan, 0);
	if(!floorPlan.data){
		std::cout << "Error reading floorPlan" << std::endl;
		exit(1);
	}

	fpColor = cv::Mat (floorPlan.rows, floorPlan.cols, CV_8UC3, cv::Scalar::all(255));
	for (int i = 0; i < fpColor.rows; ++i)
	{
		uchar * dst = fpColor.ptr<uchar>(i);
		const uchar * src = floorPlan.ptr<uchar>(i);
		for (int j = 0; j < fpColor.cols; ++j)
		{
			if(src[j]!=255)
			{
				dst[j*3] = 128;
				dst[j*3+1] = 128;
				dst[j*3+2] = 128;
			}
		}
	}

	std::vector<std::string> pointFileNames;
	
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (FLAGS_dmFolder.data())) != NULL) {
	  /* Add all the files and directories to a std::vector */
	  while ((ent = readdir (dir)) != NULL) {
	  	std::string fileName = ent->d_name;
	  	if(fileName != ".." && fileName != "." 
	  		&& fileName.find("point") != std::string::npos){
	  		pointFileNames.push_back(fileName);
	  	}
	  }
	  closedir (dir);
	}  else {
	  /* could not open directory */
	  perror ("");
	  return EXIT_FAILURE;
	}

	std::vector<std::string> rotationFileNames;
	if ((dir = opendir (FLAGS_rotFolder.data())) != NULL) {
	  /* Add all the files and directories to a std::vector */
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
	  return EXIT_FAILURE;
	}

	std::vector<std::string> zerosFileNames;
	if ((dir = opendir (FLAGS_zerosFolder.data())) != NULL) {
	  /* Add all the files and directories to a std::vector */
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
	  return EXIT_FAILURE;
	}

	if(pointFileNames.size() != rotationFileNames.size()){
		std::cout << "Not the same number of scans as rotations!" << std::endl;
		exit(-1);
	}


	sort(rotationFileNames.begin(), rotationFileNames.end());
	sort(pointFileNames.begin(), pointFileNames.end());
	sort(zerosFileNames.begin(), zerosFileNames.end());
	
	if(FLAGS_numScans == -1)
		FLAGS_numScans = pointFileNames.size()-FLAGS_startIndex;
	
	for(int i = FLAGS_startIndex; i< FLAGS_startIndex + FLAGS_numScans; ++i){
		const std::string scanName = FLAGS_dmFolder + pointFileNames[i];
		const std::string rotationFile = FLAGS_rotFolder + rotationFileNames[i];
		const std::string zerosFile = FLAGS_zerosFolder + zerosFileNames[i];
		if(FLAGS_replace)
			place::analyzePlacement(floorPlan, scanName, rotationFile, zerosFile);
		else if(!place::reshowPlacement(scanName, rotationFile, zerosFile))
			place::analyzePlacement(floorPlan, scanName, rotationFile, zerosFile);
	}
	
	return 0;
}

void place::analyzePlacement(const cv::Mat & fp, const std::string & scanName, 
	const std::string & rotationFile, const std::string & zerosFile){
	if(!FLAGS_quiteMode)
		std::cout << rotationFile << std::endl;
	std::cout << scanName << std::endl;

	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(6,6));
	cv::Mat fpEroded (fp.rows, fp.cols, CV_8UC1);
	cv::erode(fp, fpEroded, element);
	
	std::vector<cv::Mat> rotatedScans;
	place::loadInScans(scanName, rotationFile, rotatedScans, zerosFile);
	std::vector<cv::Mat> erodedScans;

	for(auto scan : rotatedScans){
		cv::Mat dst (scan.rows, scan.cols, CV_8UC1);
		erode(scan, dst, element);
		erodedScans.push_back(dst);
	}

	/*for(int i = 0; i< 4; i++){
		cvNamedWindow("Preview1", cv::WINDOW_NORMAL);
		cv::imshow("Preview1", erodedScans[i]);
		cvNamedWindow("Preview2", cv::WINDOW_NORMAL);
		cv::imshow("Preview2", rotatedScans[i]);
		cv::waitKey(0);
	}

	
	cvNamedWindow("Preview", cv::WINDOW_NORMAL);
	cv::imshow("Preview", fpEroded);
	cv::waitKey(0);*/

	std::vector<Eigen::SparseMatrix<double> > erodedSSparse;
	std::vector<Eigen::SparseMatrix<double> > rSSparse;
	for (int i = 0; i < rotatedScans.size(); ++i)
	{
		Eigen::SparseMatrix<double> eroded, scan;
		place::scanToSparse(erodedScans[i], eroded);
		place::scanToSparse(rotatedScans[i], scan);
		erodedSSparse.push_back(eroded);
		rSSparse.push_back(scan);
	}
	

	Eigen::SparseMatrix<double> fpSparse, erodedFpSparse;
	place::scanToSparse(fp, fpSparse);
	place::scanToSparse(fpEroded, erodedFpSparse);


	std::vector<Eigen::SparseMatrix<double> > fpPyramid = {fpSparse};
	std::vector<std::vector<Eigen::SparseMatrix<double> > > rSSparsePyramid = {rSSparse};
	place::createPyramid(fpPyramid, rSSparsePyramid);
	
	
	
	std::vector<Eigen::SparseMatrix<double> > erodedFpPyramid = {erodedFpSparse};
	std::vector<std::vector<Eigen::SparseMatrix<double> > > erodedSparsePyramid = {erodedSSparse};
	place::createPyramid(erodedFpPyramid, erodedSparsePyramid);

	std::vector<std::vector<Eigen::SparseMatrix<double> > > erodedSparsePyramidTrimmed;
	std::vector<std::vector<Eigen::SparseMatrix<double> > > rSSparsePyramidTrimmed;
	place::trimScanPryamids(rSSparsePyramid, rSSparsePyramidTrimmed, 
		erodedSparsePyramid, erodedSparsePyramidTrimmed);

	std::vector<Eigen::Vector3i> pointsToAnalyze;
	for(int k = 0; k < rSSparse.size(); ++k)
	{
		const int xStop = fpPyramid[FLAGS_numLevels].cols() 
		- rSSparsePyramidTrimmed[FLAGS_numLevels][k].cols();
		const int yStop = fpPyramid[FLAGS_numLevels].rows()
		- rSSparsePyramidTrimmed[FLAGS_numLevels][k].rows();
		pointsToAnalyze.reserve(xStop*yStop*(k+1));
		for (int i = 0; i < xStop; ++i)
		{
			for (int j = 0; j < yStop; ++j)
			{
				pointsToAnalyze.push_back(Eigen::Vector3i (i,j,k));
			}
		}
	}

	if(FLAGS_debugMode)
		loadInTruePlacement(scanName);
	
	std::vector<posInfo> scores;
	std::vector<Eigen::MatrixXd> scoreMatricies;
	std::vector<int> localMinima;
	Eigen::Vector4i rows, cols;
	for (int k = FLAGS_numLevels; k >= 0; --k)
	{
		levelNum = k;
		if(k == 0)
		{
			for(auto & v : truePlacement){
				pointsToAnalyze.push_back(v);
			}
		}
		place::findPlacementPointBasedV2(fpPyramid[k], rSSparsePyramidTrimmed[k],
			erodedFpPyramid[k], erodedSparsePyramidTrimmed[k], 
			scores, pointsToAnalyze);
		/*for (int i = 0; i < 4; ++i)
		{
			rows[i] = fpPyramid[k].rows() - rSSparsePyramidTrimmed[k][i].rows();
			cols[i] = fpPyramid[k].cols() - rSSparsePyramidTrimmed[k][i].cols();
		}
		blurMinima(scoreMatricies, scores, rows, cols);*/
		if(k!=0)
		{
			place::findLocalMinima(scores, localMinima, 1.0);
			findGlobalMinima(scores, localMinima);
			findPointsToAnalyze(scores, localMinima, pointsToAnalyze);
		}
		
	}

	place::findLocalMinima(scores, localMinima, 2.5);
	findGlobalMinima(scores, localMinima);

	if(FLAGS_save)
	{
		const std::string placementName = FLAGS_preDone + scanName.substr(scanName.find("_")-3, 3) 
		+ "_placement_" + scanName.substr(scanName.find(".")-3, 3) + ".txt";
		
		savePlacement(scores, localMinima, placementName);
	}
	
	if(FLAGS_visulization || FLAGS_previewOut)
		place::displayOutput(rSSparsePyramidTrimmed[0], localMinima, scores);
	if(FLAGS_debugMode)
	{
		/*place::findPlacementPointBasedV2(fpPyramid[k], rSSparsePyramidTrimmed[k],
			erodedFpPyramid[k], erodedSparsePyramidTrimmed[k], 
			scores, truePlacement);*/
		displayTruePlacement(rSSparsePyramidTrimmed[0], scores);
	}
}

void place::loadInScans(const std::string & scanName, const std::string & rotationFile, 
	std::vector<cv::Mat> & rotatedScans, const std::string & zerosFile){
	std::ifstream binaryReader (rotationFile, std::ios::in | std::ios::binary);
	std::vector<Eigen::Matrix3d> R (4);
	for (int i = 0; i < R.size(); ++i)
	{
		binaryReader.read(reinterpret_cast<char *>(&R[i]),
			sizeof(Eigen::Matrix3d));
	}

	binaryReader.close();

	binaryReader.open(zerosFile, std::ios::in | std::ios::binary);
	zeroZero.clear();
	zeroZero.resize(4);
	for (int i = 0; i < 4; ++i)
	{
		binaryReader.read(reinterpret_cast<char *>(&zeroZero[i]),
			sizeof(Eigen::Vector2d));
		binaryReader.seekg(0);
	}
	binaryReader.close();

	cv::Mat scan = cv::imread(scanName, 0);
	
	if(!scan.data){
		std::cout << "Error reading scan" << std::endl;
		exit(1);
	}
	if(FLAGS_tinyPreviewIn)
	{
		cvNamedWindow("Preview", cv::WINDOW_NORMAL);
		cv::imshow("Preview", scan);
		cv::waitKey(0);
	}

	const int maxDimension = std::max(1.1*scan.rows, 1.1*scan.cols);
	const int colOffset = (maxDimension - scan.cols)/2;
	const int rowOffset = (maxDimension - scan.rows)/2;
	for (int i = 0; i < 4; ++i)
	{
		zeroZero[i][0] += colOffset;
		zeroZero[i][0] += rowOffset;
	}

	cv::Mat widenedScan (maxDimension, maxDimension, CV_8UC1, cv::Scalar::all(255));
	for (int i = 0; i < scan.rows; ++i)
	{
		uchar * src = scan.ptr<uchar>(i);
		uchar * dst = widenedScan.ptr<uchar>(i + rowOffset);
		for (int j = 0; j < scan.cols; ++j)
		{
			dst[j+colOffset] = src[j];
		}
	}

	const Eigen::Vector3d center (widenedScan.cols/2.0, widenedScan.rows/2.0, 0.0);

	int i = 0;
	for(auto & rot : R)
	{
		cv::Mat rScan (widenedScan.rows, widenedScan.cols, CV_8UC1, cv::Scalar::all(255));
		for (int i = 0; i < widenedScan.rows; ++i)
		{
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

		
		Eigen::Vector3d tmp (zeroZero[i][0], zeroZero[i][1], 0.0);
		tmp = rot*(tmp - center) + center;
		zeroZero[i][0] = tmp[0];
		zeroZero[i][0] = tmp[1];
		i++;
	}



	if(FLAGS_visulization || FLAGS_previewIn)
	{
		for(auto & scan : rotatedScans){
			cvNamedWindow("Preview", cv::WINDOW_NORMAL);
			cv::imshow("Preview", scan);
			cv::waitKey(0);
		}
	}
}

void place::scanToSparse(const cv::Mat & scan, Eigen::SparseMatrix<double> & sparse){
	std::vector<Eigen::Triplet<double> > tripletList;

	for (int i = 0; i < scan.rows; ++i)
	{
		const uchar * src = scan.ptr<uchar>(i);
		for (int j = 0; j < scan.cols; ++j)
		{
			if(src[j] == 255)
				continue;
			double confidence = 1.0 -(double)src[j]/255.0;
			tripletList.push_back(Eigen::Triplet<double> (i,j,confidence));
		}
	}
	sparse = Eigen::SparseMatrix<double> (scan.rows, scan.cols);
	sparse.setFromTriplets(tripletList.begin(), tripletList.end());
}


void place::displayOutput(const std::vector<Eigen::SparseMatrix<double> > & rSSparseTrimmed, 
	const std::vector<int> & localMinima, const std::vector<posInfo> & scores){
	if(!FLAGS_quiteMode)
	{
		std::cout << "Num localMinima: " << localMinima.size() << std::endl;
		std::cout << "Press a key to begin displaying placement options" << std::endl;
	}
	
	cvNamedWindow("Preview", cv::WINDOW_NORMAL);
	cv::imshow("Preview", fpColor);
	cv::waitKey(0);


	for(auto & index : localMinima){
		posInfo minScore = scores[index];
		const int xOffset = minScore.x;
		const int yOffset = minScore.y;
		const Eigen::SparseMatrix<double> & currentScan = rSSparseTrimmed[minScore.rotation]; 
		cv::Mat output (floorPlan.rows, floorPlan.cols, CV_8UC3, cv::Scalar::all(255));
		fpColor.copyTo(output);

		cv::Mat_<cv::Vec3b> _output = output;

		for (int i = 0; i < currentScan.outerSize(); ++i)
		{
			for(Eigen::SparseMatrix<double>::InnerIterator it(currentScan, i); it; ++it){
				_output(it.row() + yOffset, it.col() + xOffset)[0]=0;
				_output(it.row() + yOffset, it.col() + xOffset)[1]=0;
				_output(it.row() + yOffset, it.col() + xOffset)[2]=255;

			}
		}
		
		cv::imshow("Preview", output);
		if(!FLAGS_quiteMode)
			std::cout << minScore.score <<"      " << minScore.x << "      " <<minScore.y << std::endl;
			std::cout << minScore.yasu << "      " << minScore.yasu2 << std::endl;
			std::cout << minScore.scanPixels << "    " << minScore.fpPixels << std::endl << std::endl;
		cv::waitKey(0);
		
		
		~output;
	}


	if(!FLAGS_quiteMode)
		std::cout << "Num globalMins: " << globalMins.size() << std::endl;
	for(auto & globMin : globalMins)
	{
		posInfo minScore = scores[globMin];
		const int xOffset = minScore.x;
		const int yOffset = minScore.y;
		/*if(yOffset == 0)
			continue;*/
		const Eigen::SparseMatrix<double> & currentScan = rSSparseTrimmed[minScore.rotation]; 
		cv::Mat output (floorPlan.rows, floorPlan.cols, CV_8UC3, cv::Scalar::all(255));
		fpColor.copyTo(output);

		cv::Mat_<cv::Vec3b> _output = output;

		for (int i = 0; i < currentScan.outerSize(); ++i)
		{
			for(Eigen::SparseMatrix<double>::InnerIterator it(currentScan, i); it; ++it){
				_output(it.row() + yOffset, it.col() + xOffset)[0]=0;
				_output(it.row() + yOffset, it.col() + xOffset)[1]=0;
				_output(it.row() + yOffset, it.col() + xOffset)[2]=255;

			}
		}

		output = _output;
		cv::imshow("Preview", output);
		if(!FLAGS_quiteMode)
		{
			std::cout << minScore.score <<"      " << minScore.x << "      " <<minScore.y << std::endl;
			std::cout << minScore.yasu << "      " << minScore.yasu2 << std::endl;
			std::cout << minScore.scanPixels << "    " << minScore.fpPixels << std::endl << std::endl;
		}
		
		cv::waitKey(0);
		~output;
	}
}


void place::findLocalMinima(const std::vector<posInfo> & scores, std::vector<int> & localMinima, 
	const float bias){
	localMinima.clear();
	double averageScore = 0;
	for(auto & info : scores){
		averageScore += info.score;
	}
	averageScore /= scores.size();
	double sigScores = 0;
	for(auto & info : scores){
		sigScores += (info.score - averageScore)*(info.score - averageScore);
	}
	sigScores /= (scores.size() - 1);
	sigScores = sqrt(sigScores);
	if(!FLAGS_quiteMode)
	{
		std::cout << "Average         Sigma" << std::endl;
		std::cout << averageScore << "         " << sigScores << std::endl; 
	}
	

	const double cutOff = averageScore - bias*sigScores;
	int i;
	// #pragma omp parallel for shared(i, scores) reduction(merge: localMinima) reduction(merge: globalMins)
	for (i = 1; i < (scores.size() - 1); ++i)
	{
		double lHS = scores[i-1].score - scores[i].score;
		double rHS = scores[i].score - scores[i+1].score;
		if( lHS >= 0 && rHS <= 0 && scores[i].score < cutOff)
			localMinima.push_back(i);
	}
}

void place::createPyramid(std::vector<Eigen::SparseMatrix<double> > & fpPyramid,
	std::vector<std::vector<Eigen::SparseMatrix<double> > > & rSSparsePyramid){


	std::vector<Eigen::Triplet<double> > tripletList;
	
	for (int i = 0; i < FLAGS_numLevels; ++i)
	{
		Eigen::SparseMatrix<double> & currentFP = fpPyramid[i];
		Eigen::SparseMatrix<double> fpLevel (floor(currentFP.rows()/2), 
			floor(currentFP.cols()/2));

		Eigen::MatrixXd currentFPNS = Eigen::MatrixXd(currentFP);
		
		for (int j = 0; j < (currentFPNS.rows() -1); j+=2)
		{
			for (int k = 0; k < (currentFPNS.cols() -1); k+=2)
			{
				double maxV = std::max(currentFPNS(j,k),std::max(currentFPNS(j,k+1),
					std::max(currentFPNS(j+1,k), currentFPNS(j+1,k+1))));
				tripletList.push_back(Eigen::Triplet<double> (j/2, k/2, maxV));
			}
		}


		fpLevel.setFromTriplets(tripletList.begin(), tripletList.end());
		fpPyramid.push_back(fpLevel);
		tripletList.clear();

		std::vector<Eigen::SparseMatrix<double> > rsLevel;
		for(auto & scan : rSSparsePyramid[i]){
			Eigen::SparseMatrix<double> scanLevel (floor(scan.rows()/2), floor(scan.cols()/2));
			Eigen::MatrixXd scanNS = Eigen::MatrixXd(scan);

			for (int j = 0; j < (scan.rows()-1); j+=2)
			{
				for (int k = 0; k < (scan.cols()-1); k+=2)
				{
					double maxV = std::max(scanNS(j,k),std::max(scanNS(j,k+1),
					std::max(scanNS(j+1,k), scanNS(j+1,k+1))));
					tripletList.push_back(Eigen::Triplet<double> (j/2, k/2, maxV));
				}
			}

			scanLevel.setFromTriplets(tripletList.begin(), tripletList.end());
			rsLevel.push_back(scanLevel);
			tripletList.clear();
		}

		rSSparsePyramid.push_back(rsLevel);
	}

	if(FLAGS_visulization)
	{
		for(auto & level: fpPyramid){
			cv::Mat test (level.rows(), level.cols(), CV_8UC1, cv::Scalar::all(255));
			for (int k = 0; k < level.outerSize(); ++k)
			{
				for (Eigen::SparseMatrix<double>::InnerIterator it (level, k); it; ++it)
				{
					test.at<uchar>(it.row(), it.col()) = 255 - it.value()*255;
				}
			}
			cvNamedWindow("Preview", cv::WINDOW_NORMAL);
			cv::imshow("Preview", test);
			cv::waitKey(0);
		}
	}
}

void place::trimScanPryamids(const std::vector<std::vector<Eigen::SparseMatrix<double> > > & rSSparsePyramid,
	std::vector<std::vector<Eigen::SparseMatrix<double> > > & rSSparsePyramidTrimmed, 
	const std::vector<std::vector<Eigen::SparseMatrix<double> > > & erodedSparsePyramid,
	std::vector<std::vector<Eigen::SparseMatrix<double> > > & erodedSparsePyramidTrimmed){

	std::vector<Eigen::Triplet<double> > tripletList;
	int first4 = 0;
	for(int level = 0; level < rSSparsePyramid.size(); ++level){
		std::vector<Eigen::SparseMatrix<double> > levelTrimmed;
		std::vector<Eigen::SparseMatrix<double> > erodedLevelTrimed;
		for(int i = 0; i < rSSparsePyramid[level].size(); ++i){
			const Eigen::SparseMatrix<double> * scan = &rSSparsePyramid[level][i];
			const Eigen::SparseMatrix<double> * erodedScan = &erodedSparsePyramid[level][i];

			Eigen::SparseMatrix<double> scanThreshHolded (scan->rows(), scan->cols());
			Eigen::SparseMatrix<double> eScanThreshHolded (erodedScan->rows(), erodedScan->cols());

			for (int k = 0; k < scan->outerSize(); ++k)
			{
				for (Eigen::SparseMatrix<double>::InnerIterator it (*scan, k); it; ++it)
				{
					if(it.value() > 0.85){
						tripletList.push_back(Eigen::Triplet<double> (it.row(), it.col(), it.value()));
					}
				}
			}
			scanThreshHolded.setFromTriplets(tripletList.begin(), tripletList.end());
			tripletList.clear();

			for (int k = 0; k < erodedScan->outerSize(); ++k)
			{
				for (Eigen::SparseMatrix<double>::InnerIterator it (*erodedScan, k); it; ++it)
				{
					if(it.value() > 0.85){
						tripletList.push_back(Eigen::Triplet<double> (it.row(), it.col(), it.value()));
					}
				}
			}
			eScanThreshHolded.setFromTriplets(tripletList.begin(), tripletList.end());
			tripletList.clear();


			int minRow, minCol, maxRow, maxCol;
			maxRow = maxCol = 0;
			minRow = scanThreshHolded.rows();
			minCol = scanThreshHolded.cols();

			for (int k = 0; k < scanThreshHolded.outerSize(); ++k)
			{
				for (Eigen::SparseMatrix<double>::InnerIterator it (scanThreshHolded, k); it; ++it)
				{
					if(it.value() !=0){
						maxRow = std::max(maxRow, it.row());
						minRow = std::min(minRow, it.row());

						maxCol = std::max(maxCol, it.col());
						minCol = std::min(minCol, it.col());
					}
				}
			}

			Eigen::SparseMatrix<double> trimmed = scanThreshHolded.block(minRow, minCol, 
				maxRow - minRow, maxCol - minCol);
			Eigen::SparseMatrix<double> erodedTrimmed = eScanThreshHolded.block(minRow, minCol, 
				maxRow - minRow, maxCol - minCol);
		
			
			if(first4 < 4 )
			{
				zeroZero[first4][0] -= minCol;
				zeroZero[first4][1] -= minRow;
				++first4;
			}
			
			trimmed.makeCompressed();
			levelTrimmed.push_back(trimmed);
			erodedLevelTrimed.push_back(erodedTrimmed);
		}
		
		rSSparsePyramidTrimmed.push_back(levelTrimmed);
		erodedSparsePyramidTrimmed.push_back(erodedLevelTrimed);
	}

	if(FLAGS_visulization)
	{
		for(auto & level: rSSparsePyramidTrimmed){
			for(auto & scan : level){
				cv::Mat test (scan.rows(), scan.cols(), CV_8UC1, cv::Scalar::all(255));
				for (int k = 0; k < scan.outerSize(); ++k)
				{
					for (Eigen::SparseMatrix<double>::InnerIterator it (scan, k); it; ++it)
					{
						test.at<uchar>(it.row(), it.col()) = 255 - it.value()*255;
					}
				}
				cvNamedWindow("Preview", cv::WINDOW_NORMAL);
				cv::imshow("Preview", test);
				cv::waitKey(0);
			}
		}
	}
}

void place::findPlacementPointBasedV2(const Eigen::SparseMatrix<double> & fp, 
	const std::vector<Eigen::SparseMatrix<double> > & scans,const Eigen::SparseMatrix<double> & fpE, 
	const std::vector<Eigen::SparseMatrix<double> > & scansE, std::vector<posInfo> & scores,
	const std::vector<Eigen::Vector3i> & points){
	
	if(!FLAGS_quiteMode)
		std::cout << points.size() << std::endl;

	scores.clear();
	scores.reserve(points.size());


	#pragma omp parallel shared(fp, scans, points, scansE)
	{
		std::vector<posInfo> privateScores;
		#pragma omp for nowait schedule(static)
		for(int i = 0; i< points.size(); ++i)
		{
			
			const Eigen::Vector3i point = points[i];
			const int scanIndex = point[2];
			const int xStop = fp.cols() - scans[scanIndex].cols();
			const int yStop = fp.rows() - scans[scanIndex].rows();

			if(point[0] < 0 || point[0] >=xStop)
				continue;
			if(point[1] < 0 || point[1] >= yStop)
				continue;
			

			Eigen::SparseMatrix<double> currentFP = fp.block(point[1], point[0], 
				scans[scanIndex].rows(), scans[scanIndex].cols());
			currentFP.makeCompressed();

			if(currentFP.nonZeros() == 0)
				continue;

			double score;
			double diffEntropy = 0;
			double scanFPsetDiff = 0;
			double fpScanSetDiff = 0;

			/*if(point[0] == static_cast<int>(1187.0/pow(2, levelNum)) &&
				point[1] == static_cast<int>(183.0/pow(2,levelNum)))
			{
				cv::Mat test (diff.rows(), diff.cols(), CV_8UC1, cv::Scalar::all(128));
				for (int i = 0; i < diff.outerSize(); ++i)
				{
					for(Eigen::SparseMatrix<double>::InnerIterator it (diff, i); it; ++it)
					{
						
						test.at<uchar>(it.row(), it.col()) = std::max(0,std::min(255,
							static_cast<int>(it.value()*128 +128)));
					}
				}
				cvNamedWindow("Preview", cv::WINDOW_NORMAL);
				cv::imshow("Preview", test);
				

				cv::Mat test2 (currentFP.rows(), currentFP.cols(), CV_8UC1, cv::Scalar::all(128));
				for (int i = 0; i < currentFP.outerSize(); ++i)
				{
					for(Eigen::SparseMatrix<double>::InnerIterator it (currentFP, i); it; ++it)
					{
						
						test2.at<uchar>(it.row(), it.col()) = std::max(0,std::min(255,
							static_cast<int>(it.value()*128 +128)));
					}
				}
				cvNamedWindow("Preview2", cv::WINDOW_NORMAL);
				cv::imshow("Preview2", test2);

				cv::Mat test3 (scans[scanIndex].rows(), scans[scanIndex].cols(), CV_8UC1, cv::Scalar::all(128));
				for (int i = 0; i < scans[scanIndex].outerSize(); ++i)
				{
					for(Eigen::SparseMatrix<double>::InnerIterator it (scans[scanIndex], i); it; ++it)
					{
						
						test3.at<uchar>(it.row(), it.col()) = std::max(0,std::min(255,
							static_cast<int>(it.value()*128 +128)));
					}
				}
				cvNamedWindow("Preview3", cv::WINDOW_NORMAL);
				cv::imshow("Preview3", test3);
				cv::waitKey(0);
			}*/
			
			
			if(FLAGS_metricNumber == 1)
			{
				const double fpScore = currentFP.squaredNorm();
				Eigen::SparseMatrix<double> diff = scans[scanIndex] - currentFP;
				const double diffScore = diff.squaredNorm();

				for(int i = 0; i < diff.outerSize(); ++i){
					for(Eigen::SparseMatrix<double>::InnerIterator it (diff, i); it; ++it){
						if(it.value() != 0)
						{
							const double tmp = it.value()*it.value()/diffScore;
							diffEntropy -= tmp*log(tmp);
						}
					}
				}

				const double placement = diffScore - fpScore;
				score = (placement > 0) ? placement*diffEntropy
				: placement/diffEntropy;
			} else if(FLAGS_metricNumber == 2 || FLAGS_metricNumber == 3)
			{
				Eigen::SparseMatrix<double> currentFPE = fpE.block(point[1], point[0], 
				scans[scanIndex].rows(), scans[scanIndex].cols());

				Eigen::SparseMatrix<double> diff = scans[scanIndex] - currentFPE;

				for(int i = 0; i < diff.outerSize(); ++i){
					for(Eigen::SparseMatrix<double>::InnerIterator it (diff, i); it; ++it){
						if(it.value() > 0){
							scanFPsetDiff += it.value();
						}
					}
				}

				diff = currentFP - scansE[scanIndex];
				diff.makeCompressed();
				for(int i = 0; i < diff.outerSize(); ++i){
					for(Eigen::SparseMatrix<double>::InnerIterator it (diff, i); it; ++it){
						if(it.value() > 0){
							fpScanSetDiff += it.value();
						}
					}
				}
				if(FLAGS_metricNumber == 2)
				{
					score = scanFPsetDiff + 
						std::max(0.0, fpScanSetDiff - 0.5*scanFPsetDiff)*0.05;
				} else
				{
					score = scanFPsetDiff/scans[scanIndex].nonZeros()
						+ fpScanSetDiff/currentFP.nonZeros();
				}

			} else
			{
				Eigen::MatrixXd currentFPNS = Eigen::MatrixXd(currentFP);
				Eigen::MatrixXd scanNS = Eigen::MatrixXd(scans[scanIndex]);
				Eigen::MatrixXd scanToFpDT = distanceTransform(scanNS, currentFPNS);
				Eigen::MatrixXd fpToScanDT = distanceTransform(currentFPNS, scanNS);
				
				double scanDTscore = 0;
				double fpDTscore = 0;
				const double * scanToFpDTPtr = scanToFpDT.data();
				const double * fpToScanDTPtr = fpToScanDT.data();
				for (int i = 0; i < scanToFpDT.size(); ++i)
				{
					scanDTscore += *(scanToFpDTPtr + i);
					fpDTscore += *(fpToScanDTPtr + i);
				}
				score = scanDTscore/scans[scanIndex].nonZeros() + 
					fpDTscore/currentFP.nonZeros();
			}
			

			

			posInfo tmp;
			tmp.x = point[0];
			tmp.y = point[1];
			tmp.score = score;
			tmp.yasu = scanFPsetDiff;
			tmp.yasu2 = fpScanSetDiff;
			tmp.scanPixels = scans[scanIndex].nonZeros();
			tmp.fpPixels = currentFP.nonZeros();
			tmp.rotation = scanIndex;
			tmp.entropy = diffEntropy;
			privateScores.push_back(tmp);
			

		}

		#pragma omp for schedule(static) ordered
		for (int i = 0; i < omp_get_num_threads(); ++i)
		{
			#pragma omp ordered
			scores.insert(scores.end(), privateScores.begin(), privateScores.end());
		}
	}

	if(!FLAGS_quiteMode)
		std::cout << "Done: " << scores.size() << std::endl;
}

void place::findPointsToAnalyze(const std::vector<posInfo> & scores, const std::vector<int> & localMinima,
	std::vector<Eigen::Vector3i> & pointsToAnalyze){
	pointsToAnalyze.clear();
	pointsToAnalyze.reserve(/*localMinima.size()*4 + */globalMins.size()*4);

	for(auto & index : localMinima){
		posInfo minInfo = scores[index];
		int x = minInfo.x;
		int y = minInfo.y;
		int rotIndex = minInfo.rotation;
		pointsToAnalyze.push_back(Eigen::Vector3i(2*x,2*y,rotIndex));
		pointsToAnalyze.push_back(Eigen::Vector3i(2*x + 1,2*y,rotIndex));
		pointsToAnalyze.push_back(Eigen::Vector3i(2*x,2*y + 1,rotIndex));
		pointsToAnalyze.push_back(Eigen::Vector3i(2*x + 1,2*y + 1,rotIndex));
	}

	for(auto & globMin : globalMins)
	{
		posInfo minInfo = scores[globMin];
		int x = minInfo.x;
		int y = minInfo.y;
		int rotIndex = minInfo.rotation;
		pointsToAnalyze.push_back(Eigen::Vector3i(2*x,2*y,rotIndex));
		pointsToAnalyze.push_back(Eigen::Vector3i(2*x + 1,2*y,rotIndex));
		pointsToAnalyze.push_back(Eigen::Vector3i(2*x,2*y + 1,rotIndex));
		pointsToAnalyze.push_back(Eigen::Vector3i(2*x + 1,2*y + 1,rotIndex));
	}
	
}



void place::findGlobalMinima(const std::vector<posInfo> & scores, const std::vector<int> & localMinima){
	globalMins.clear();

	double minScore = 5e30;
	for(auto & info : scores){
		minScore = std::min(info.score, minScore);
	}

	if(!FLAGS_quiteMode)
		std::cout << "Min score: " << minScore << std::endl;

	for(int i = 0; i < scores.size(); ++i){
		if(scores[i].score >=0)
		{
			if(scores[i].score <= 1.01*minScore)
				if(notInLocalMin(i, localMinima))
					globalMins.push_back(i);
		} else
		{
			if(scores[i].score <= minScore/1.01)
				if(notInLocalMin(i, localMinima))
					globalMins.push_back(i);
		}

	}
}

void place::findPointsToAnalyzeV2(const std::vector<posInfo> & scores, std::vector<Eigen::Vector3i> & pointsToAnalyze){
	pointsToAnalyze.clear();

	const int scale = pow(2, FLAGS_numLevels);
	for(auto & globMin : globalMins)
	{
		posInfo info = scores[globMin];
		for (int i = -FLAGS_numLevels; i < FLAGS_numLevels; ++i)
		{
			for (int j = -FLAGS_numLevels; j < FLAGS_numLevels; ++j)
			{
				pointsToAnalyze.push_back(
					Eigen::Vector3i (info.x*scale+i,info.y*scale+j,info.rotation));
			}
		}
	}
}

void place::savePlacement(const std::vector<posInfo> & scores, const std::vector<int> & localMinima, 
	const std::string & outName){
	std::ofstream out (outName, std::ios::out);
	std::ofstream outB (outName.substr(0, outName.find(".")) + ".dat", std::ios::out | std::ios::binary);
	out << "Score x y rotation NCC" << std::endl;
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

	
	std::vector<cv::Mat> rotatedScans;
	place::loadInScans(scanName, rotationFile, rotatedScans, zerosFile);

	std::vector<cv::Mat> trimmedScans;
	trimScans(rotatedScans, trimmedScans);
	

	int numLoc, numGlob;
	in.read(reinterpret_cast<char *>(&numLoc), sizeof(numLoc));
	in.read(reinterpret_cast<char *>(&numGlob), sizeof(numGlob));

	cvNamedWindow("Preview", cv::WINDOW_NORMAL);

	if(!FLAGS_quiteMode)
		std::cout << "Showing localMinima, " << numLoc << std::endl;
	std::vector<posInfo> scores;
	for (int i = 0; i < numLoc; ++i)
	{
		posInfo minScore;
		in.read(reinterpret_cast<char *>(&minScore), sizeof(minScore));
		minScore.x -= zeroZero[minScore.rotation][0];
		minScore.y -= zeroZero[minScore.rotation][1];
		
		cv::Mat bestScan = trimmedScans[minScore.rotation];

		const int xOffset = minScore.x;
		const int yOffset = minScore.y;
		cv::Mat output (floorPlan.rows, floorPlan.cols, CV_8UC3, cv::Scalar::all(255));
		fpColor.copyTo(output);
		
		for (int i = 0; i < bestScan.rows; ++i)
		{
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
		if(!FLAGS_quiteMode)
		{
			std::cout << minScore.score <<"      " << minScore.x << "      " <<minScore.y << std::endl;
			std::cout << minScore.yasu << "      " << minScore.yasu2 << std::endl;
			std::cout << minScore.scanPixels << "    " << minScore.fpPixels << std::endl << std::endl;
		}

		cv::imshow("Preview", output);
		cv::waitKey(0);
	}

	if(!FLAGS_quiteMode)
		std::cout << "Showing globalMins, " << numGlob << std::endl;
	for (int i = 0; i < numGlob; ++i)
	{
		posInfo minScore;
		in.read(reinterpret_cast<char *>(&minScore), sizeof(minScore));
		minScore.x -= zeroZero[minScore.rotation][0];
		minScore.y -= zeroZero[minScore.rotation][1];
		
		cv::Mat bestScan = trimmedScans[minScore.rotation];

		const int xOffset = minScore.x;
		const int yOffset = minScore.y;
		cv::Mat output (floorPlan.rows, floorPlan.cols, CV_8UC3, cv::Scalar::all(255));
		fpColor.copyTo(output);
		
		for (int i = 0; i < bestScan.rows; ++i)
		{
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
		if(!FLAGS_quiteMode)
		{
			std::cout << minScore.score <<"      " << minScore.x << "      " <<minScore.y << std::endl;
			std::cout << minScore.yasu << "      " << minScore.yasu2 << std::endl;
			std::cout << minScore.scanPixels << "    " << minScore.fpPixels << std::endl << std::endl;
		}

		cv::imshow("Preview", output);
		cv::waitKey(0);
	}

	return true;
}

bool place::notInLocalMin(const int i, const std::vector<int> & localMinima){
	if(localMinima.size() == 0)
		return true;
	if(localMinima.size() == 1)
		return !(i == localMinima[0]);

	if(i == localMinima[localMinima.size()/2]){
		return false;
	}else if( i < localMinima[localMinima.size()/2]){
		return notInLocalMin(i, 
			std::vector<int>(localMinima.begin(), localMinima.begin() + localMinima.size()/2));
	} else{
		return notInLocalMin(i, 
			std::vector<int>(localMinima.begin() + localMinima.size()/2, localMinima.end()));
	}
}

void place::blurMinima(std::vector<Eigen::MatrixXd> & scoreMatricies, const std::vector<posInfo> & scores, 
	const Eigen::Vector4i & rows, const Eigen::Vector4i & cols){
	scoreMatricies.clear();

	for (int i = 0; i < rows.size(); ++i)
	{
		scoreMatricies.push_back(Eigen::MatrixXd::Zero(rows[i], cols[i]));
	}
	
	Eigen::MatrixXd kernel (5,5);
	const double sigma = 1.25;
	for (int i = -2; i < kernel.rows()-2; ++i)
	{
		for (int j = -2; j < kernel.cols()-2; ++j)
		{
			kernel(i+2,j+2) = 1.0/(2*PI*sigma*sigma)*exp(-1.0*static_cast<double>(i*i + j*j)/(2*sigma*sigma));
		}
	}

	const double * kernlPtr = kernel.data();

	for(auto minScore : scores){
		if(minScore.rotation == 0)
			scoreMatricies[0](minScore.y, minScore.x) = minScore.score;

		if(minScore.rotation == 1)
			scoreMatricies[1](minScore.y, minScore.x) = minScore.score;

		if(minScore.rotation == 2)
			scoreMatricies[2](minScore.y, minScore.x) = minScore.score;

		if(minScore.rotation == 3)
			scoreMatricies[3](minScore.y, minScore.x) = minScore.score;
	}

	for (int k = 0; k < rows.size(); ++k)
	{
		Eigen::MatrixXd blurredScore = Eigen::MatrixXd::Zero(rows[k], cols[k]);
		for (int i = 2; i < rows[k] - 2; ++i)
		{
			for (int j = 2; j < cols[k] - 2; ++j)
			{
				const Eigen::MatrixXd toBlur = scoreMatricies[k].block(i-2, j-2, 5, 5);
				const double * blurPtr = toBlur.data();
				double value = 0;
				for (int a = 0; a < kernel.size(); ++a)
				{
					value += (*(kernlPtr + a))*(*(blurPtr + a));
				}
				blurredScore(i,j) = value;
			}
		}
		scoreMatricies[k] = blurredScore;
	}
	

	double aveScore = 0;
	int count = 0;
	for (int k = 0; k < rows.size(); ++k)
	{
		const double * scoreMatrixPtr = scoreMatricies[k].data();
		for (int i = 0; i < scoreMatricies[k].size(); ++i)
		{
			if(*(scoreMatrixPtr + i) != 0)
			{
				aveScore += *(scoreMatrixPtr + i);
				count ++;
			}
		}
	}
	
	aveScore /= count;
	double sigScores = 0;
	
	for (int k = 0; k < rows.size(); ++k)
	{
		const double * scoreMatrixPtr = scoreMatricies[k].data();
		for (int i = 0; i <scoreMatricies[k].size(); ++i)
		{
			if(*(scoreMatrixPtr + i) != 0)
			{
				const double tmp = *(scoreMatrixPtr + i) - aveScore;
				sigScores += tmp*tmp;
			}
			
		}
	}

	
	sigScores /= count;
	sigScores = sqrt(sigScores);


	if(FLAGS_visulization)
	{
		for (int k = 0; k < rows.size(); ++k)
		{
			cv::Mat blurScore (rows[k], cols[k], CV_8UC1, cv::Scalar::all(255));
			for (int i = 0; i < blurScore.rows; ++i)
			{
				uchar * dst = blurScore.ptr<uchar>(i);
				for (int j = 0; j < blurScore.cols; ++j)
				{
					if(scoreMatricies[k](i,j) != 0)
					{
						const int gray = std::max(0, std::min(255,
							 static_cast<int>(255.0 * (
							 	((scoreMatricies[k](i,j) - aveScore)/(3.0*sigScores) + 1.0)/2.0))));
						dst[j] = 255 - gray;
					}
				}
			}
			cvNamedWindow("Preview", cv::WINDOW_NORMAL);
			cv::imshow("Preview", blurScore);
			cv::waitKey(0);
		}
		
	}	
}

void place::trimScans(const std::vector<cv::Mat> & rotatedScans, std::vector<cv::Mat> & trimmedScans){
	int k = 0;
	for(auto scan : rotatedScans){
		int minRow = scan.rows;
		int minCol = scan.cols;
		int maxRow = 0;
		int maxCol = 0;

		for (int i = 0; i < scan.rows; ++i)
		{
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

		cv::Mat trimmedScan (maxRow - minRow, maxCol - minCol, CV_8UC1);
		for (int i = minRow; i < maxRow; ++i)
		{
			const uchar * src = scan.ptr<uchar>(i);
			uchar * dst = trimmedScan.ptr<uchar>(i-minRow);
			for (int j = minCol; j < maxCol; ++j)
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

void place::loadInTruePlacement(const std::string & scanName){
	const std::string placementName = FLAGS_preDone + scanName.substr(scanName.find("_")-3, 3) 
	+ "_placement_" + scanName.substr(scanName.find(".")-3, 3) + ".dat";
	std::ifstream in (placementName, std::ios::in | std::ios::binary);

	int numLoc, numGlob;
	in.read(reinterpret_cast<char *>(&numLoc), sizeof(numLoc));
	in.read(reinterpret_cast<char *>(&numGlob), sizeof(numGlob));

	std::vector<posInfo> tmp (numLoc + numGlob);
	for (int i = 0; i < numLoc + numGlob; ++i)
	{
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
	for (int i = 0; i < truePlacement.size(); ++i)
	{
		tmp.push_back(offset + i);
	}

	std::cout << "displaying true placement" << std::endl;
	globalMins.clear();
	place::displayOutput(rSSparseTrimmed, tmp, scores);

}

Eigen::MatrixXd place::distanceTransform(Eigen::MatrixXd & source, Eigen::MatrixXd & comparision){
	Eigen::MatrixXd DT (source.rows(), source.cols());

	for (int y = 0; y < source.rows(); ++y)
	{
		for (int x = 0; x < source.cols(); ++x)
		{
			if(source(y,x) != 0 && comparision(y,x) == 0)
			{
				double r = 1.0;
				bool searching = true;
				while(searching)
				{
					for (int yOffset = -sqrt(r); yOffset < sqrt(r); ++yOffset)
					{

						int xOffset = sqrt(r*r - yOffset*yOffset);

						if(y + yOffset < 0 || y + yOffset >= comparision.rows())
							continue;
						if(x + xOffset < 0 || x + xOffset >= comparision.cols())
							continue;

						if(comparision(y+yOffset, x+xOffset) !=0)
							searching = false;
					}
					++r;
				}
				DT(y,x) = r;
			} else
				DT(y,x) = 0.0;
		}
	}
	return DT;
}