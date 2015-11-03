#include "placeScan.h"


#include <iostream>
#include <dirent.h>
#include <fstream>
#include <math.h>
#include <time.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


using namespace cv;
using namespace std;
using namespace Eigen;

constexpr float fpScale = 86.0*3.28084;  /*pixels per meter */
/*Something like 75 is looking more likely however...Maybe the scanner
isn't in meters?*/
constexpr double PI = 3.14159265;
static Mat floorPlan, fpColor;
static vector<Vector3i> truePlacement;
static vector<int> globalMins;
static int levelNum = 10;
static vector<Vector2d> zeroZero;


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

int main(int argc, char *argv[])
{

	gflags::ParseCommandLineFlags(&argc, &argv, true);
	if(FLAGS_debugMode)
	{
		FLAGS_save = false;
		FLAGS_replace = true;
	}

	floorPlan = imread(FLAGS_floorPlan, 0);
	if(!floorPlan.data){
		cout << "Error reading floorPlan" << endl;
		exit(1);
	}

	fpColor = Mat (floorPlan.rows, floorPlan.cols, CV_8UC3, Scalar::all(255));
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

	vector<string> pointFileNames;
	
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (FLAGS_dmFolder.data())) != NULL) {
	  /* Add all the files and directories to a vector */
	  while ((ent = readdir (dir)) != NULL) {
	  	string fileName = ent->d_name;
	  	if(fileName != ".." && fileName != "." 
	  		&& fileName.find("point") != string::npos){
	  		pointFileNames.push_back(fileName);
	  	}
	  }
	  closedir (dir);
	}  else {
	  /* could not open directory */
	  perror ("");
	  return EXIT_FAILURE;
	}

	vector<string> rotationFileNames;
	if ((dir = opendir (FLAGS_rotFolder.data())) != NULL) {
	  /* Add all the files and directories to a vector */
	  while ((ent = readdir (dir)) != NULL) {
	  	string fileName = ent->d_name;
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

	vector<string> zerosFileNames;
	if ((dir = opendir (FLAGS_zerosFolder.data())) != NULL) {
	  /* Add all the files and directories to a vector */
	  while ((ent = readdir (dir)) != NULL) {
	  	string fileName = ent->d_name;
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
		cout << "Not the same number of scans as rotations!" << endl;
		exit(-1);
	}


	sort(rotationFileNames.begin(), rotationFileNames.end());
	sort(pointFileNames.begin(), pointFileNames.end());
	sort(zerosFileNames.begin(), zerosFileNames.end());
	
	if(FLAGS_numScans == -1)
		FLAGS_numScans = pointFileNames.size()-FLAGS_startIndex;
	
	for(int i = FLAGS_startIndex; i< FLAGS_startIndex + FLAGS_numScans; ++i){
		const string scanName = FLAGS_dmFolder + pointFileNames[i];
		const string rotationFile = FLAGS_rotFolder + rotationFileNames[i];
		const string zerosFile = FLAGS_zerosFolder + zerosFileNames[i];
		if(FLAGS_replace)
			analyzePlacement(floorPlan, scanName, rotationFile, zerosFile);
		else if(!reshowPlacement(scanName, rotationFile, zerosFile))
			analyzePlacement(floorPlan, scanName, rotationFile, zerosFile);
	}
	
	return 0;
}

void analyzePlacement(const Mat & fp, const string & scanName, 
	const string & rotationFile, const string & zerosFile){
	if(!FLAGS_quiteMode)
		cout << rotationFile << endl;
	cout << scanName << endl;

	Mat element = getStructuringElement(MORPH_RECT, Size(6,6));
	Mat fpEroded (fp.rows, fp.cols, CV_8UC1);
	erode(fp, fpEroded, element);
	
	vector<Mat> rotatedScans;
	loadInScans(scanName, rotationFile, rotatedScans, zerosFile);
	vector<Mat> erodedScans;

	for(auto scan : rotatedScans){
		Mat dst (scan.rows, scan.cols, CV_8UC1);
		erode(scan, dst, element);
		erodedScans.push_back(dst);
	}

	/*for(int i = 0; i< 4; i++){
		cvNamedWindow("Preview1", WINDOW_NORMAL);
		imshow("Preview1", erodedScans[i]);
		cvNamedWindow("Preview2", WINDOW_NORMAL);
		imshow("Preview2", rotatedScans[i]);
		waitKey(0);
	}

	
	cvNamedWindow("Preview", WINDOW_NORMAL);
	imshow("Preview", fpEroded);
	waitKey(0);*/

	vector<SparseMatrix<double> > erodedSSparse;
	vector<SparseMatrix<double> > rSSparse;
	for (int i = 0; i < rotatedScans.size(); ++i)
	{
		SparseMatrix<double> eroded, scan;
		scanToSparse(erodedScans[i], eroded);
		scanToSparse(rotatedScans[i], scan);
		erodedSSparse.push_back(eroded);
		rSSparse.push_back(scan);
	}
	

	SparseMatrix<double> fpSparse, erodedFpSparse;
	scanToSparse(fp, fpSparse);
	scanToSparse(fpEroded, erodedFpSparse);


	vector<SparseMatrix<double> > fpPyramid = {fpSparse};
	vector<vector<SparseMatrix<double> > > rSSparsePyramid = {rSSparse};
	createPyramid(fpPyramid, rSSparsePyramid);
	
	
	
	vector<SparseMatrix<double> > erodedFpPyramid = {erodedFpSparse};
	vector<vector<SparseMatrix<double> > > erodedSparsePyramid = {erodedSSparse};
	createPyramid(erodedFpPyramid, erodedSparsePyramid);

	vector<vector<SparseMatrix<double> > > erodedSparsePyramidTrimmed;
	vector<vector<SparseMatrix<double> > > rSSparsePyramidTrimmed;
	trimScanPryamids(rSSparsePyramid, rSSparsePyramidTrimmed, 
		erodedSparsePyramid, erodedSparsePyramidTrimmed);

	vector<Vector3i> pointsToAnalyze;
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
				pointsToAnalyze.push_back(Vector3i (i,j,k));
			}
		}
	}

	if(FLAGS_debugMode)
		loadInTruePlacement(scanName);
	
	vector<posInfo> scores;
	vector<MatrixXd> scoreMatricies;
	vector<int> localMinima;
	Vector4i rows, cols;
	for (int k = FLAGS_numLevels; k >= 0; --k)
	{
		levelNum = k;
		if(k == 0)
		{
			for(auto & v : truePlacement){
				pointsToAnalyze.push_back(v);
			}
		}
		findPlacementPointBasedV2(fpPyramid[k], rSSparsePyramidTrimmed[k],
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
			findLocalMinima(scores, localMinima, 1.0);
			findGlobalMinima(scores, localMinima);
			findPointsToAnalyze(scores, localMinima, pointsToAnalyze);
		}
		
	}

	findLocalMinima(scores, localMinima, 2.5);
	findGlobalMinima(scores, localMinima);

	if(FLAGS_save)
	{
		const string placementName = FLAGS_preDone + scanName.substr(scanName.find("_")-3, 3) 
		+ "_placement_" + scanName.substr(scanName.find(".")-3, 3) + ".txt";
		
		savePlacement(scores, localMinima, placementName);
	}
	
	if(FLAGS_visulization || FLAGS_previewOut)
		displayOutput(rSSparsePyramidTrimmed[0], localMinima, scores);
	if(FLAGS_debugMode)
	{
		/*findPlacementPointBasedV2(fpPyramid[k], rSSparsePyramidTrimmed[k],
			erodedFpPyramid[k], erodedSparsePyramidTrimmed[k], 
			scores, truePlacement);*/
		displayTruePlacement(rSSparsePyramidTrimmed[0], scores);
	}
}

void loadInScans(const string & scanName, const string & rotationFile, 
	vector<Mat> & rotatedScans, const string & zerosFile){
	ifstream binaryReader (rotationFile, ios::in | ios::binary);
	vector<Matrix3d> R (4);
	for (int i = 0; i < R.size(); ++i)
	{
		binaryReader.read(reinterpret_cast<char *>(&R[i]),
			sizeof(Matrix3d));
	}

	binaryReader.close();

	binaryReader.open(zerosFile, ios::in | ios::binary);
	zeroZero.clear();
	zeroZero.resize(4);
	for (int i = 0; i < 4; ++i)
	{
		binaryReader.read(reinterpret_cast<char *>(&zeroZero[i]),
			sizeof(Vector2d));
		binaryReader.seekg(0);
	}
	binaryReader.close();

	Mat scan = imread(scanName, 0);
	
	if(!scan.data){
		cout << "Error reading scan" << endl;
		exit(1);
	}
	if(FLAGS_tinyPreviewIn)
	{
		cvNamedWindow("Preview", WINDOW_NORMAL);
		imshow("Preview", scan);
		waitKey(0);
	}

	const int maxDimension = max(1.1*scan.rows, 1.1*scan.cols);
	const int colOffset = (maxDimension - scan.cols)/2;
	const int rowOffset = (maxDimension - scan.rows)/2;
	for (int i = 0; i < 4; ++i)
	{
		zeroZero[i][0] += colOffset;
		zeroZero[i][0] += rowOffset;
	}

	Mat widenedScan (maxDimension, maxDimension, CV_8UC1, Scalar::all(255));
	for (int i = 0; i < scan.rows; ++i)
	{
		uchar * src = scan.ptr<uchar>(i);
		uchar * dst = widenedScan.ptr<uchar>(i + rowOffset);
		for (int j = 0; j < scan.cols; ++j)
		{
			dst[j+colOffset] = src[j];
		}
	}

	const Vector3d center (widenedScan.cols/2.0, widenedScan.rows/2.0, 0.0);

	int i = 0;
	for(auto & rot : R)
	{
		Mat rScan (widenedScan.rows, widenedScan.cols, CV_8UC1, Scalar::all(255));
		for (int i = 0; i < widenedScan.rows; ++i)
		{
			uchar * dst = rScan.ptr<uchar>(i);
			for (int j = 0; j < widenedScan.cols; ++j)
			{
				const Vector3d pixel (j, i, 0.0);
				const Vector3d src = rot*(pixel-center) + center;
				if(src[0] < 0 || src[0] >= widenedScan.cols )
					continue;
				if(src[1] < 0 || src[1] >= widenedScan.rows)
					continue;
				dst[j] = widenedScan.at<uchar>(src[0], src[1]);
			}
		}
		rotatedScans.push_back(rScan);

		
		Vector3d tmp (zeroZero[i][0], zeroZero[i][1], 0.0);
		tmp = rot*(tmp - center) + center;
		zeroZero[i][0] = tmp[0];
		zeroZero[i][0] = tmp[1];
		i++;
	}



	if(FLAGS_visulization || FLAGS_previewIn)
	{
		for(auto & scan : rotatedScans){
			cvNamedWindow("Preview", WINDOW_NORMAL);
			imshow("Preview", scan);
			waitKey(0);
		}
	}
}

void scanToSparse(const Mat & scan, SparseMatrix<double> & sparse){
	vector<Triplet<double> > tripletList;

	for (int i = 0; i < scan.rows; ++i)
	{
		const uchar * src = scan.ptr<uchar>(i);
		for (int j = 0; j < scan.cols; ++j)
		{
			if(src[j] == 255)
				continue;
			double confidence = 1.0 -(double)src[j]/255.0;
			tripletList.push_back(Triplet<double> (i,j,confidence));
		}
	}
	sparse = SparseMatrix<double> (scan.rows, scan.cols);
	sparse.setFromTriplets(tripletList.begin(), tripletList.end());
}


void displayOutput(const vector<SparseMatrix<double> > & rSSparseTrimmed, 
	const vector<int> & localMinima, const vector<posInfo> & scores){
	if(!FLAGS_quiteMode)
	{
		cout << "Num localMinima: " << localMinima.size() << endl;
		cout << "Press a key to begin displaying placement options" << endl;
	}
	
	cvNamedWindow("Preview", WINDOW_NORMAL);
	imshow("Preview", fpColor);
	waitKey(0);


	for(auto & index : localMinima){
		posInfo minScore = scores[index];
		const int xOffset = minScore.x;
		const int yOffset = minScore.y;
		const SparseMatrix<double> & currentScan = rSSparseTrimmed[minScore.rotation]; 
		Mat output (floorPlan.rows, floorPlan.cols, CV_8UC3, Scalar::all(255));
		fpColor.copyTo(output);

		Mat_<Vec3b> _output = output;

		for (int i = 0; i < currentScan.outerSize(); ++i)
		{
			for(SparseMatrix<double>::InnerIterator it(currentScan, i); it; ++it){
				_output(it.row() + yOffset, it.col() + xOffset)[0]=0;
				_output(it.row() + yOffset, it.col() + xOffset)[1]=0;
				_output(it.row() + yOffset, it.col() + xOffset)[2]=255;

			}
		}
		
		imshow("Preview", output);
		if(!FLAGS_quiteMode)
			cout << minScore.score <<"      " << minScore.x << "      " <<minScore.y << endl;
			cout << minScore.yasu << "      " << minScore.yasu2 << endl;
			cout << minScore.scanPixels << "    " << minScore.fpPixels << endl << endl;
		waitKey(0);
		
		
		~output;
	}


	if(!FLAGS_quiteMode)
		cout << "Num globalMins: " << globalMins.size() << endl;
	for(auto & globMin : globalMins)
	{
		posInfo minScore = scores[globMin];
		const int xOffset = minScore.x;
		const int yOffset = minScore.y;
		/*if(yOffset == 0)
			continue;*/
		const SparseMatrix<double> & currentScan = rSSparseTrimmed[minScore.rotation]; 
		Mat output (floorPlan.rows, floorPlan.cols, CV_8UC3, Scalar::all(255));
		fpColor.copyTo(output);

		Mat_<Vec3b> _output = output;

		for (int i = 0; i < currentScan.outerSize(); ++i)
		{
			for(SparseMatrix<double>::InnerIterator it(currentScan, i); it; ++it){
				_output(it.row() + yOffset, it.col() + xOffset)[0]=0;
				_output(it.row() + yOffset, it.col() + xOffset)[1]=0;
				_output(it.row() + yOffset, it.col() + xOffset)[2]=255;

			}
		}

		output = _output;
		imshow("Preview", output);
		if(!FLAGS_quiteMode)
		{
			cout << minScore.score <<"      " << minScore.x << "      " <<minScore.y << endl;
			cout << minScore.yasu << "      " << minScore.yasu2 << endl;
			cout << minScore.scanPixels << "    " << minScore.fpPixels << endl << endl;
		}
		
		waitKey(0);
		~output;
	}
}


void findLocalMinima(const vector<posInfo> & scores, vector<int> & localMinima, 
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
		cout << "Average         Sigma" << endl;
		cout << averageScore << "         " << sigScores << endl; 
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

void createPyramid(vector<SparseMatrix<double> > & fpPyramid,
	vector<vector<SparseMatrix<double> > > & rSSparsePyramid){


	vector<Triplet<double> > tripletList;
	
	for (int i = 0; i < FLAGS_numLevels; ++i)
	{
		SparseMatrix<double> & currentFP = fpPyramid[i];
		SparseMatrix<double> fpLevel (floor(currentFP.rows()/2), 
			floor(currentFP.cols()/2));

		MatrixXd currentFPNS = MatrixXd(currentFP);
		
		for (int j = 0; j < (currentFPNS.rows() -1); j+=2)
		{
			for (int k = 0; k < (currentFPNS.cols() -1); k+=2)
			{
				double maxV = max(currentFPNS(j,k),max(currentFPNS(j,k+1),
					max(currentFPNS(j+1,k), currentFPNS(j+1,k+1))));
				tripletList.push_back(Triplet<double> (j/2, k/2, maxV));
			}
		}


		fpLevel.setFromTriplets(tripletList.begin(), tripletList.end());
		fpPyramid.push_back(fpLevel);
		tripletList.clear();

		vector<SparseMatrix<double> > rsLevel;
		for(auto & scan : rSSparsePyramid[i]){
			SparseMatrix<double> scanLevel (floor(scan.rows()/2), floor(scan.cols()/2));
			MatrixXd scanNS = MatrixXd(scan);

			for (int j = 0; j < (scan.rows()-1); j+=2)
			{
				for (int k = 0; k < (scan.cols()-1); k+=2)
				{
					double maxV = max(scanNS(j,k),max(scanNS(j,k+1),
					max(scanNS(j+1,k), scanNS(j+1,k+1))));
					tripletList.push_back(Triplet<double> (j/2, k/2, maxV));
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
			Mat test (level.rows(), level.cols(), CV_8UC1, Scalar::all(255));
			for (int k = 0; k < level.outerSize(); ++k)
			{
				for (SparseMatrix<double>::InnerIterator it (level, k); it; ++it)
				{
					test.at<uchar>(it.row(), it.col()) = 255 - it.value()*255;
				}
			}
			cvNamedWindow("Preview", WINDOW_NORMAL);
			imshow("Preview", test);
			waitKey(0);
		}
	}
}

void trimScanPryamids(const vector<vector<SparseMatrix<double> > > & rSSparsePyramid,
	vector<vector<SparseMatrix<double> > > & rSSparsePyramidTrimmed, 
	const vector<vector<SparseMatrix<double> > > & erodedSparsePyramid,
	vector<vector<SparseMatrix<double> > > & erodedSparsePyramidTrimmed){

	vector<Triplet<double> > tripletList;
	int first4 = 0;
	for(int level = 0; level < rSSparsePyramid.size(); ++level){
		vector<SparseMatrix<double> > levelTrimmed;
		vector<SparseMatrix<double> > erodedLevelTrimed;
		for(int i = 0; i < rSSparsePyramid[level].size(); ++i){
			const SparseMatrix<double> * scan = &rSSparsePyramid[level][i];
			const SparseMatrix<double> * erodedScan = &erodedSparsePyramid[level][i];

			SparseMatrix<double> scanThreshHolded (scan->rows(), scan->cols());
			SparseMatrix<double> eScanThreshHolded (erodedScan->rows(), erodedScan->cols());

			for (int k = 0; k < scan->outerSize(); ++k)
			{
				for (SparseMatrix<double>::InnerIterator it (*scan, k); it; ++it)
				{
					if(it.value() > 0.85){
						tripletList.push_back(Triplet<double> (it.row(), it.col(), it.value()));
					}
				}
			}
			scanThreshHolded.setFromTriplets(tripletList.begin(), tripletList.end());
			tripletList.clear();

			for (int k = 0; k < erodedScan->outerSize(); ++k)
			{
				for (SparseMatrix<double>::InnerIterator it (*erodedScan, k); it; ++it)
				{
					if(it.value() > 0.85){
						tripletList.push_back(Triplet<double> (it.row(), it.col(), it.value()));
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
				for (SparseMatrix<double>::InnerIterator it (scanThreshHolded, k); it; ++it)
				{
					if(it.value() !=0){
						maxRow = max(maxRow, it.row());
						minRow = min(minRow, it.row());

						maxCol = max(maxCol, it.col());
						minCol = min(minCol, it.col());
					}
				}
			}

			SparseMatrix<double> trimmed = scanThreshHolded.block(minRow, minCol, 
				maxRow - minRow, maxCol - minCol);
			SparseMatrix<double> erodedTrimmed = eScanThreshHolded.block(minRow, minCol, 
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
		assert(levelTrimmed.size() == 4);
		rSSparsePyramidTrimmed.push_back(levelTrimmed);
		erodedSparsePyramidTrimmed.push_back(erodedLevelTrimed);
	}

	if(FLAGS_visulization)
	{
		for(auto & level: rSSparsePyramidTrimmed){
			for(auto & scan : level){
				Mat test (scan.rows(), scan.cols(), CV_8UC1, Scalar::all(255));
				for (int k = 0; k < scan.outerSize(); ++k)
				{
					for (SparseMatrix<double>::InnerIterator it (scan, k); it; ++it)
					{
						test.at<uchar>(it.row(), it.col()) = 255 - it.value()*255;
					}
				}
				cvNamedWindow("Preview", WINDOW_NORMAL);
				imshow("Preview", test);
				waitKey(0);
			}
		}
	}
}

void findPlacementPointBasedV2(const SparseMatrix<double> & fp, 
	const vector<SparseMatrix<double> > & scans,const SparseMatrix<double> & fpE, 
	const vector<SparseMatrix<double> > & scansE, vector<posInfo> & scores,
	const vector<Vector3i> & points){
	
	if(!FLAGS_quiteMode)
		cout << points.size() << endl;

	scores.clear();
	scores.reserve(points.size());


	#pragma omp parallel shared(fp, scans, points, scansE)
	{
		vector<posInfo> privateScores;
		#pragma omp for nowait schedule(static)
		for(int i = 0; i< points.size(); ++i)
		{
			
			const Vector3i point = points[i];
			const int scanIndex = point[2];
			const int xStop = fp.cols() - scans[scanIndex].cols();
			const int yStop = fp.rows() - scans[scanIndex].rows();

			if(point[0] < 0 || point[0] >=xStop)
				continue;
			if(point[1] < 0 || point[1] >= yStop)
				continue;
			

			SparseMatrix<double> currentFP = fp.block(point[1], point[0], 
				scans[scanIndex].rows(), scans[scanIndex].cols());
			currentFP.makeCompressed();

			if(currentFP.nonZeros() == 0)
				continue;

			SparseMatrix<double> currentFPE = fpE.block(point[1], point[0], 
				scans[scanIndex].rows(), scans[scanIndex].cols());
			currentFPE.makeCompressed();

			double fpScore = currentFP.squaredNorm();

			SparseMatrix<double> diff = scans[scanIndex] - currentFPE;
			diff.makeCompressed();

			/*if(point[0] == static_cast<int>(1187.0/pow(2, levelNum)) &&
				point[1] == static_cast<int>(183.0/pow(2,levelNum)))
			{
				Mat test (diff.rows(), diff.cols(), CV_8UC1, Scalar::all(128));
				for (int i = 0; i < diff.outerSize(); ++i)
				{
					for(SparseMatrix<double>::InnerIterator it (diff, i); it; ++it)
					{
						
						test.at<uchar>(it.row(), it.col()) = max(0,min(255,
							static_cast<int>(it.value()*128 +128)));
					}
				}
				cvNamedWindow("Preview", WINDOW_NORMAL);
				imshow("Preview", test);
				

				Mat test2 (currentFP.rows(), currentFP.cols(), CV_8UC1, Scalar::all(128));
				for (int i = 0; i < currentFP.outerSize(); ++i)
				{
					for(SparseMatrix<double>::InnerIterator it (currentFP, i); it; ++it)
					{
						
						test2.at<uchar>(it.row(), it.col()) = max(0,min(255,
							static_cast<int>(it.value()*128 +128)));
					}
				}
				cvNamedWindow("Preview2", WINDOW_NORMAL);
				imshow("Preview2", test2);

				Mat test3 (scans[scanIndex].rows(), scans[scanIndex].cols(), CV_8UC1, Scalar::all(128));
				for (int i = 0; i < scans[scanIndex].outerSize(); ++i)
				{
					for(SparseMatrix<double>::InnerIterator it (scans[scanIndex], i); it; ++it)
					{
						
						test3.at<uchar>(it.row(), it.col()) = max(0,min(255,
							static_cast<int>(it.value()*128 +128)));
					}
				}
				cvNamedWindow("Preview3", WINDOW_NORMAL);
				imshow("Preview3", test3);
				waitKey(0);
			}*/
			

			double diffScore = diff.squaredNorm();
			double scanFPsetDiff = 0;
			double fpScanSetDiff = 0;

			
			double diffEntropy = 0;
			for(int i = 0; i < diff.outerSize(); ++i){
				for(SparseMatrix<double>::InnerIterator it (diff, i); it; ++it){
					if(it.value() > 0){
						const double tmp = it.value()*it.value()/diffScore;
						diffEntropy -= tmp*log(tmp);
						scanFPsetDiff += it.value();
						/*++scanFPsetDiff;*/
					}
				}
			}

			diff = currentFP - scansE[scanIndex];
			for(int i = 0; i < diff.outerSize(); ++i){
				for(SparseMatrix<double>::InnerIterator it (diff, i); it; ++it){
					if(it.value() > 0){
						fpScanSetDiff += it.value();
						/*++fpScanSetDiff;*/
					}
				}
			}

			


			

			/*MatrixXd currentFPNS = MatrixXd(currentFP);
			MatrixXd currentScanNS = MatrixXd(scans[scanIndex]);
			const double fpMean = currentFPNS.mean();
			const double scanMean = currentScanNS.mean();
			const double* fpData = currentFPNS.data();
			const double* scanData = currentScanNS.data();
			double pq, qq, pp;
			pq = qq = pp = 0;
			for (int i = 0; i < currentFPNS.size(); ++i)
			{
				const double diffP = *(fpData + i) - fpMean;
				const double diffQ = *(scanData + i) - scanMean;
				pq += diffP*diffQ;
				qq += diffQ*diffQ;
				pp += diffP*diffP;
			}
			const double nccScore = pq/sqrt(pp*qq);	*/

			const double placement = diffScore - fpScore;
			const double scoreV1 = (placement > 0) ? placement*diffEntropy
				: placement/diffEntropy;
			const double scoreV2 = scanFPsetDiff + 
				max(0.0, fpScanSetDiff - 0.5*scanFPsetDiff)*0.05;
			const double scoreV3 = scanFPsetDiff/scans[scanIndex].nonZeros() + fpScanSetDiff/currentFP.nonZeros();

			posInfo tmp;
			tmp.x = point[0];
			tmp.y = point[1];
			tmp.score = scoreV3;
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
		cout << "Done: " << scores.size() << endl;
}

void findPointsToAnalyze(const vector<posInfo> & scores, const vector<int> & localMinima,
	vector<Vector3i> & pointsToAnalyze){
	pointsToAnalyze.clear();
	pointsToAnalyze.reserve(/*localMinima.size()*4 + */globalMins.size()*4);

	for(auto & index : localMinima){
		posInfo minInfo = scores[index];
		int x = minInfo.x;
		int y = minInfo.y;
		int rotIndex = minInfo.rotation;
		pointsToAnalyze.push_back(Vector3i(2*x,2*y,rotIndex));
		pointsToAnalyze.push_back(Vector3i(2*x + 1,2*y,rotIndex));
		pointsToAnalyze.push_back(Vector3i(2*x,2*y + 1,rotIndex));
		pointsToAnalyze.push_back(Vector3i(2*x + 1,2*y + 1,rotIndex));
	}

	for(auto & globMin : globalMins)
	{
		posInfo minInfo = scores[globMin];
		int x = minInfo.x;
		int y = minInfo.y;
		int rotIndex = minInfo.rotation;
		pointsToAnalyze.push_back(Vector3i(2*x,2*y,rotIndex));
		pointsToAnalyze.push_back(Vector3i(2*x + 1,2*y,rotIndex));
		pointsToAnalyze.push_back(Vector3i(2*x,2*y + 1,rotIndex));
		pointsToAnalyze.push_back(Vector3i(2*x + 1,2*y + 1,rotIndex));
	}
	
}



void findGlobalMinima(const vector<posInfo> & scores, const vector<int> & localMinima){
	globalMins.clear();

	double minScore = 5e30;
	for(auto & info : scores){
		minScore = min(info.score, minScore);
	}

	if(!FLAGS_quiteMode)
		cout << "Min score: " << minScore << endl;

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

void findPointsToAnalyzeV2(const vector<posInfo> & scores, vector<Vector3i> & pointsToAnalyze){
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
					Vector3i (info.x*scale+i,info.y*scale+j,info.rotation));
			}
		}
	}
}

void savePlacement(const vector<posInfo> & scores, const vector<int> & localMinima, 
	const string & outName){
	ofstream out (outName, ios::out);
	ofstream outB (outName.substr(0, outName.find(".")) + ".dat", ios::out | ios::binary);
	out << "Score x y rotation NCC" << endl;
	const int numLoc = localMinima.size();
	const int numGlob = globalMins.size();
	outB.write(reinterpret_cast<const char *>(&numLoc), sizeof(numLoc));
	outB.write(reinterpret_cast<const char *>(&numGlob), sizeof(numGlob));
	out << "localMinima: " << endl;
	for(auto index : localMinima){
		posInfo minScore = scores[index];
		minScore.x += zeroZero[minScore.rotation][0];
		minScore.y += zeroZero[minScore.rotation][1];
		out << minScore.score << " " << minScore.x  << " "
			<< minScore.y << " " << minScore.rotation << endl;
		
		outB.write(reinterpret_cast<const char *> (&minScore), sizeof(minScore));
	}
	out << endl << "globalMins: " << endl;
	for(auto index : globalMins){
		posInfo minScore = scores[index];
		minScore.x += zeroZero[minScore.rotation][0];
		minScore.y += zeroZero[minScore.rotation][1];
		out << minScore.score << " " << minScore.x  << " "
			<< minScore.y  << " " << minScore.rotation << endl;
		
		outB.write(reinterpret_cast<const char *> (&minScore), sizeof(minScore));
	}
	out.close();
	outB.close();
}

bool reshowPlacement(const string & scanName, const string & rotationFile,
	const string & zerosFile){
	const string placementName = FLAGS_preDone + scanName.substr(scanName.find("_")-3, 3) 
	+ "_placement_" + scanName.substr(scanName.find(".")-3, 3) + ".dat";
	if(!FLAGS_quiteMode)
		cout << placementName << endl;

	ifstream in (placementName, ios::in | ios::binary);
	if(!in.is_open())
		return false;
	if(!FLAGS_previewOut)
		return true;

	
	vector<Mat> rotatedScans;
	loadInScans(scanName, rotationFile, rotatedScans, zerosFile);

	vector<Mat> trimmedScans;
	trimScans(rotatedScans, trimmedScans);
	

	int numLoc, numGlob;
	in.read(reinterpret_cast<char *>(&numLoc), sizeof(numLoc));
	in.read(reinterpret_cast<char *>(&numGlob), sizeof(numGlob));

	cvNamedWindow("Preview", WINDOW_NORMAL);

	if(!FLAGS_quiteMode)
		cout << "Showing localMinima, " << numLoc << endl;
	vector<posInfo> scores;
	for (int i = 0; i < numLoc; ++i)
	{
		posInfo minScore;
		in.read(reinterpret_cast<char *>(&minScore), sizeof(minScore));
		minScore.x -= zeroZero[minScore.rotation][0];
		minScore.y -= zeroZero[minScore.rotation][1];
		
		Mat bestScan = trimmedScans[minScore.rotation];

		const int xOffset = minScore.x;
		const int yOffset = minScore.y;
		Mat output (floorPlan.rows, floorPlan.cols, CV_8UC3, Scalar::all(255));
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
			cout << minScore.score <<"      " << minScore.x << "      " <<minScore.y << endl;
			cout << minScore.yasu << "      " << minScore.yasu2 << endl;
			cout << minScore.scanPixels << "    " << minScore.fpPixels << endl << endl;
		}

		imshow("Preview", output);
		waitKey(0);
	}

	if(!FLAGS_quiteMode)
		cout << "Showing globalMins, " << numGlob << endl;
	for (int i = 0; i < numGlob; ++i)
	{
		posInfo minScore;
		in.read(reinterpret_cast<char *>(&minScore), sizeof(minScore));
		minScore.x -= zeroZero[minScore.rotation][0];
		minScore.y -= zeroZero[minScore.rotation][1];
		
		Mat bestScan = trimmedScans[minScore.rotation];

		const int xOffset = minScore.x;
		const int yOffset = minScore.y;
		Mat output (floorPlan.rows, floorPlan.cols, CV_8UC3, Scalar::all(255));
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
			cout << minScore.score <<"      " << minScore.x << "      " <<minScore.y << endl;
			cout << minScore.yasu << "      " << minScore.yasu2 << endl;
			cout << minScore.scanPixels << "    " << minScore.fpPixels << endl << endl;
		}

		imshow("Preview", output);
		waitKey(0);
	}

	return true;
}

bool notInLocalMin(const int i, const vector<int> & localMinima){
	if(localMinima.size() == 0)
		return true;
	if(localMinima.size() == 1)
		return !(i == localMinima[0]);

	if(i == localMinima[localMinima.size()/2]){
		return false;
	}else if( i < localMinima[localMinima.size()/2]){
		return notInLocalMin(i, 
			vector<int>(localMinima.begin(), localMinima.begin() + localMinima.size()/2));
	} else{
		return notInLocalMin(i, 
			vector<int>(localMinima.begin() + localMinima.size()/2, localMinima.end()));
	}
}

void blurMinima(vector<MatrixXd> & scoreMatricies, const vector<posInfo> & scores, 
	const Vector4i & rows, const Vector4i & cols){
	scoreMatricies.clear();

	for (int i = 0; i < rows.size(); ++i)
	{
		scoreMatricies.push_back(MatrixXd::Zero(rows[i], cols[i]));
	}
	
	MatrixXd kernel (5,5);
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
		MatrixXd blurredScore = MatrixXd::Zero(rows[k], cols[k]);
		for (int i = 2; i < rows[k] - 2; ++i)
		{
			for (int j = 2; j < cols[k] - 2; ++j)
			{
				const MatrixXd toBlur = scoreMatricies[k].block(i-2, j-2, 5, 5);
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
			Mat blurScore (rows[k], cols[k], CV_8UC1, Scalar::all(255));
			for (int i = 0; i < blurScore.rows; ++i)
			{
				uchar * dst = blurScore.ptr<uchar>(i);
				for (int j = 0; j < blurScore.cols; ++j)
				{
					if(scoreMatricies[k](i,j) != 0)
					{
						const int gray = max(0, min(255,
							 static_cast<int>(255.0 * (
							 	((scoreMatricies[k](i,j) - aveScore)/(3.0*sigScores) + 1.0)/2.0))));
						dst[j] = 255 - gray;
					}
				}
			}
			cvNamedWindow("Preview", WINDOW_NORMAL);
			imshow("Preview", blurScore);
			waitKey(0);
		}
		
	}	
}

void trimScans(const vector<Mat> & rotatedScans, vector<Mat> & trimmedScans){
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
					minRow = min(i, minRow);
					minCol = min(j, minCol);
					maxRow = max(i, maxRow);
					maxCol = max(j, maxCol);
				}
			}
		}

		Mat trimmedScan (maxRow - minRow, maxCol - minCol, CV_8UC1);
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
		k++;
	}
}

void loadInTruePlacement(const string & scanName){
	const string placementName = FLAGS_preDone + scanName.substr(scanName.find("_")-3, 3) 
	+ "_placement_" + scanName.substr(scanName.find(".")-3, 3) + ".dat";
	cout << placementName << endl;
	ifstream in (placementName, ios::in | ios::binary);

	int numLoc, numGlob;
	in.read(reinterpret_cast<char *>(&numLoc), sizeof(numLoc));
	in.read(reinterpret_cast<char *>(&numGlob), sizeof(numGlob));

	vector<posInfo> tmp (numLoc + numGlob);
	for (int i = 0; i < numLoc + numGlob; ++i)
	{
		in.read(reinterpret_cast<char *>(&tmp[i]), sizeof(posInfo));
	}

	truePlacement.clear();
	for(auto & s : tmp){
		Vector3i tmp2 (s.x - zeroZero[s.rotation][0], 
			s.y - zeroZero[s.rotation][1], s.rotation);
		truePlacement.push_back(tmp2);
	}
}

void displayTruePlacement(const vector<SparseMatrix<double> > & rSSparseTrimmed,
	const vector<posInfo> & scores){

	vector<int> tmp;
	const int offset = scores.size() - truePlacement.size();
	for (int i = 0; i < truePlacement.size(); ++i)
	{
		tmp.push_back(offset + i);
	}

	cout << "displaying true placement" << endl;
	globalMins.clear();
	displayOutput(rSSparseTrimmed, tmp, scores);

}