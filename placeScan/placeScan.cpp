#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <Eigen/Sparse>
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
#include <omp.h>


using namespace cv;
using namespace std;
using namespace Eigen;

const float fpScale = 86.0*3.28084;  /*pixels per meter */
/*Something like 75 is looking more likely however...Maybe the scanner
isn't in meters?*/
const double PI = 3.14159265;


typedef struct
{
	double score = 0;
	double entropy = 0;
	double NCC = 0;
	int rotation = 0;
	int x = 0;
	int y = 0;
	
} posInfo;

#pragma omp declare reduction (merge : std::vector<posInfo> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp declare reduction (merge : std::vector<int> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))



void analyzePlacement(const SparseMatrix<double> &, const string &, const string &);
void findLocalMinima(const vector<posInfo> &, vector<int> &, const int);
void createPyramid(vector<SparseMatrix<double> > &, vector<vector<SparseMatrix<double> > > &);
void trimScanPryamid(const vector<vector<SparseMatrix<double> > > &, 
	vector<vector<SparseMatrix<double> > > &);
void findPlacementPointBasedV2(const SparseMatrix<double> &, 
	const vector<SparseMatrix<double> > &, vector<posInfo> &, const vector<Vector3i> &);
void findPointsToAnalyze(const vector<posInfo> &, const vector<int> &, vector<Vector3i> &);
void findGlobalMinima(const vector<posInfo> &, const vector<int> &);
void findPointsToAnalyzeV2(const vector<posInfo> &, vector<Vector3i> &);
void displayOutput(const vector<SparseMatrix<double> > &, const vector<int> & ,
	const vector<posInfo> &);
void savePlacement(const vector<posInfo> &, const vector<int> &, const string & outName);
bool reshowPlacement(const string &, const string &);
void loadInScans(const string &, const string &, vector<Mat> &);
bool notInLocalMin(const int, const vector<int> &);
void scansToSparse(const vector<Mat> &, vector<SparseMatrix<double> > &);
void blurMinima(MatrixXd &, const vector<posInfo> &, int, int, int);

static Mat floorPlan, fpColor;

vector<Vector4i> croppedInfo;
vector<int> globalMins;

DEFINE_bool(visulization, false, 
	"Turns on all visulization options that do not impact performance");
DEFINE_bool(previewIn, false, "Turns on a preview of the scan before it is placed");
DEFINE_bool(previewOut, true, "Shows a preview of the scans placement before saving");
DEFINE_bool(replace, false, "Forces the program to redo the placemet of all scans given");
DEFINE_bool(quiteMode, false, "Turns of all status reports");
DEFINE_string(floorPlan, "/home/erik/Projects/3DscanData/DUC/floorPlans/DUC-floor-1_cropped.png", 
	"Path to the floor plan that the scan should be placed on");
DEFINE_string(dmFolder, "/home/erik/Projects/3DscanData/DUC/densityMaps/",
	"Path to folder containing densityMaps");
DEFINE_string(rotFolder, "/home/erik/Projects/3DscanData/DUC/densityMaps/rotations/",
	"Path to folder containing the dominate direction rotations");
DEFINE_string(preDone, "/home/erik/Projects/3DscanData/DUC/placementOptions/",
	"Path to folder containing previous placements of a scan");
DEFINE_int32(startIndex, 0, "Scan number to start with");
DEFINE_int32(numScans, -1, 
	"Number of scans to place, default or -1 will cause all scans in the folder to placed");
DEFINE_int32(numLevels, 5, "Number of levels in the pyramid");

int main(int argc, char *argv[])
{

	gflags::ParseCommandLineFlags(&argc, &argv, true);

	floorPlan = imread(FLAGS_floorPlan, 0);
	if(!floorPlan.data){
		cout << "Error reading floorPlan" << endl;
		exit(1);
	}

	fpColor = Mat (floorPlan.rows, floorPlan.cols, CV_8UC3, Scalar::all(255));
	for (int i = 0; i < fpColor.rows; ++i)
	{
		uchar * dst = fpColor.ptr<uchar>(i);
		uchar * src = floorPlan.ptr<uchar>(i);
		for (int j = 0; j < fpColor.cols; ++j)
		{
			if(src[j]!=255)
			{
				dst[j*3] = 0;
				dst[j*3+1] = 0;
				dst[j*3+2] = 0;
			}
		}
	}


	SparseMatrix<double> fpSparse (floorPlan.rows, floorPlan.cols);
	vector<Triplet<double> > tripletList;
	for (int i = 0; i < floorPlan.rows; ++i)
	{
		const uchar * src = floorPlan.ptr<uchar>(i);
		for (int j = 0; j < floorPlan.cols; ++j)
		{
			if(src[j] == 255)
				continue;
			tripletList.push_back(Triplet<double> (i,j, 1.0-(double)src[j]/255.0));
		}
	}
	fpSparse.setFromTriplets(tripletList.begin(), tripletList.end());
	tripletList.clear();
	
	vector<string> pointFileNames;
	vector<string> entropyFileNames;

	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (FLAGS_dmFolder.data())) != NULL) {
	  /* Add all the files and directories to a vector */
	  while ((ent = readdir (dir)) != NULL) {
	  	string fileName = ent->d_name;
	  	if(fileName != ".." && fileName != "." 
	  		&& fileName.find("point") != string::npos){
	  		pointFileNames.push_back(fileName);
	  	} else if(fileName != ".." && fileName != "." 
	  		&& fileName.find("entropy") != string::npos)
	  	{
	  		entropyFileNames.push_back(fileName);
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

	if(entropyFileNames.size() != rotationFileNames.size() || 
		entropyFileNames.size() != pointFileNames.size()){
		cout << "Not the same number of scans as rotations!" << endl;
		exit(-1);
	}

	sort(entropyFileNames.begin(), entropyFileNames.end());
	sort(rotationFileNames.begin(), rotationFileNames.end());
	sort(pointFileNames.begin(), pointFileNames.end());
	
	if(FLAGS_numScans == -1)
		FLAGS_numScans = pointFileNames.size()-FLAGS_startIndex;
	
	for(int i = FLAGS_startIndex; i< FLAGS_startIndex + FLAGS_numScans; ++i){
		const string scanName = FLAGS_dmFolder + pointFileNames[i];
		const string rotationFile = FLAGS_rotFolder + rotationFileNames[i];
		if(FLAGS_replace)
			analyzePlacement(fpSparse, scanName, rotationFile);
		else if(!reshowPlacement(scanName, rotationFile))
			analyzePlacement(fpSparse, scanName, rotationFile);
	}
	
	return 0;
}

void analyzePlacement(const SparseMatrix<double> & fpSparse,
	const string & scanName, const string & rotationFile){
	if(!FLAGS_quiteMode)
		cout << scanName << endl << rotationFile << endl;
	
	vector<SparseMatrix<double> > rSSparse;
	vector<Mat> rotatedScans;
	loadInScans(scanName, rotationFile, rotatedScans);
	scansToSparse(rotatedScans, rSSparse);

	
	vector<int> localMinima;
	vector<SparseMatrix<double> > fpPyramid;
	vector<vector<SparseMatrix<double> > > rSSparsePyramid;
	fpPyramid.push_back(fpSparse);
	rSSparsePyramid.push_back(rSSparse);

	createPyramid(fpPyramid, rSSparsePyramid);

	vector<vector<SparseMatrix<double> > > rSSparsePyramidTrimmed;
	trimScanPryamid(rSSparsePyramid, rSSparsePyramidTrimmed);

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
	
	vector<posInfo> scores;
	MatrixXd scoreMatrix;
	for (int k = FLAGS_numLevels; k > 0; --k)
	{
		findPlacementPointBasedV2(fpPyramid[k], rSSparsePyramidTrimmed[k], 
			scores, pointsToAnalyze);
		for (int i = 0; i < 4; ++i)
		{
			blurMinima(scoreMatrix, scores, 
			fpPyramid[k].rows() - rSSparsePyramidTrimmed[k][i].rows(), 
			fpPyramid[k].cols() - rSSparsePyramidTrimmed[k][i].cols(), i);
		}
		
		findLocalMinima(scores, localMinima, 1);
		findGlobalMinima(scores, localMinima);
		findPointsToAnalyze(scores, localMinima, pointsToAnalyze);
	}

	findPlacementPointBasedV2(fpPyramid[0], rSSparsePyramidTrimmed[0], 
			scores, pointsToAnalyze);
	for (int i = 0; i < 4; ++i)
		{
			blurMinima(scoreMatrix, scores, 
			fpPyramid[0].rows() - rSSparsePyramidTrimmed[0][i].rows(), 
			fpPyramid[0].cols() - rSSparsePyramidTrimmed[0][i].cols(), i);
		}
	findLocalMinima(scores, localMinima, 3);
	findGlobalMinima(scores, localMinima);

	if(FLAGS_visulization || FLAGS_previewOut)
		displayOutput(rSSparsePyramidTrimmed[0], localMinima, scores);

	const string placementName = FLAGS_preDone + scanName.substr(scanName.find("_")-3, 3) 
	+ "_placement_" + scanName.substr(scanName.find(".")-3, 3) + ".txt";
	savePlacement(scores, localMinima, placementName);
}

void loadInScans(const string & scanName, const string & rotationFile, 
	vector<Mat> & rotatedScans){
	ifstream binaryReader (rotationFile, ios::in | ios::binary);
	vector<Matrix3d> R (4);
	for (int i = 0; i < R.size(); ++i)
	{
		binaryReader.read(reinterpret_cast<char *>(&R[i]),
			sizeof(Matrix3d));
	}

	binaryReader.close();
	

	Mat scan = imread(scanName, 0);
	
	if(!scan.data){
		cout << "Error reading scan" << endl;
		exit(1);
	}
	const int maxDimension = max(1.1*scan.rows, 1.1*scan.cols);
	const int colOffset = (maxDimension - scan.cols)/2;
	const int rowOffset = (maxDimension - scan.rows)/2;
	croppedInfo.push_back(Vector4i (maxDimension, 0, 0, 0));

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

void scansToSparse(const vector<Mat> & rotatedScans, 
	vector<SparseMatrix<double> > & rSSparse){
	vector<Triplet<double> > tripletList;
	for(auto scan : rotatedScans){
		for (int i = 0; i < scan.rows; ++i)
		{
			uchar * src = scan.ptr<uchar>(i);
			for (int j = 0; j < scan.cols; ++j)
			{
				if(src[j] == 255)
					continue;
				double confidence = 1.0 -(double)src[j]/255.0;
				tripletList.push_back(Triplet<double> (i,j,confidence));
			}
		}
		SparseMatrix<double> sScan (scan.rows, scan.cols);
		sScan.setFromTriplets(tripletList.begin(), tripletList.end());
		

		rSSparse.push_back(sScan);
		tripletList.clear();
	}
}

void displayOutput(const vector<SparseMatrix<double> > & rSSparseTrimmed, 
	const vector<int> & localMinima, const vector<posInfo> & scores){
	if(!FLAGS_quiteMode)
	{
		cout << "Num localMinima: " << localMinima.size() << endl;
		cout << "Press a key to begin displaying placement options" << endl;
	}
	
	cvNamedWindow("Preview", WINDOW_NORMAL);
	imshow("Preview", floorPlan);
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
			cout << "NCC: " << minScore.NCC << endl;
		}
		
		waitKey(0);
		~output;
	}
}


void findLocalMinima(const vector<posInfo> & scores, vector<int> & localMinima, 
	const int bias){
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
	#pragma omp parallel for shared(i, scores) reduction(merge: localMinima) reduction(merge: globalMins)
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
		
		for (int j = 0; j < (currentFP.rows() -1); j+=2)
		{
			for (int k = 0; k < (currentFP.cols() -1); k+=2)
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

void trimScanPryamid(const vector<vector<SparseMatrix<double> > > & rSSparsePyramid,
	vector<vector<SparseMatrix<double> > > & rSSparsePyramidTrimmed){
	vector<Triplet<double> > tripletList;
	for(auto & level : rSSparsePyramid){
		vector<SparseMatrix<double> > levelTrimmed;
		for(auto & scan : level){

			SparseMatrix<double> threshHolded (scan.rows(), scan.cols());
			for (int k = 0; k < scan.outerSize(); ++k)
			{
				for (SparseMatrix<double>::InnerIterator it (scan, k); it; ++it)
				{
					if(it.value() > 0.85){
						tripletList.push_back(Triplet<double> (it.row(), it.col(), it.value()));
					}
				}
			}
			threshHolded.setFromTriplets(tripletList.begin(), tripletList.end());
			tripletList.clear();


			int minRow, minCol, maxRow, maxCol;
			maxRow = maxCol = 0;
			minRow = threshHolded.rows();
			minCol = threshHolded.cols();

			for (int k = 0; k < threshHolded.outerSize(); ++k)
			{
				for (SparseMatrix<double>::InnerIterator it (threshHolded, k); it; ++it)
				{
					if(it.value() !=0){
						maxRow = max(maxRow, it.row());
						minRow = min(minRow, it.row());

						maxCol = max(maxCol, it.col());
						minCol = min(minCol, it.col());
					}
				}
			}

			SparseMatrix<double> trimmed = threshHolded.block(minRow, minCol, 
				maxRow - minRow, maxCol - minCol);
			trimmed.makeCompressed();
			levelTrimmed.push_back(trimmed);
			croppedInfo.push_back(Vector4i (minRow, minCol, maxRow, maxCol));
		}
		rSSparsePyramidTrimmed.push_back(levelTrimmed);
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
	const vector<SparseMatrix<double> > & scans, vector<posInfo> & scores,
	const vector<Vector3i> & points){
	scores.clear();
	if(!FLAGS_quiteMode)
		cout << points.size() << endl;

	scores.reserve(points.size());


	#pragma omp parallel shared(fp, scans, points)
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

			double fpScore = currentFP.squaredNorm();

			SparseMatrix<double> diff = currentFP - scans[scanIndex];
			diff.makeCompressed();

			/*Mat test (diff.rows(), diff.cols(), CV_8UC1, Scalar::all(255));
			for (int i = 0; i < diff.outerSize(); ++i)
			{
				for(SparseMatrix<double>::InnerIterator it (diff, i); it; ++it)
				{
					test.at<uchar>(it.row(), it.col()) = 255 - abs(it.value())*255;
				}
			}
			cvNamedWindow("Preview", WINDOW_NORMAL);
			imshow("Preview", test);
			waitKey(0);*/

			double diffScore = diff.squaredNorm();
			
			double diffEntropy = 0;
			for(int i = 0; i < diff.outerSize(); ++i){
				for(SparseMatrix<double>::InnerIterator it (diff, i); it; ++it){
					if(it.value() != 0){
						const double tmp = it.value()*it.value()/diffScore;
						diffEntropy -= tmp*log(tmp);
						
					}	
				}
			}


			

			MatrixXd currentFPNS = MatrixXd(currentFP);
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
			const double nccScore = pq/sqrt(pp*qq);	

			const double placement = (diffScore - fpScore);
			const double score = (placement > 0) ? placement*diffEntropy
				: placement/diffEntropy;

			posInfo tmp;
			tmp.x = point[0];
			tmp.y = point[1];
			tmp.score = score;
			tmp.rotation = scanIndex;
			tmp.entropy = diffEntropy;
			tmp.NCC = nccScore;
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
	outB.write(reinterpret_cast<const char *>(&numLoc), sizeof(int));
	outB.write(reinterpret_cast<const char *>(&numGlob), sizeof(int));


	for(auto index : localMinima){
		posInfo minScore = scores[index];
		out << minScore.score << " " << minScore.x  << " "
			<< minScore.y  << " " << minScore.rotation
			<< " " << minScore.NCC << endl;
		outB.write(reinterpret_cast<const char *> (&minScore), sizeof(posInfo));
	}
	out << endl;
	for(auto index : globalMins){
		posInfo minScore = scores[index];
		out << minScore.score << " " << minScore.x  << " "
			<< minScore.y  << " " << minScore.rotation
			<< " " << minScore.NCC << endl;
		outB.write(reinterpret_cast<const char *> (&minScore), sizeof(posInfo));
	}
	out.close();
	outB.close();
	croppedInfo.clear();
}

bool reshowPlacement(const string & scanName, const string & rotationFile){
	const string placementName = FLAGS_preDone + scanName.substr(scanName.find("_")-3, 3) 
	+ "_placement_" + scanName.substr(scanName.find(".")-3, 3) + ".dat";
	ifstream in (placementName, ios::in | ios::binary);
	if(!in.is_open())
		return false;
	if(!FLAGS_previewOut)
		return true;
	vector<Mat> rotatedScans;
	loadInScans(scanName, rotationFile, rotatedScans);
	int numScans;
	in.read(reinterpret_cast<char *>(&numScans), sizeof(int));

	cout << "showing localMinima" << endl;
	for (int i = 0; i < numScans; ++i)
	{
		posInfo minScore;
		in.read(reinterpret_cast<char *>(&minScore), sizeof(posInfo));
		Mat bestScan = rotatedScans[minScore.rotation];

	}

	return true;
}

bool notInLocalMin(const int i, const vector<int> & localMinima){
	if(localMinima.size() == 0)
		return true;
	if(localMinima.size() == 1)
		return !(i == localMinima[0]);

	for(auto & num : localMinima){
		if(i == num)
			return false;
	}
	return true;

	/*if(i == localMinima[localMinima.size()/2]){
		return false;
	}else if( i < localMinima[localMinima.size()/2]){
		return notInLocalMin(i, 
			vector<int>(localMinima.begin(), localMinima.begin() + localMinima.size()/2));
	} else{
		return notInLocalMin(i, 
			vector<int>(localMinima.begin() + localMinima.size()/2, localMinima.end()));
	}*/
}

void blurMinima(MatrixXd & scoreMatrix, const vector<posInfo> & scores, 
	int rows, int cols, int rotNumber){
	
	scoreMatrix = MatrixXd::Zero(rows, cols);
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
		if(minScore.rotation == rotNumber)
			scoreMatrix(minScore.y, minScore.x) = minScore.score;
	}

	MatrixXd blurredScore (scoreMatrix);
	for (int i = 2; i < scoreMatrix.rows() - 2; ++i)
	{
		for (int j = 2; j < scoreMatrix.cols() - 2; ++j)
		{
			const MatrixXd toBlur = scoreMatrix.block(i-2, j-2, 5, 5);
			const double * blurPtr = toBlur.data();
			double value = 0;
			for (int i = 0; i < kernel.size(); ++i)
			{
				value += (*(kernlPtr + i))*(*(blurPtr + i));
			}
			blurredScore(i,j) = value;
		}
	}
	scoreMatrix = blurredScore;

	double aveScore = 0;
	int count = 0;
	const double * scoreMatrixPtr =scoreMatrix.data();
	for (int i = 0; i < scoreMatrix.size(); ++i)
	{
		if(*(scoreMatrixPtr + i) != 0)
		{
			aveScore += *(scoreMatrixPtr + i);
			count ++;
		}
	}
	aveScore /= count;
	double sigScores = 0;
	
	for (int i = 0; i <scoreMatrix.size(); ++i)
	{
		if(*(scoreMatrixPtr + i) != 0)
		{
			const double tmp = *(scoreMatrixPtr + i) - aveScore;
			sigScores += tmp*tmp;
		}
		
	}
	sigScores /= count;
	sigScores = sqrt(sigScores);


	if(FLAGS_visulization)
	{
		Mat blurScore (scoreMatrix.rows(), scoreMatrix.cols(), CV_8UC1, Scalar::all(255));
		for (int i = 0; i < blurScore.rows; ++i)
		{
			uchar * dst = blurScore.ptr<uchar>(i);
			for (int j = 0; j < blurScore.cols; ++j)
			{
				if(scoreMatrix(i,j) != 0)
				{
					const int gray = max(0, min(255,
						 static_cast<int>(255.0 * (
						 	((scoreMatrix(i,j) - aveScore)/(2.0*sigScores) + 1.0)/2.0))));
					dst[j] = 255 - gray;
				}
			}
		}
		cvNamedWindow("Preview", WINDOW_NORMAL);
		imshow("Preview", blurScore);
		waitKey(0);
	}	
}