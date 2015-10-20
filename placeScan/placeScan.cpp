#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <Eigen/Sparse>
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
	int rotation = 0;
	int x = 0;
	int y = 0;
	
} posInfo;

#pragma omp declare reduction (merge : std::vector<posInfo> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp declare reduction (merge : std::vector<int> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))



void analyzePlacement(const SparseMatrix<double> &, const string &, const string &);
void findPlacementPointBased(const SparseMatrix<double> &
	, const vector<SparseMatrix<double> > &, vector<posInfo> &);
void findLocalMinima(const vector<posInfo> &, vector<int> &);
void findLocalMinimaIntermediary(const vector<posInfo> &, vector<int> &);
void createPyramid(const SparseMatrix<double> &, const vector<SparseMatrix<double> > &,
	vector<posInfo> &, vector<int> &);
void findPlacementPointBasedV2(const SparseMatrix<double> &, 
	const vector<SparseMatrix<double> > &, vector<posInfo> &, const vector<Vector3i> &);
void findPointsToAnalyze(const vector<posInfo> &, const vector<int> &, vector<Vector3i> &);

ofstream positionsFile ("positions.txt", ios::out);
Mat floorPlan;
int numScansToProcess = 0;
int startIndex = 0;
vector<Vector4i> croppedInfo;
int numLevels = 3;
vector<int> globalMins;

int main(int argc, char const *argv[])
{
	if(argc !=4 && argc !=5 && argc!=6){
		cout << "Usage: placeScan.o <floorPlan>.png <scansFolder>/ <scanRotationsFolder>/" << endl;
		cout << " Optional:<startIndex> Optional:<numberofScans>" << endl;
		return -1;
	}

	if(argc >=5){
		string tmp = argv[4];
		startIndex = stoi(tmp);
	}
	if(argc = 6){
		string tmp = argv[5];
		numScansToProcess = stoi(tmp);
	}

	floorPlan = imread(argv[1], 0);
	if(!floorPlan.data){
		cout << "Error reading floorPlan" << endl;
		exit(1);
	}

	positionsFile << "Scan Name, Rotation Number, xOffset, yOffset" << endl;

	/*Mat croppedFloorPlan (floorPlan.rows-500, floorPlan.cols-3000, CV_8UC1);
	for (int i = 0; i < floorPlan.rows-500; ++i)
	{
		uchar * src = floorPlan.ptr<uchar>(i);
		uchar * dst = croppedFloorPlan.ptr<uchar>(i);
		for (int j = 1500; j < floorPlan.cols-1500; ++j)
		{
			dst[j-1500] = src[j];
		}
	}

	cvNamedWindow("Preview", WINDOW_NORMAL);
	imshow("Preview", croppedFloorPlan);
	waitKey(0);
	imwrite("DUC-floor-1_cropped.png", croppedFloorPlan);
	exit(0);*/

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

	/*Mat test (fpSparse.rows(), fpSparse.cols(), CV_8UC1, Scalar::all(255));
	for (int i = 0; i < fpSparse.outerSize(); ++i)
	{
		for(SparseMatrix<double>::InnerIterator it (fpSparse, i); it; ++it)
		{
			test.at<uchar>(it.row(), it.col()) = 255 - it.value()*255;
		}

	}
	cvNamedWindow("Preview", WINDOW_NORMAL);
	imshow("Preview", test);
	waitKey(0);*/

	
	
	
	vector<string> pointFileNames;
	vector<string> entropyFileNames;
	const char * scanFolder = argv[2];
	const char * rotationsFolder = argv[3]; 

	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (scanFolder)) != NULL) {
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
	if ((dir = opendir (rotationsFolder)) != NULL) {
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
	if(numScansToProcess == 0)
	{
		for(int i = startIndex; i< pointFileNames.size(); ++i){
			const string scanName = scanFolder + pointFileNames[i];
			const string rotationFile = rotationsFolder + rotationFileNames[i];
			analyzePlacement(fpSparse, scanName, rotationFile);
		}
	} else{
		for(int i = startIndex; i< numScansToProcess + startIndex; ++i){
			const string scanName = scanFolder + pointFileNames[i];
			const string rotationFile = rotationsFolder + rotationFileNames[i];
			analyzePlacement(fpSparse, scanName, rotationFile);
		}
	}

	
	
	

	return 0;
}

void analyzePlacement(const SparseMatrix<double> & fpSparse,
	const string & scanName, const string & rotationFile){
	cout << scanName << endl << rotationFile << endl;

	ifstream binaryReader (rotationFile, ios::in | ios::binary);
	vector<Matrix3d> R (4);
	for (int i = 0; i < R.size(); ++i)
	{
		binaryReader.read(reinterpret_cast<char *>(&R[i]),
			sizeof(Matrix3d));
	}

	binaryReader.close();
	

	Mat scan = imread(scanName, 0);
	vector<Mat> rotatedScans;
	vector<SparseMatrix<double> > rSSparse;
	const Vector3d center (scan.cols/2.0, scan.rows/2.0, 0.0);



	if(!scan.data){
		cout << "Error reading scan" << endl;
		exit(1);
	}

	for(auto & rot : R)
	{
		Mat rScan (scan.rows, scan.cols, CV_8UC1, Scalar::all(255));
		for (int i = 0; i < scan.rows; ++i)
		{
			uchar * dst = rScan.ptr<uchar>(i);
			for (int j = 0; j < scan.cols; ++j)
			{
				Vector3d pixel (j, i, 0.0);
				Vector3d src = rot*(pixel-center) + center;
				if(src[0] < 0 || src[0] >= scan.cols )
					continue;
				if(src[1] < 0 || src[1] >= scan.rows)
					continue;
				dst[j] = scan.at<uchar>(src[0], src[1]);
			}
		}
		rotatedScans.push_back(rScan);
	}

	/*for(auto & scan : rotatedScans){
		cvNamedWindow("Preview", WINDOW_NORMAL);
		imshow("Preview", scan);
		waitKey(0);
	}*/

	vector<Triplet<double> > tripletList;
	for(auto & scan : rotatedScans){
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
			
	/*for(auto & sparse : rSSparse){
		Mat test (sparse.rows(), sparse.cols(), CV_8UC1, Scalar::all(255));

		for (int k = 0; k < sparse.outerSize(); ++k)
		{
			for (SparseMatrix<double>::InnerIterator it (sparse, k); it; ++it)
			{
				test.at<uchar>(it.row(), it.col()) = 255 - it.value()*255;
			}
		}
		cvNamedWindow("Preview", WINDOW_NORMAL);
		imshow("Preview", test);
		waitKey(0);
	}*/
		
	vector<posInfo> scores;
	vector<int> localMinima;

	createPyramid(fpSparse, rSSparse, scores, localMinima);
	


	vector<SparseMatrix<double> > rSSparseTrimmed;
	for(auto & scan : rSSparse){

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
		rSSparseTrimmed.push_back(trimmed);
	}
	
	/*findPlacementPointBased(fpSparse, rSSparseTrimmed, scores);
	findLocalMinima(scores, localMinima);*/

	cout << "Num localMinima: " << localMinima.size() << endl;
	cout << "Press a key to begin displaying placement options" << endl;
	cvNamedWindow("Preview", WINDOW_NORMAL);
	imshow("Preview", floorPlan);
	waitKey(0);

	

	
	for(int & index : localMinima){
		posInfo minScore = scores[index];
		const int xOffset = minScore.x;
		const int yOffset = minScore.y;
		const SparseMatrix<double> & currentScan = rSSparseTrimmed[minScore.rotation]; 
		Mat output (floorPlan.rows, floorPlan.cols, CV_8UC1);
		floorPlan.copyTo(output);
		for (int i = 0; i < currentScan.outerSize(); ++i)
		{
			for(SparseMatrix<double>::InnerIterator it(currentScan, i); it; ++it){
				output.at<uchar>(it.row() + yOffset, it.col() + xOffset) 
					= 255 - 255*it.value();
			}
		}

		if(minScore.x > 1210 && minScore.x < 1225 && minScore.y > 385 && minScore.y < 395)
		{
			imshow("Preview", output);
			cout << minScore.score <<"   " << minScore.entropy << "      " << minScore.x << "      " <<minScore.y << endl;
			waitKey(0);
		} else 
		{
			imshow("Preview", output);
			cout << minScore.score <<"      " << minScore.x << "      " <<minScore.y << endl;
			waitKey(1);
		}
		
		~output;
	}



	cout << "Num globalMins: " << globalMins.size() << endl;
	for(auto & globMin : globalMins)
	{
		posInfo minScore = scores[globMin];
		const int xOffset = minScore.x;
		const int yOffset = minScore.y;
		/*if(yOffset == 0)
			continue;*/
		const SparseMatrix<double> & currentScan = rSSparseTrimmed[minScore.rotation]; 
		Mat output (floorPlan.rows, floorPlan.cols, CV_8UC1);
		floorPlan.copyTo(output);
		for (int i = 0; i < currentScan.outerSize(); ++i)
		{
			for(SparseMatrix<double>::InnerIterator it(currentScan, i); it; ++it){
				output.at<uchar>(it.row() + yOffset, it.col() + xOffset) 
					= 255 - 255*it.value();
			}
		}


		imshow("Preview", output);
		cout << minScore.score <<"      " << minScore.x << "      " <<minScore.y << endl;
		waitKey(0);
		~output;
		Vector4i cropped = croppedInfo[minScore.rotation];

		positionsFile << scanName << ", " << minScore.rotation << ", " 
			<< xOffset - cropped[0] << ", " << yOffset - cropped[1] << endl;
	}
	

	

}



void findPlacementPointBased(const SparseMatrix<double> & fp, 
	const vector<SparseMatrix<double> > & scans, vector<posInfo> & scores){
		
	int k;
	#pragma omp for
	for(k = 0; k< 1/*scans.size()*/; ++k)
	{
		const int xStop = fp.cols() - scans[k].cols();
		const int yStop = fp.rows() - scans[k].rows();
		scores.reserve(xStop*yStop);
		cout << k << "    " << xStop << "    " << yStop << endl;
		int xOffset;
		#pragma omp parallel for shared(fp, scans, k, xOffset) reduction(merge: scores)
		for (xOffset = 500; xOffset < 2500/*xStop*/; xOffset++)
		{
			
			for (int yOffset = 0; yOffset < 500/*yStop*/; yOffset++)
			{
				SparseMatrix<double> currentFP = fp.block(yOffset, xOffset, 
					scans[k].rows(), scans[k].cols());
				SparseMatrix<double> diff = currentFP - scans[k];

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

				double score = diff.squaredNorm();
				double fpScore = currentFP.squaredNorm();

				double diffEntropy = 0;
				for(int i = 0; i < diff.outerSize(); ++i){
					for(SparseMatrix<double>::InnerIterator it (diff, i); it; ++it){
						if(it.value() != 0){
							const double tmp = it.value()*it.value()/score;
							diffEntropy -= tmp*log(tmp);
						}	
					}
				}

				posInfo tmp;
				tmp.x = xOffset;
				tmp.y = yOffset;
				tmp.score = diffEntropy*(score - fpScore);
				tmp.rotation = k;
				tmp.entropy = diffEntropy;
				scores.push_back(tmp);
			}

			if(xOffset % (xStop/10) == 0){
				cout << xOffset << ".." << flush;
			}
		}
		cout << endl;
	}

	cout << "Done" << endl;
}


void findLocalMinima(const vector<posInfo> & scores, vector<int> & localMinima){
	localMinima.clear();
	globalMins.clear();
	double averageScore = 0;
	double minScore = 5e30;
	ofstream csv ("output.txt", ios::out);
	for(auto & info : scores){
		averageScore += info.score;
		minScore = min(info.score, minScore);
		csv << info.x << "," << info.y << "," << info.score << endl;
	}
	csv.close();
	averageScore /= scores.size();
	double sigScores = 0;
	for(auto & info : scores){
		sigScores += (info.score - averageScore)*(info.score - averageScore);
	}
	sigScores /= (scores.size() - 1);
	sigScores = sqrt(sigScores);
	cout << "Average         Sigma" << endl;
	cout << averageScore << "         " << sigScores << endl; 
	cout << "Min: " << minScore << endl;

	if(abs(scores[0].score) <= abs(1.01*minScore))
			globalMins.push_back(0);

	/*if(abs(scores[1].score) <= abs(1.1*minScore))
		globalMins.push_back(1);*/

	const double cutOff = averageScore - 3.0*sigScores;
	int i;
	#pragma omp parallel for shared(i, scores) reduction(merge: localMinima) reduction(merge: globalMins)
	for (i = 1; i < (scores.size() - 1); ++i)
	{
		double lHS = scores[i-1].score - scores[i].score;
		double rHS = scores[i].score - scores[i+1].score;
		if( lHS >= 0 && rHS <= 0 && scores[i].score < cutOff)
			localMinima.push_back(i);
		/*if(scores[i].score < (averageScore -1.5*sigScores))
			localMinima.push_back(i);*/

		if(abs(scores[i].score) < abs(1.01* minScore))
			globalMins.push_back(i);
	}

	/*if(abs(scores[scores.size() - 2].score) <= abs(1.01*minScore))
			globalMins.push_back(scores.size() - 2);*/

	if(abs(scores[scores.size()-1].score) <= abs(1.01*minScore))
			globalMins.push_back(scores.size()-1);
}

void createPyramid(const SparseMatrix<double> & fp, const vector<SparseMatrix<double> >& rSSparse, 
	vector<posInfo> & scores, vector<int> & localMinima){

	vector<SparseMatrix<double> > fpPyramid;
	fpPyramid.push_back(fp);
	vector<vector<SparseMatrix<double> > > rSSparsePyramid;
	rSSparsePyramid.push_back(rSSparse);
	vector<Triplet<double> > tripletList;
	
	for (int i = 0; i < numLevels; ++i)
	{
		SparseMatrix<double> & currentFP = fpPyramid[i];
		SparseMatrix<double> fpLevel (floor(currentFP.rows()/2), 
			floor(currentFP.cols()/2));

		MatrixXd currentFPNS = MatrixXd(currentFP);
		
		for (int j = 0; j < (currentFP.rows() -1); j+=2)
		{
			for (int k = 0; k < (currentFP.cols() -1); k+=2)
			{
				/*double average = (currentFPNS(j,k) + currentFPNS(j,k+1) +
					currentFPNS(j+1,k) + currentFPNS(j+1,k+1))/4.0;
				if(average >= 0.5)
				{
					tripletList.push_back(Triplet<double> (j/2, k/2, 1.0));
				}
				else if(average == 0.5)
				{
					int value = rand()%2;
					tripletList.push_back(Triplet<double> (j/2, k/2, value));
				}*/

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
					/*double average = (scanNS(j,k) + scanNS(j,k+1) +
						scanNS(j+1,k) + scanNS(j+1,k+1))/2.0;*/

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

	/*for(auto & level: rSSparsePyramid){
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
	}*/

	


	vector<vector<SparseMatrix<double> > > rSSparsePyramidTrimmed;
	double levelNum = 8;
	for(auto & level : rSSparsePyramid){
		vector<SparseMatrix<double> > levelTrimmed;
		for(auto & scan : level){

			SparseMatrix<double> threshHolded (scan.rows(), scan.cols());
			for (int k = 0; k < scan.outerSize(); ++k)
			{
				for (SparseMatrix<double>::InnerIterator it (scan, k); it; ++it)
				{
					if(it.value() > 0.85/*/(levelNum/8)*/){
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
		levelNum++;
	}
	


	/*for(auto & level: rSSparsePyramidTrimmed){
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
	}*/



	vector<Vector3i> pointsToAnalyze;
	for(int k = 0; k < 1/*rSSparse.size()*/; ++k)
	{
		const int xStop = fpPyramid[numLevels].cols() 
		- rSSparsePyramidTrimmed[numLevels][k].cols();
		const int yStop = fpPyramid[numLevels].rows()
		- rSSparsePyramidTrimmed[numLevels][k].rows();
		pointsToAnalyze.reserve(xStop*yStop*(k+1));
		for (int i = 0; i < xStop; ++i)
		{
			for (int j = 0; j < yStop; ++j)
			{
				pointsToAnalyze.push_back(Vector3i (i,j,k));
			}
		}
	}
	
	

	for (int k = numLevels; k > 0; --k)
	{
		findPlacementPointBasedV2(fpPyramid[k], rSSparsePyramidTrimmed[k], 
			scores, pointsToAnalyze);
		findLocalMinimaIntermediary(scores, localMinima);
		findPointsToAnalyze(scores, localMinima, pointsToAnalyze);

		Mat currentFP (fpPyramid[k].rows(), fpPyramid[k].cols(), CV_8UC1, Scalar::all(255));
		for (int i = 0; i < fpPyramid[k].outerSize(); ++i)
		{
			for(SparseMatrix<double>::InnerIterator it (fpPyramid[k], i); it; ++it){
				currentFP.at<uchar>(it.row(), it.col()) = 255 - 255*it.value();
			}
		}

		for(auto & minScore : scores){
			if(minScore.x >= 1200.0/(2*(k+1)) && minScore.x <= 1250.0/(2*(k+1)) 
				&& minScore.y >= 300.0/(2*(k+1)) && minScore.y <= 350.0/(2*(k+1))
				&& minScore.rotation == 0)
			{
				cout << minScore.score <<"   " << minScore.entropy 
				<< "      " << minScore.x << "      " <<minScore.y 
				<< "   "  << minScore.rotation << endl;

				const int xOffset = minScore.x;
				const int yOffset = minScore.y;
				/*if(yOffset == 0)
					continue;*/
				const SparseMatrix<double> & currentScan = rSSparsePyramidTrimmed[k][minScore.rotation]; 
				Mat output (currentFP.rows, currentFP.cols, CV_8UC1);
				currentFP.copyTo(output);
				for (int i = 0; i < currentScan.outerSize(); ++i)
				{
					for(SparseMatrix<double>::InnerIterator it(currentScan, i); it; ++it){
						output.at<uchar>(it.row() + yOffset, it.col() + xOffset) 
							= 255 - 255*it.value();
					}
				}

				cvNamedWindow("Preview", WINDOW_NORMAL);
				imshow("Preview", output);
				waitKey(0);
				~output;
			}
		}
	}

	findPlacementPointBasedV2(fpPyramid[0], rSSparsePyramidTrimmed[0], 
			scores, pointsToAnalyze);
	findLocalMinima(scores, localMinima);

}

void findPlacementPointBasedV2(const SparseMatrix<double> & fp, 
	const vector<SparseMatrix<double> > & scans, vector<posInfo> & scores,
	const vector<Vector3i> & points){
	scores.clear();
	cout << points.size() << endl;
	scores.reserve(points.size());

	int i;

	#pragma omp parallel for shared(fp, scans, points, i) reduction(merge: scores)
	for(i = 0; i< points.size(); ++i)
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

		double score = diff.squaredNorm();
		

		double diffEntropy = 0;
		for(int i = 0; i < diff.outerSize(); ++i){
			for(SparseMatrix<double>::InnerIterator it (diff, i); it; ++it){
				if(it.value() != 0){
					const double tmp = it.value()*it.value()/score;
					diffEntropy -= tmp*log(tmp);
					
				}	
			}
		}

		/*double fpEntropy = 0;
		for(int i = 0; i < currentFP.outerSize(); ++i){
			for(SparseMatrix<double>::InnerIterator it (currentFP, i); it; ++it){
				if(it.value() != 0){
					const double tmp = it.value()*it.value()/fpScore;
					fpEntropy -= tmp*log(tmp);
					
				}	
			}
		}*/


		

		posInfo tmp;
		tmp.x = point[0];
		tmp.y = point[1];
		tmp.score = (score - fpScore/2.0)*diffEntropy;
		tmp.rotation = scanIndex;
		tmp.entropy = diffEntropy;
		scores.push_back(tmp);
		

	}

	/*double aveE, sigE, aveS, sigS;
	aveE = sigE = aveS = sigS = 0;
	for(auto & info : scores){
		aveE += info.entropy;
		aveS += info.score;
	}
	aveE /= scores.size();
	aveS /= scores.size();

	for(auto & info : scores){
		const double tmp = info.entropy-aveE;
		sigE += tmp*tmp;
		const double tmp2 = info.score - aveS;
		sigS += tmp2*tmp2;
	}
	sigE /= scores.size();
	sigE = sqrt(sigE);
	sigS /= scores.size();
	sigS = sqrt(sigS);

	for(auto & info : scores){
		info.entropy = (info.entropy - aveE)/(sigE) + 10.0;
		info.score = (info.score - aveS)/(sigS) + 10.0;
	}

	for(auto & info : scores){
		info.score *= info.entropy;
	}*/


	cout << "Done: " << scores.size() << endl;
}

void findPointsToAnalyze(const vector<posInfo> & scores, const vector<int> & localMinima,
	vector<Vector3i> & pointsToAnalyze){
	pointsToAnalyze.clear();
	pointsToAnalyze.reserve(localMinima.size()*4 + globalMins.size()*4);

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

void findLocalMinimaIntermediary(const vector<posInfo> & scores, vector<int> & localMinima){
	localMinima.clear();
	globalMins.clear();
	double averageScore = 0;
	double minScore = 5e30;
	ofstream csv ("output.txt", ios::out);
	for(auto & info : scores){
		averageScore += info.score;
		minScore = min(info.score, minScore);
		csv << info.x << "," << info.y << "," << info.score << endl;
	}
	csv.close();
	averageScore /= scores.size();
	double sigScores = 0;
	for(auto & info : scores){
		sigScores += (info.score - averageScore)*(info.score - averageScore);
	}
	sigScores /= (scores.size() - 1);
	sigScores = sqrt(sigScores);
	cout << "Average         Sigma" << endl;
	cout << averageScore << "         " << sigScores << endl; 
	cout << "Min: " << minScore << endl;

	if(abs(scores[0].score) <= abs(1.01*minScore))
			globalMins.push_back(0);

	const double cutOff = averageScore - 1.0*sigScores;
	int i;
	#pragma omp parallel for shared(i, scores) reduction(merge: localMinima) reduction(merge: globalMins)
	for (i = 1; i < (scores.size() - 1); ++i)
	{
		const double lHS = scores[i-1].score - scores[i].score;
		const double rHS = scores[i].score - scores[i+1].score;
		if( lHS >= 0 && rHS <= 0 && scores[i].score < cutOff)
			localMinima.push_back(i);
		
		if(abs(scores[i].score) <= abs(1.01*minScore))
			globalMins.push_back(i);
	}

	if(abs(scores[scores.size()-1].score) <= abs(1.01*minScore))
			globalMins.push_back(scores.size()-1);
}
