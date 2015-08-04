#include <opencv2/core.hpp>
#include "opencv2/imgproc.hpp"
#include <stdio.h>
#include <iostream>
#include <string>
#include <dirent.h>
#include <vector>
#include <fstream>
#include <sstream>

#define NUM_ENTRY_PER_LINE 19
#define PI 3.14159
#define FPS 15

static float PANO_H = 1024*3;

#define max(a,b) ((a > b) ? a : b)
#define min(a,b) ((a < b) ? a : b)


using namespace std;
using namespace cv;




Mat readInRotation(ifstream &, float );
void csvToBinary (ifstream &, ofstream &);
int project_v_to_vp (int ,  Mat & );
int project_u_to_up(int, int , Mat & );
void image_coords_to_pano_coords(float *, float *, Mat &);
void pano_coords_to_image_coords(float *, float *, Mat &);

Mat K;

double maxT, maxP, minT, minP;


int main(int argc, char const *argv[])
{
	if(argc != 2){
		cout << "Usage: test.out <IMU_File.txt> " <<endl;
		return -1;
	}

	K = (Mat_<float>(3,3) << 3867.214290/3.25, 0, 1220.329623/3.25, 
		0, 3850.296435/3.25, 2077.400229/3.25, 0, 0, 1);

	maxT = maxP = 0;
	minT = minP = 0;

	cout << "K = " << endl << " " << K << endl << endl;

	const char * imuFile = argv[1];

	ifstream csvFile (imuFile, ios::in );

	ofstream binaryFileOut("output.dat",  ios::out | ios::binary);

	csvToBinary(csvFile, binaryFileOut);
	csvFile.close();
	binaryFileOut.close();
	

	ifstream binaryFile("output.dat",  ios::in | ios::binary);



	Mat R = readInRotation(binaryFile, 1/15.0);

	cout << "R = " << endl << " " << R << endl << endl;

	float img [3] = {1280, 720, 0};
	float pano [3];

	image_coords_to_pano_coords(img, pano, R);

	cout << "Image: ";
	for(float f : img){
		cout << f << ",";
	}

	cout << endl;
	cout << "Pano: ";
	for(float f : pano){
		cout << f << ",";
	}

	cout << endl;


	pano_coords_to_image_coords(pano, img, R);

	cout << "Image: ";
	for(float f : img){
		cout << f << ",";
	}

	cout << endl;
	cout << "Pano: ";
	for(float f : pano){
		cout << f << ",";
	}

	cout << endl;



	float corner1 [3], corner2 [3], corner3 [3], corner4 [3];
	float imageCoords [2] = {0,0};

	image_coords_to_pano_coords(imageCoords, corner1, R);

	imageCoords [0] = 720;
	image_coords_to_pano_coords(imageCoords, corner2, R);

	imageCoords[1] = 1280;
	image_coords_to_pano_coords(imageCoords, corner3, R);

	imageCoords[0] = 0;
	image_coords_to_pano_coords(imageCoords, corner4, R);

	cout << "Corner1 : ";
	for(float f : corner1){
		cout << f << ",";
	}

	cout <<endl;

	cout << "Corner2 : ";
	for(float f : corner2){
		cout << f << ",";
	}

	cout <<endl;
	cout << "Corner3 : ";
	for(float f : corner3){
		cout << f << ",";
	}

	cout <<endl;
	cout << "Corner4 : ";
	for(float f : corner4){
		cout << f << ",";
	}

	cout <<endl;


	
	binaryFile.close();
	return 0;
}


Mat readInRotation(ifstream & file, float  timeSamp){

	cout << "Reading the file" << endl;
	float time1, time2;
	file.seekg(0);


	file.read(reinterpret_cast<char *> (& time1), sizeof(float));

	int i = 1;
	bool run  = true;
	while(run){
		file.seekg(sizeof(float) * NUM_ENTRY_PER_LINE *i);
		file.read(reinterpret_cast<char *> (& time2), sizeof(float));

		if(time1 <= timeSamp && time2 >= timeSamp)
			run = false;
		else
			time1 = time2;

		++i;
	}
	
	cout << "Time 1: " << time1 << endl << "Time 2: " << time2 << endl;
	Mat rot_matrix (3,3, CV_32F);


	file.read(reinterpret_cast< char *> (&rot_matrix.at<float>(0,0)), sizeof(float)*3);
	file.read(reinterpret_cast< char *> (&rot_matrix.at<float>(1,0)), sizeof(float)*3); 
	file.read(reinterpret_cast< char *> (&rot_matrix.at<float>(2,0)), sizeof(float)*3);


	
	return rot_matrix.t();


	
}


void csvToBinary(ifstream & csv, ofstream & binary){
	string line;
	getline(csv, line); //dump the header

	while(getline(csv, line)){
		
		size_t pos1, pos2;
		pos1 = pos2 = 0;

		pos1 = line.find("\t");
		float number = stof (line.substr(0, pos1));
		binary.write(reinterpret_cast<const char *> (& number), sizeof(float));


		while((pos2 = line.find("\t", pos1 +1)) != string::npos){
			number = stof (line.substr(pos1 + 1, pos2 - pos1 -1));
			binary.write(reinterpret_cast<const char *> (& number), sizeof(float));

			pos1 = pos2;
		}

		number = stof (line.substr(pos1 +1));
		binary.write(reinterpret_cast<const char *> (& number), sizeof(float));
		
	} 



}


int project_v_to_vp (int v, Mat & R){
	int u = 0;

	float xi = u ;
	float yi = v ;



	Mat image_coords = (Mat_<float>(3,1) << xi, yi, 1);

	Mat world_coords = R.t() * K.inv() * image_coords;


	double r = world_coords.at<float>(0) * world_coords.at<float>(0) 
		+ world_coords.at<float>(1) * world_coords.at<float>(1) 
		+ world_coords.at<float>(2) * world_coords.at<float>(2);


	r = sqrt(r);

	double phi = acos(world_coords.at<float>(2,0)/r);




	return (int) (phi* PANO_H );
}


int project_u_to_up(int u, int v, Mat & R){
	float xi = u ;
	float yi = v ;




	Mat image_coords = (Mat_<float>(3,1) << xi, yi, 1);

	Mat world_coords = R.t() * K.inv() * image_coords;



	double theta = 
		atan2(world_coords.at<float>(1,0),world_coords.at<float>(0,0)) + PI;



	return (int) (theta* PANO_H );
}


void image_coords_to_pano_coords(float * img_coords, float * pano_coords , Mat & R){

	Mat image_coords = (Mat_<float>(3,1) << img_coords[0], img_coords[1], 1);

	Mat world_coords = R * K.inv() * image_coords;



	float r = world_coords.at<float>(0) * world_coords.at<float>(0) 
		+ world_coords.at<float>(1) * world_coords.at<float>(1) 
		+ world_coords.at<float>(2) * world_coords.at<float>(2);


	r = sqrt(r);

	float phi = acos(world_coords.at<float>(2)/r);

	float theta = 
		atan2(world_coords.at<float>(1),world_coords.at<float>(0)) + PI;

	pano_coords [0] = theta/PI * PANO_H;
	pano_coords [1] = phi/PI * PANO_H;
	pano_coords [2] = r;

	
}


void pano_coords_to_image_coords(float * pano_coords, float * img_coords, Mat & R){
	float theta = pano_coords[0]*PI/PANO_H - PI;
	float phi = pano_coords[1]*PI/PANO_H;
	float x = pano_coords[2] * cos(theta) * sin(phi);
	float y = pano_coords[2] * sin(theta) * sin(phi);
	float z = pano_coords[2] * cos(phi);

	Mat world_coords = (Mat_<float>(3,1) << x, y, z);


	Mat image_coords = K * R.t() * world_coords;

	img_coords [0] = image_coords.at<float>(0);
	img_coords [1] = image_coords.at<float>(1);
	img_coords [2] = image_coords.at<float>(2);

	
}