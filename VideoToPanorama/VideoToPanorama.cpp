#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <stdio.h>
#include <iostream>
#include <string>
#include <dirent.h>
#include <vector>
#include <fstream>
#include <math.h>
#include <time.h>


#define NUM_ENTRY_PER_LINE 19
#define PI 3.14159
#define FPS 7


constexpr float PANO_H = 1024;

using namespace std;
using namespace cv;



Mat readInRotation(ifstream &, float );
void csvToBinary (ifstream &, ofstream &);
int projectImageToPanorama(string &, ifstream &);
void image_coords_to_pano_coords(float *, float *, Mat &);
void pano_coords_to_image_coords(float *, float *, Mat &);
inline float max (float a, float b) {return (a > b) ? a : b;}
inline float min (float a, float b) {return (a < b) ? a : b;}




Mat panorama (PANO_H, PANO_H*2, CV_8UC3, Scalar::all(255));
Mat K;

int main(int argc, char** argv)
{
	//Moto X K
	K = (Mat_<float>(3,3) << 3867.214290/3.25, 0, 1220.329623/3.25, 
		0, 3850.296435/3.25, 2077.400229/3.25, 0, 0, 1);
	

	//Nexus 9 K
	K = (Mat_<float>(3,3) << 1082, 0, 360, 0, 1085, 640, 0, 0, 1);
	




	if ( argc != 4 )
    {
        printf("usage: ./VideoToPanorama.out <Images_Path> <IMU_File_Path.txt> <Panorama_Output_Path.png>\n");
        return -1;
    }



	const char* imageFolderPath = argv[1];


	std::vector<string> imageNames;
	

	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (imageFolderPath)) != NULL) {
	  /* Add all the files and directories to a vector */
	  while ((ent = readdir (dir)) != NULL) {
	  	string imageName = ent->d_name;
	  	if(imageName != ".." && imageName != "."){
	  		imageNames.push_back(argv[1] + imageName);
	  	}
	  }
	  closedir (dir);
	}  else {
	  /* could not open directory */
	  perror ("");
	  return EXIT_FAILURE;
	}

	const char * imuFile = argv[2];

	ifstream csvFile (imuFile, ios::in );

	ofstream binaryFileOut("output.dat",  ios::out | ios::binary);

	csvToBinary(csvFile, binaryFileOut);
	csvFile.close();
	binaryFileOut.close();
	

	

	cvNamedWindow("panorama", WINDOW_NORMAL);
	imshow("panorama", panorama);
	waitKey(0);

	sort(imageNames.begin(), imageNames.end());
	// while(true){
	ifstream binaryFile("output.dat",  ios::in | ios::binary);

	clock_t startTime, endTime;
	startTime = clock();

	for (auto & image : imageNames){
		projectImageToPanorama(image, binaryFile);

		imshow("panorama", panorama);
		waitKey(1);	
	}

	endTime = clock();
	float seconds = ((float) endTime - (float)startTime)/CLOCKS_PER_SEC;
	
	cout << "Time to stitch : " << seconds << endl;
	binaryFile.close();
	// waitKey(5000);
	// panorama = Mat (PANO_H, PANO_H*2, CV_8UC3, Scalar::all(255));
	
	

// }



	

	cout << "DONE\nHit enter to save or ctrl-c the terminal to not save" << endl;

	cvNamedWindow("panorama", WINDOW_NORMAL);
	imshow("panorama", panorama);
	waitKey(0);



	imwrite(argv[3], panorama);
	cout << "Saved" << endl;
	return 0;
}



Mat readInRotation(ifstream & file, float  timeSamp){

	cout << "Reading the file" << endl;
	float time1, time2;
	file.seekg(0);

	file.read(reinterpret_cast<char *> (& time1), sizeof(float));

	int i = 1;
	while(file && !file.eof()){
		file.seekg(sizeof(float) * NUM_ENTRY_PER_LINE *i);
		file.read(reinterpret_cast<char *> (& time2), sizeof(float));

		
		if((time1 <= timeSamp && time2 >= timeSamp))
			break;
		else
			time1 = time2;
		
		++i;
	}

	if(file && !file.eof()){
		Mat rot_matrix (3,3, CV_32F);


		file.read(reinterpret_cast< char *> (&rot_matrix.at<float>(0,0)), sizeof(float)*3);
		file.read(reinterpret_cast< char *> (&rot_matrix.at<float>(1,0)), sizeof(float)*3); 
		file.read(reinterpret_cast< char *> (&rot_matrix.at<float>(2,0)), sizeof(float)*3); 


		return rot_matrix.t();
	} else{
		return Mat (0,0, CV_32F);
		cout << "EOF" << endl;
	}

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


int projectImageToPanorama(string & imageName, ifstream & imuFile){
	clock_t startTime, endTime;
	startTime = clock();
	Mat img = imread (imageName, 1);

	if ( !img.data )
    {
    	cout << "Error reading image" << endl;
        throw "Error reading image";
    }


	size_t pos  = imageName.find("-");
	float imgNum = stof (imageName.substr(pos +1, 5));

	cout << imgNum << endl;

	Mat rot_matrix = readInRotation(imuFile, (imgNum-1)/FPS);

	if(rot_matrix.empty())
		return -1;
	
	
	int channels = img.channels();
	int nRows = img.rows;
	int nCols = img.cols;


	float panoCoords [2] = {0, 0};
	int vp, up;


	cout << "Sticthing " << endl;

	
	float corner1 [3], corner2 [3], corner3 [3], corner4 [3];
	float imageCoords [2] = {0,0};

	image_coords_to_pano_coords(imageCoords, corner1, rot_matrix);

	imageCoords [0] = nCols;
	image_coords_to_pano_coords(imageCoords, corner2, rot_matrix);

	imageCoords[1] = nRows;
	image_coords_to_pano_coords(imageCoords, corner3, rot_matrix);

	imageCoords[0] = 0;
	image_coords_to_pano_coords(imageCoords, corner4, rot_matrix); 


	int width = max(corner2[0] - corner1[0], corner3[0] - corner4[0]);
	int height = max(corner1[1] - corner4[1], corner2[1] - corner3[1]);
	int startRow = min(corner3[1], corner4[1]);
	int startCol = min(corner1[0], corner4[0]);
	width *= channels;
	startCol *= channels;


	for (int i = startRow; i < height + startRow; ++i)
	{
		uchar * dst = panorama.ptr<uchar>(i);
		
		panoCoords[0] = startCol/channels;
		panoCoords[1] = i;

		pano_coords_to_image_coords(panoCoords, imageCoords, rot_matrix);

		
		if( imageCoords[1] < nRows && imageCoords[1] >= 0){

			uchar * src = img.ptr<uchar>((int) imageCoords[1]);
			for (int j = startCol; j < width + startCol; j+=channels)
			{
				panoCoords[0] = j/channels;

				pano_coords_to_image_coords(panoCoords, imageCoords, rot_matrix);


				if(imageCoords[0] < nCols && imageCoords[0] >= 0){
					int srcRow = imageCoords[0];
					srcRow *= channels;

					if(srcRow % 3 !=0 || j %3 !=0){
						cout << "Image Row: " << srcRow
							<< "Pano Row" << j << endl;
						throw "EXIT_FAILURE";
					}

					for (int k = 0; k < channels; ++k)
					{
						if(dst[j + k] == 255)
							dst[j+k] = src[srcRow + k];
						else
							dst[j+k] = (dst[j+k] + src[srcRow + k])/2;
					}
				}


			}
		}
	}

	endTime = clock();
	float seconds = ((float)endTime - (float)startTime)/CLOCKS_PER_SEC;


	cout << "Done sticthing \t" << flush;
	cout << seconds << endl;

	return 0;


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

	pano_coords [0] = theta/PI * PANO_H; //col #
	pano_coords [1] = phi/PI * PANO_H; //row #

	
}


void pano_coords_to_image_coords(float * pano_coords, float * img_coords, Mat & R){
	float theta = pano_coords[0]*PI/PANO_H - PI;
	float phi = pano_coords[1]*PI/PANO_H;
	float x = cos(theta) * sin(phi);
	float y = sin(theta) * sin(phi);
	float z = cos(phi);

	Mat world_coords = (Mat_<float>(3,1) << x, y, z);

	Mat image_coords = K * R.t() * world_coords;

	img_coords [0] = 720 - 1- image_coords.at<float>(0)/image_coords.at<float>(2);
	img_coords [1] = image_coords.at<float>(1)/image_coords.at<float>(2);	
}




/*int project_v_to_vp (int v, Mat & R){
	int u = 0;

	float xi = u ;
	float yi = v ;



	Mat image_coords = (Mat_<float>(3,1) << xi, yi, 1);

	Mat world_coords = R * K.inv() * image_coords;


	double r = world_coords.at<float>(0) * world_coords.at<float>(0) 
		+ world_coords.at<float>(1) * world_coords.at<float>(1) 
		+ world_coords.at<float>(2) * world_coords.at<float>(2);


	r = sqrt(r);

	double phi = acos(world_coords.at<float>(2)/r);


	return (int) (phi/PI* PANO_H );
}


int project_u_to_up(int u, int v, Mat & R){
	float xi = u ;
	float yi = v ;




	Mat image_coords = (Mat_<float>(3,1) << xi, yi, 1);

	Mat world_coords = R * K.inv() * image_coords;



	double theta = 
		atan2(world_coords.at<float>(1),world_coords.at<float>(0)) + PI;



	return (int) (theta/PI* PANO_H );
}*/

