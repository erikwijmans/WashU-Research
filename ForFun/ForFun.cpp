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


using namespace std;
using namespace cv;


int main(int argc, char const *argv[])
{

	const char * imageFolderPath = "C:/Users/Ashley/Desktop/a/";
	std::vector<string> imageNames;



	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (imageFolderPath)) != NULL) {
	  /* Add all the files and directories to a vector */
	  while ((ent = readdir (dir)) != NULL) {
	  	string imageName = ent->d_name;
	  	if(imageName != ".." && imageName != "."){
	  		imageNames.push_back(imageFolderPath + imageName);
	  	}
	  }
	  closedir (dir);
	}  else {
	  /* could not open directory */
	  perror ("");
	  return EXIT_FAILURE;
	}


	for (int i = 0; i < imageNames.size(); i++)
	{
		Mat img = imread(imageNames[i], 1);
		cvNamedWindow("THE CAGE", WINDOW_AUTOSIZE);
		imshow("THE CAGE", img);
		waitKey(0);
	}



	return 0;
}