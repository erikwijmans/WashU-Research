#include <stdio.h>
#include <iostream>
#include <string>
#include <dirent.h>
#include <vector>
#include <fstream>
#include <math.h>
#include <time.h>
#include <algorithm>

#include <gflags/gflags.h>

DEFINE_bool(redo, false, "Redo all the cloud_normals");
DEFINE_string(outFolder, "/home/erik/Projects/3DscanData/DUC/Floor1/binaryFiles/", 
	"Path to binary files");
DEFINE_string(inFolder, "/home/erik/Projects/3DscanData/DUC/Floor1/PTXFiles/",
 "Path to Output");

using namespace std;
void csvToBinary(const string &, const string &);

int main(int argc, char *argv[])
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	std::vector<string> csvFileNames;
	
	
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (FLAGS_inFolder.data())) != NULL) {
	  /* Add all the files and directories to a vector */
	  while ((ent = readdir (dir)) != NULL) {
	  	string fileName = ent->d_name;
	  	if(fileName != ".." && fileName != "."){
	  		csvFileNames.push_back(fileName);
	  	}
	  }
	  closedir (dir);
	}  else {
	  /* could not open directory */
	  perror ("");
	  return EXIT_FAILURE;
	} 

	sort(csvFileNames.begin(), csvFileNames.end());
	for (int i = 0; i < csvFileNames.size(); ++i)
	{
		const string csvFileName = FLAGS_inFolder + csvFileNames[i];
		const string binaryFileName = FLAGS_outFolder + 
			csvFileNames[i].substr(0,csvFileNames[i].find(".")) + ".dat";
		csvToBinary(csvFileName, binaryFileName);
	}
		

	return 0;
}

void csvToBinary(const string & fileNameIn, const string& fileNameOut){

	if(!FLAGS_redo) {
		ifstream out (fileNameOut, ios::in | ios::binary);
		if(out.is_open())
			return;
	}

	ifstream scanFile (fileNameIn, ios::in);
  ofstream binaryFile (fileNameOut, ios::out | ios::binary);
  
  
  int columns, rows;
 	scanFile >> columns >> rows;
  string line;

  binaryFile.write(reinterpret_cast<const char *> (& columns), sizeof(int));
  binaryFile.write(reinterpret_cast<const char *> (& rows), sizeof(int));


  for (int i = 0; i < 9; ++i)
  {
  	getline(scanFile, line);
  }

	size_t i = 0;
    for (int k = 0; k < columns * rows; ++k) {
	    float point [3];
		scanFile >> point[0] >> point[1] >> point[2];

		double itmp[4];
		for (int j = 0; j < 4; ++j)
			scanFile >> itmp[j];

		if(itmp[0] < 0.2){
			float temp = 0;
			binaryFile.write(reinterpret_cast<const char *> (& temp), 
				sizeof(float));
			binaryFile.write(reinterpret_cast<const char *> (& temp), 
				sizeof(float));
			binaryFile.write(reinterpret_cast<const char *> (& temp), 
				sizeof(float));
		} else{
			for (int i = 0; i < 3; ++i)
			{
				binaryFile.write(reinterpret_cast<const char *> (& point[i]), 
					sizeof(float));
			}
		}
	    i++;
	}
	scanFile.close();
	binaryFile.close();
}