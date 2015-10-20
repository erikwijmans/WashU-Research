#include <stdio.h>
#include <iostream>
#include <string>
#include <dirent.h>
#include <vector>
#include <fstream>
#include <math.h>
#include <time.h>
#include <algorithm>

using namespace std;
void csvToBinary(const string &, const string &);

int main(int argc, char const *argv[])
{

	if(argc <3){
		cout << "Usage:  csvToBinary <csvFileFolder>/ <binaryFileFolder>/" << endl;
		return -1;
	}

	std::vector<string> csvFileNames;
	const char * fileFolderIn = argv[1];
	const char * fileFolderOut = argv[2];
	
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (fileFolderIn)) != NULL) {
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
		string csvFileName = fileFolderIn + csvFileNames[i];
		string binaryFileName = fileFolderOut + 
			csvFileNames[i].substr(0,csvFileNames[i].find(".")) + ".dat";
		csvToBinary(csvFileName, binaryFileName);
	}
		

	return 0;
}

void csvToBinary(const string & fileNameIn, const string& fileNameOut){
	cout << fileNameIn << endl << fileNameOut << endl;
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

		if((i % 2000000) == 0)
	    	cout << "." << flush;
	    

	    
	    i++;

	}
	cout << "Done" << endl;
	scanFile.close();
	binaryFile.close();
}