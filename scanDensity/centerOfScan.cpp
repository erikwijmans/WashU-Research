#include <stdio.h>
#include <iostream>
#include <string>
#include <dirent.h>
#include <vector>
#include <fstream>
#include <math.h>
#include <time.h>

using namespace std;
void parseLine(string &, float *, float *);

int main(int argc, char const *argv[])
{
	if(argc != 3){
		cout << "Usage: centerOfScan <scanfile>.ptx <outputfile>.ptx" << endl;
		return -1;
	}

	const char * fileNameIn = argv[1];
	const char * fileNameOut = argv[2];

	ifstream scanFile (fileNameIn, ios::in);
	ofstream centerFile(fileNameOut, ios::out);
	int columns, rows;
   	scanFile >> columns >> rows;
   	centerFile << columns << endl << rows << endl;
   	string line;
   	getline(scanFile, line);
   	for (int i = 0; i < 8; ++i)
    {
    	getline(scanFile, line);
    	centerFile << line << endl;

    }

 	size_t i = 0;
   	while(getline(scanFile, line)) {
	    float point [3], rgb[3];
	    parseLine(line, point, rgb);
	    
		if((i % 2000000) == 0){
	    	cout << "Working..." << i << endl;
	    }
	    i++;

		if(point[0]*point[0] + point[1]*point[1] > 0.8)
			continue;
		
		centerFile << line << endl;
	}

	scanFile.close();
	centerFile.close();

	return 0;
}


void parseLine(string & line, float * point, float * rgb){
	size_t pos1, pos2;
	pos1 = pos2 = 0;

	pos1 = line.find(" ");
	point[0] = stof(line.substr(0, pos1));
	int i = 1;
	while((pos2 = line.find(" ", pos1 +1 )) != string::npos){
			if(i < 3){
				point[i] = stof(line.substr(pos1+1, pos2 - pos1 -1));
			} else if ( i > 3){
				rgb[i-4] = stof(line.substr(pos1+1, pos2 - pos1 -1));
			}

			pos1 = pos2;
			i++;
	}

	rgb[2] = stof(line.substr(pos1 +1));
}