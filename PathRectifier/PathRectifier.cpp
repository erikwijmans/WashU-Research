#include <Eigen/SparseCholesky>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <math.h>



using namespace Eigen;
using namespace std;


int F = 0;

void printPath(const char *, VectorXf & );
void csvToBinary (ifstream &, fstream &);
void shiftToPositive(VectorXf &);
inline float reduceNoise (float a) {return (a > 0.15) ? a : 0;}




int main(int argc, char** argv)
{
	if ( argc != 2 )
    {
        printf("usage: ./PathRectifier <Path.txt> \n");
        return -1;
    }


	const char * pathFile = argv[1];

	ifstream csvFile (pathFile, ios::in );
	fstream binaryFile ("output.dat", ios::in | ios::out | ios::binary);

	csvToBinary(csvFile, binaryFile);
	csvFile.close();


	VectorXf givenX (2*F);
	binaryFile.seekg(0);
	binaryFile.read(reinterpret_cast<char *> (&givenX(0)), sizeof(float)*2*F);
	binaryFile.close();


	shiftToPositive(givenX);
	printPath("startPath.html", givenX);
	
	SparseMatrix<float> A (4*F-2, 2*F);
	A.reserve(6 + (2*F-2)*2 + (2*F-4)*3);
	VectorXf x(2*F) , b(4*F-2);

	float weight = 0.4;

	for (int i = 0; i < 2*F-2; ++i)
	{
		A.insert(4+i, i) = 1*weight;
		A.insert(4+i, i+2) = -1*weight;
	}

	weight = 0.8;
	for (int i = 0; i < 2*F-4; ++i)
	{
		A.insert(4+2*F-2 + i, i) = 1*weight;
		A.insert(4+2*F-2 + i, i + 2) = -2*weight;
		A.insert(4+2*F-2 + i, i + 4) = 1*weight;
	}



	b = A*givenX;

	for (int i = 4+2*F-2; i < 2*F-4; ++i)
	{
		b(i) = reduceNoise(b(i));
	}


	A.insert(0,0) = 1;
	A.insert(1,1) = 1;
	A.insert(2,0) = 1;
	A.insert(2,2*F-2) = 1;
	A.insert(3, 1) = -1;
	A.insert(3, 2*F-1) = -1;

	SparseMatrix<float> APrime (2*F, 2*F);
	VectorXf bPrime (2*F);

	APrime = A.transpose() * A;
	bPrime = A.transpose() * b;
	
	SimplicialLDLT<SparseMatrix<float> > solver;
	solver.compute(APrime);

	if(solver.info() !=Success){
		cout << "Could not compute decompesition" << endl;
		return -1; 
	}

	x = solver.solve(bPrime);

	if(solver.info()!=Success) {
	  cout << "Could not solve" << endl;
	  return -1;
	}

	shiftToPositive(x);

	printPath("Path.html", x);
	
	return 0;
}

void printPath(const char * name, VectorXf & x){
	ofstream pathWriter (name, ios::out);
	pathWriter << "<!DOCTYPE html>" << endl << "<html>"
		<< endl << "<body>" << endl << endl 
		<< "<svg height=\"500\" width=\"500\">" << endl << "<polyline points=\"";
	for (int i = 0; i < 2*F ; i+=2)
	{
		pathWriter << x(i) << "," << x(i+1) << " "; 
	}

	pathWriter << "\"" << " style=\"fill:none;stroke:black;stroke-width:2\" />" 
		<< endl << "Sorry, your browser does not support inline SVG."
		<< endl << "</svg>" << endl << endl << "</body>" << endl 
		<< "</html>" << flush;

	pathWriter.close();
}

void csvToBinary(ifstream & csv, fstream & out){
	string line;

	while(getline(csv, line)){
		size_t pos1 = line.find(",");
		float number = stof (line.substr(0, pos1));
		number *= 100;
		out.write(reinterpret_cast<const char *> (& number), sizeof(float));


		number = stof (line.substr(pos1 +1));
		number *= 100;
		out.write(reinterpret_cast<const char *> (& number), sizeof(float));
		F++;
	} 
}



void shiftToPositive(VectorXf & toShift){
	float minX, minY;
	minX = minY = 0;
	for (int i = 0; i < 2*F; i+=2)
	{
		if(toShift(i) < minX)
			minX = toShift(i);
		if(toShift(i+1) < minY)
			minY = toShift(i+1);
	}

	for (int i = 0; i < 2*F; i+=2)
	{
		toShift(i) -= minX;
		toShift(i+1) -= minY;
	}

}