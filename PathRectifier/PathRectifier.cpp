
/*
Takes a path and solves for what the correct path should by making some assumptions.
For least squares regression method, it assumes that the start and end points must be the same.  It also assumes
that the acceleration and velocity at any point must be the same in the input and output paths.  Also, it locks the start point to (0,0)

For linear programming (which is solved with Gurobi).  It assumes the start and end points must be the same.  It also assumes that the
acceleration at any point must be the same in both the input and output paths.

*/



#include <Eigen/SparseCholesky>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <math.h>
#include <gurobi_c++.h>

#define PI 3.14159

using namespace Eigen;
using namespace std;


int F = 0;
float weight = 1;

void printPath(const char *, VectorXf & );
//csv files are converted to binary because c++ reads csv files poorly at best
void csvToBinary (ifstream &, fstream &);
void shiftToPositive(VectorXf &);
inline float reduceNoise (float a) {return (a > 10) ? a : 0;}
VectorXf lPSolver(VectorXf &);
void rotatePoints(VectorXf &, Matrix2f &);

Matrix2f  rotationMatrix(float theta){
  Matrix2f R;
  R(0,0) = cos(theta);
  R(1,0) = sin(theta);
  R(0,1) = -sin(theta);
  R(1,1) = cos(theta);


  return R;
}



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

  weight = 1;

  for (int i = 0; i < 2*F-2; ++i)
  {
    A.insert(4+i, i) = 1*weight;
    A.insert(4+i, i+2) = -1*weight;
  }

  weight = 1;
  for (int i = 0; i < 2*F-4; ++i)
  {
    A.insert(4+2*F-2 + i, i) = 1*weight;
    A.insert(4+2*F-2 + i, i + 2) = -2*weight;
    A.insert(4+2*F-2 + i, i + 4) = 1*weight;
  }



  b = A*givenX;

  /*for (int i = 4+2*F-2; i < 2*F-4; ++i)
  {
    b(i) = reduceNoise(b(i));
  }*/


  A.insert(0,0) = 1;
  A.insert(1,1) = 1;
  A.insert(2,0) = 1;
  A.insert(2,2*F-2) = 1;
  A.insert(3, 1) = -1;
  A.insert(3, 2*F-1) = -1;

  clock_t startTime, endTime;
  startTime = clock();
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

  endTime = clock();

  float seconds = ((float) endTime - (float)startTime)/CLOCKS_PER_SEC;

  cout << "Time to solve : " << seconds << endl;



  VectorXf GrobX =  lPSolver(b);

  Matrix2f R = rotationMatrix(6.0/180.0*PI);
  rotatePoints(GrobX, R);
  rotatePoints(x, R);

  shiftToPositive(x);
  shiftToPositive(GrobX);


  printPath("GurobiPath.html", GrobX);
  printPath("Path.html", x);
  x.normalize();
  GrobX.normalize();

  double error = 0.0;
  for (int i = 0; i < 2*F; ++i)
  {
    error = (x[i] - GrobX[i]) * (x[i] - GrobX[i]);
  }

  error = sqrt(error);

  cout <<"Difference = " << error << endl;

  return 0;
}

void printPath(const char * name, VectorXf & x){
  ofstream pathWriter (name, ios::out);
  pathWriter << "<!DOCTYPE html>" << endl << "<html>"
    << endl << "<body>" << endl << endl
    << "<svg height=\"1000\" width=\"1000\">" << endl
    << "<image xlink:href=\"file:/home/erik/Projects/c++/PathRectifier/Bryan+Jolley.png\""
    << "x=\"-50\" y=\"-300\" width=\"1000\" height=\"1000\" />" << endl << endl
    << "<polyline points=\"" << flush;
  for (int i = 0; i < 2*F ; i+=2)
  {
    pathWriter << x(i) << "," << x(i+1) << " " << flush;
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

VectorXf lPSolver(VectorXf & b){
  try {
      GRBEnv env = GRBEnv();

      GRBModel model = GRBModel(env);

      GRBVar * dVarList = model.addVars(2*F-4, GRB_CONTINUOUS);
      GRBVar * xyVarList = model.addVars(2*F, GRB_CONTINUOUS);
      model.update();

      GRBLinExpr obj = 0.0;
      for (int i = 0; i < 2*F-4; ++i)
      {
        obj += dVarList[i];
      }

      model.setObjective(obj, GRB_MINIMIZE);
      model.addConstr(xyVarList[0] == xyVarList[2*F-2]);
      model.addConstr(xyVarList[1] == xyVarList[2*F-1]);


      for (int i = 0; i < 2*F-4; ++i)
      {
        model.addConstr(-1*dVarList[i] <= xyVarList[i]
          - 2*xyVarList[i+2]+xyVarList[i+4] - b(4+2*F-2+i));
        model.addConstr(xyVarList[i] - 2*xyVarList[i+2]
          +xyVarList[i+4] - b(4+2*F-2+i) <= dVarList[i]);

      }


      model.optimize();

      VectorXf x (2*F);
      for (int i = 0; i < 2*F; ++i)
      {
        x(i) = xyVarList[i].get(GRB_DoubleAttr_X);
      }

      shiftToPositive(x);

      return x;


    } catch(GRBException e) {
      cout << "Error code = " << e.getErrorCode() << endl;
      cout << e.getMessage() << endl;
    } catch(...) {
      cout << "Exception during optimization" << endl;
    }


}

void rotatePoints(VectorXf & points, Matrix2f & R){
  Vector2f xAxis (1,0);
  Vector2f yAxis (0,1);

  xAxis = R*xAxis;
  yAxis = R*yAxis;

  for (int i = 0; i < points.size(); i+=2)
  {
    float oldX = points(i);
    float oldY = points(i+1);

    Vector2f oldPoint (oldX, oldY);

    points(i) = xAxis.dot(oldPoint);
    points(i+1) = yAxis.dot(oldPoint);
  }
}