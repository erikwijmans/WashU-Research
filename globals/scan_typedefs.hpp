#ifndef SCAN_TYPEDEFS_HPP
#define SCAN_TYPEDEFS_HPP

#include <eigen3/Eigen/StdVector>
#include <eigen3/Eigen/Eigen>
#include <fstream>

#define NUM_ROTS 4

const int panoResolution = 500;
const double PI = 3.14159265358979323846;

namespace place {
  typedef struct {
    double pA = 0.0, feA = 0.0, feB = 0.0, fx = 0.0;
    double w = 0.0;
  } edgeWeight;

  typedef struct {
    double weight = 0.0;
    std::vector<int> incident;
  } hOrder;
} // place

namespace Eigen {
	typedef Matrix< float, 1344, 1 > Vector1344f;
	typedef Matrix<place::edgeWeight, Dynamic, Dynamic> MatrixXE;
  typedef Array<place::hOrder, Dynamic, Dynamic> ArrayXH;
  typedef Matrix<char, Dynamic, Dynamic> MatrixXb;
	typedef Matrix< int, Dynamic, Dynamic, RowMajor > RowMatrixXi;
} // Eigen

typedef struct {
	Eigen::Vector1344f descriptor;
	Eigen::Vector3d position;

	void writeToFile(std::ofstream & out) {
		out.write(reinterpret_cast<const char *>(descriptor.data()), 
			descriptor.size()*sizeof(float));
		out.write(reinterpret_cast<const char *>(position.data()), sizeof(position));
	}

	void loadFromFile(std::ifstream & in) {
		in.read(reinterpret_cast<char *>(descriptor.size()),
			descriptor.size()*sizeof(float));
		in.read(reinterpret_cast<char *>(position.data()), sizeof(position));
	}

} SHOT1344WithXYZ;

namespace scan {
	typedef struct {
		Eigen::Vector3f point;
		double intensity;
		unsigned char r, g, b;
	} PointXYZRGBA;
	
	typedef struct {
		Eigen::Vector3f point;
		unsigned char rgb [3];

		void writeToFile(std::ofstream & out) {
			out.write(reinterpret_cast<const char *>(point.data()), sizeof(point));
			out.write(reinterpret_cast<const char *>(rgb), 3*sizeof(char));
		}
		void loadFromFile(std::ifstream & in) {
			in.read(reinterpret_cast<char *>(point.data()), sizeof(point));
			in.read(reinterpret_cast<char *>(rgb), 3*sizeof(char));
		}
	} PointXYZRGB;
} // scan

namespace place {
	typedef struct {
		double score = 0;
		double scanFP = 0;
		double fpScan = 0;
		int rotation = 0;
		int fpPixels = 0;
		int scanPixels = 0;
		int x = 0;
		int y = 0;
	} posInfo;

	typedef struct {
		const posInfo *** maps;
		double exclusionSize;
		int rows, cols;
	} exclusionMap;

	typedef struct {
    std::vector<Eigen::MatrixXb> v;
    size_t c;
  } voxel;

	typedef struct {
		posInfo s;
		int scanNum;
	} moreInfo;

	typedef struct {
		posInfo s;
    double w;
		int color;
	} node;

  typedef struct {
    int X1;
    int Y1;
    int X2;
    int Y2;
  } rect;

  typedef struct {
    Eigen::Vector3i zZ;
    int x, y, z;
    double vox, s;
  } metaData;
} // place


#endif // SCAN_TYPEDEFS_HPP
