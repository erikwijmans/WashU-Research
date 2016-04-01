#ifndef SCAN_TYPEDEFS_HPP
#define SCAN_TYPEDEFS_HPP

#include <eigen3/Eigen/StdVector>
#include <eigen3/Eigen/Eigen>
#include <fstream>
#include <memory>

#define NUM_ROTS 4

const int panoResolution = 500;
const double PI = 3.14159265358979323846;

static inline Eigen::Vector3i alignedToSource(const auto & aligned,
  const Eigen::Matrix3d & R, const auto & zZ) {
	Eigen::Vector3d AD  (aligned[0], aligned[1], aligned[2]);
	Eigen::Vector3d zZD (zZ[0], zZ[1], zZ[0]);
  return (R*(AD - zZD) + zZD).cast<int>();
}

static inline Eigen::Vector3i sourceToAligned(const auto & source,
  const Eigen::Matrix3d & R, const auto & zZ) {
	Eigen::Vector3d SD (source[0], source[1], source[2]);
	Eigen::Vector3d zZD (zZ[0], zZ[1], zZ[2]);
  return (R.inverse()*(SD - zZD) + zZD).cast<int>();
}

namespace place {
  class edge {
	  public:
	    double pA, feA, feB, fx;
	    double w, shotW;
	    edge () : pA {0}, feA {0}, feB {0}, fx {0}, w {0}, shotW {0}
	    {};
	    edge (double pA, double feA, double feB, double fx, double w,
	    	double shotW) : pA {pA}, feA {feA}, feB {feB}, w {w}, shotW {shotW}
	    {};
  };

  typedef struct {
    double weight = 0.0;
    std::vector<int> incident;
  } hOrder;
} // place

namespace Eigen {
	typedef Matrix< float, 1344, 1 > Vector1344f;
	typedef Array<place::edge, Dynamic, Dynamic> MatrixXE;
  typedef Array<place::hOrder, Dynamic, Dynamic> ArrayXH;
  typedef Matrix<char, Dynamic, Dynamic> MatrixXb;
	typedef Matrix< int, Dynamic, Dynamic, RowMajor > RowMatrixXi;
} // Eigen

typedef struct SHOT1344WithXYZ {
	std::shared_ptr<Eigen::Vector1344f> descriptor;
	Eigen::Vector3d position;

	SHOT1344WithXYZ() {
		descriptor = std::make_shared<Eigen::Vector1344f>();
	}

	void writeToFile(std::ofstream & out) {
		out.write(reinterpret_cast<const char *>(descriptor->data()), 
			descriptor->size()*sizeof(float));
		out.write(reinterpret_cast<const char *>(position.data()), sizeof(position));
	}

	void loadFromFile(std::ifstream & in) {
		in.read(reinterpret_cast<char *>(descriptor->data()),
			descriptor->size()*sizeof(float));
		in.read(reinterpret_cast<char *>(position.data()), sizeof(position));
	}

} SHOT1344WithXYZ;

namespace scan {
	typedef struct {
		Eigen::Vector3f point;
		double intensity;
		unsigned char rgb [3];
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
  } voxelGrid;

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

  class cube {
	  private:
	  	void swap(int & a, int & b) {
	  		int savedA = a;
	  		a = b;
	  		b = savedA;
	  	};
    public:
      int X1, Y1, Z1;
      int X2, Y2, Z2;
      cube() : X1 {0}, X2 {0}, Y1 {0}, Y2 {0}, Z1 {0}, Z2 {0} 
      {};
      int volume() {
      	const int width = this->X2 - this->X1;
      	const int length = this->Y2 - this->Y1;
      	const int height = this->Z2 - this->Z1;
      	return width*length*height;
      };
      void rotate(const Eigen::Matrix3d & R, const Eigen::Vector3i & zZ) {
      	Eigen::Vector3i corner1 (X1, X2, 0), corner2 (X2, Y1, 0),
      		corner3 (X2, Y2, 0), corner4(X1, Y2, 0);
      	corner1 = sourceToAligned(corner1, R, zZ);
      	corner2 = sourceToAligned(corner2, R, zZ);
      	corner3 = sourceToAligned(corner3, R, zZ);
      	corner4 = sourceToAligned(corner4, R, zZ);


      	X1 = std::min(std::min(std::min(corner1[0], corner2[0]), corner3[0]),
      		corner4[0]);
      	Y1 = std::min(std::min(std::min(corner1[1], corner2[1]), corner3[1]),
      		corner4[1]);

      	X2 = std::max(std::max(std::max(corner1[0], corner2[0]), corner3[0]),
      		corner4[0]);
      	Y2 = std::max(std::max(std::max(corner1[1], corner2[1]), corner3[1]),
      		corner4[1]);

      	if (X1 > X2) swap(X1, X2);
      	if (Y1 > Y2) swap(Y1, Y2);
      };
  };

} // place

static std::ostream & operator<<(std::ostream & os, const place::cube & print) {
  os << "(" << print.X1 << ", " << print.Y1 << ", " << print.Z1 << ")" << std::endl;
  os << "      " << "(" << print.X2 << ", " << print.Y2 << ", " << print.Z2 <<  ")";
  return os;
}

static std::ostream & operator<<(std::ostream & os, const place::edge & print) {
  os << "edge: " << print.w << " shot: " << print.shotW << std::endl;
  os << print.pA << "  " << print.feA << std::endl;
  os << print.fx << "  " << print.feB;
  return os;
}

static std::ostream & operator<<(std::ostream & os, const place::posInfo * print) {
	os << print->score <<"      " << print->x << "      " 
		<< print->y << "      " << print->rotation << std::endl;
	os << print->scanFP << "      " << print->fpScan << std::endl;
	os << print->scanPixels << "    " << print->fpPixels;
	return os;
}

#endif // SCAN_TYPEDEFS_HPP
