#ifndef SCAN_TYPEDEFS_HPP
#define SCAN_TYPEDEFS_HPP

#include <eigen3/Eigen/StdVector>
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/Sparse>
#include <fstream>
#include <memory>

#define NUM_ROTS 4

const int panoResolution = 4267;
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

	SHOT1344WithXYZ() : descriptor {std::make_shared<Eigen::Vector1344f> ()}
	{
	};

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

template<typename MatrixType>
void saveMatrixAsSparse(const MatrixType & mat, std::ofstream & out) {
	typedef typename MatrixType::Scalar Scalar;
	int numNonZeros = 0, rows = mat.rows(), cols = mat.cols();
	const Scalar * dataPtr = mat.data();
	for (int i = 0; i < mat.size(); ++i)
		if (*(dataPtr + i)) ++numNonZeros;

	out.write(reinterpret_cast<const char *>(&numNonZeros), sizeof(numNonZeros));
	out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
	out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));

	for (int i = 0; i < mat.size(); ++i) {
		if (*(dataPtr + i)) {
			out.write(reinterpret_cast<const char *>(&i),sizeof(i));
			out.write(reinterpret_cast<const char *>(dataPtr + i), sizeof(Scalar));
		}
	}
}


template<typename MatrixType>
void loadMatrixFromSparse(MatrixType & mat, std::ifstream & in) {
	typedef typename MatrixType::Scalar Scalar;
	int numNonZeros, rows, cols;

	in.read(reinterpret_cast<char *>(&numNonZeros), sizeof(numNonZeros));
	in.read(reinterpret_cast<char *>(&rows), sizeof(rows));
	in.read(reinterpret_cast<char *>(&cols), sizeof(cols));

	mat = MatrixType::Zero(rows, cols);
	Scalar * dataPtr = mat.data();

	for (int i = 0; i < numNonZeros; ++i) {
		int index;
		in.read(reinterpret_cast<char *>(&index), sizeof(index));
		in.read(reinterpret_cast<char *>(dataPtr + index), sizeof(Scalar));
	}
}

template<typename SparseMatrixType>
void saveSparseMatrix(SparseMatrixType & mat, std::ofstream & out) {
	typedef typename SparseMatrixType::Scalar Scalar;

	int rows = mat.rows(), cols = mat.cols(), numNonZeros = mat.nonZeros();
	out.write(reinterpret_cast<const char *>(&numNonZeros), sizeof(numNonZeros));
	out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
	out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));

	for (int i = 0; mat.outerSize(); ++i) {
		for (typename SparseMatrixType::InnerIterator it (mat, i); it; ++it) {
			int index = it.col()*rows + it.row();
			Scalar value = it.value();
			out.write(reinterpret_cast<const char *>(&index),sizeof(index));
			out.write(reinterpret_cast<const char *>(&value), sizeof(Scalar));
		}
	}
}

template<typename SparseMatrixType>
void loadSparseMatrix(SparseMatrixType & mat, std::ifstream & in) {
	typedef typename SparseMatrixType::Scalar Scalar;
	typedef Eigen::Triplet<Scalar> TripType;

	int rows, cols, numNonZeros;
	in.read(reinterpret_cast<char *>(&numNonZeros), sizeof(numNonZeros));
	in.read(reinterpret_cast<char *>(&rows), sizeof(rows));
	in.read(reinterpret_cast<char *>(&cols), sizeof(cols));
	mat.resize(rows, cols);
	mat.reserve(numNonZeros);
	std::vector<TripType> tripletList;
	tripletList.reserve(numNonZeros);

	for (int i = 0; i < numNonZeros; ++i) {
		int index;
		Scalar value;
		in.read(reinterpret_cast<char *>(&index), sizeof(index));
		in.read(reinterpret_cast<char *>(&value), sizeof(Scalar));
		int col = floor(index/rows);
		int row = index % rows;
		tripletList.push_back(TripType (row, col, value));
	}
	mat.setFromTriplets(tripletList.begin(), tripletList.end());
	mat.prune(1);
}

template<typename SparseVectorType>
void saveSpareVector(const SparseVectorType & vec, std::ofstream & out) {
	typedef typename SparseVectorType::Scalar Scalar;
	int nonZeros = vec.nonZeros(), size = vec.size();
	out.write(reinterpret_cast<const char *>(&nonZeros), sizeof(nonZeros));
	out.write(reinterpret_cast<const char *>(&size), sizeof(size));
	for (int i = 0; i < vec.outerSize(); ++i) {
		for (typename SparseVectorType::InnerIterator it (vec, i); it; ++it) {
			Scalar value = it.value();
			short row = it.row();
			out.write(reinterpret_cast<const char *>(&value), sizeof(Scalar));
			out.write(reinterpret_cast<const char *>(&row), sizeof(row));
		}
	}
}

template<typename SparseVectorType>
void loadSparseVetor(SparseVectorType & vec, std::ifstream & in) {
	typedef typename SparseVectorType::Scalar Scalar;
	int nonZeros, size;
	in.read(reinterpret_cast<char *>(&nonZeros), sizeof(nonZeros));
	in.read(reinterpret_cast<char *>(&size), sizeof(size));
	vec.resize(size);
	vec.reserve(nonZeros);
	for (int i = 0; i < nonZeros; ++i) {
		Scalar value;
		short row;
		in.read(reinterpret_cast<char *>(&value), sizeof(Scalar));
		in.read(reinterpret_cast<char *>(&row), sizeof(row));
		vec.coeffRef(row) = value;
	}
}

typedef struct SPARSE1344WithXYZ {
	typedef Eigen::SparseVector<float> VecType;
	std::shared_ptr<VecType> descriptor;
	Eigen::Vector3d position;

	SPARSE1344WithXYZ() : descriptor {std::make_shared<VecType> (1344)}
	{
	};

	void writeToFile(std::ofstream & out) {
		saveSpareVector(*descriptor, out);
		out.write(reinterpret_cast<const char *>(position.data()), sizeof(position));
	}

	void loadFromFile(std::ifstream & in) {
		loadSparseVetor(*descriptor, in);
		in.read(reinterpret_cast<char *>(position.data()), sizeof(position));
	}

} SPARSE1344WithXYZ;

namespace scan {
	typedef struct {
		Eigen::Vector3f point;
		float intensity;
		unsigned char rgb [3];

		void writeToFile(std::ofstream & out) {
			out.write(reinterpret_cast<const char *>(point.data()), sizeof(point));
			out.write(reinterpret_cast<const char *>(&intensity), sizeof(intensity));
			out.write(reinterpret_cast<const char *>(rgb), 3*sizeof(char));
		}
		void loadFromFile(std::ifstream & in) {
			in.read(reinterpret_cast<char *>(point.data()), sizeof(point));
			in.read(reinterpret_cast<char *>(&intensity), sizeof(intensity));
			in.read(reinterpret_cast<char *>(rgb), 3*sizeof(char));
		}

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
    Eigen::Vector3i zZ;
    size_t c;

    void writeToFile(std::ofstream & out) {
    	int numZ = v.size();
    	out.write(reinterpret_cast<const char *>(&numZ), sizeof(numZ));
    	for (int k = 0; k < numZ; ++k) {
    		saveMatrixAsSparse(v[k], out);
    	}
    	out.write(reinterpret_cast<const char *>(zZ.data()), sizeof(zZ));
    	out.write(reinterpret_cast<const char *>(&c), sizeof(c));
    };

    void loadFromFile(std::ifstream & in) {
    	int numZ;
    	in.read(reinterpret_cast<char *>(&numZ), sizeof(numZ));
    	v.resize(numZ);
    	for (int k = 0; k < numZ; ++k) {
    		loadMatrixFromSparse(v[k], in);
    	}
    	in.read(reinterpret_cast<char *>(zZ.data()), sizeof(zZ));
    	in.read(reinterpret_cast<char *>(&c), sizeof(c));

    }
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
    void writeToFile(std::ofstream & out) {
    	out.write(reinterpret_cast<const char *>(zZ.data()), sizeof(zZ));
    	out.write(reinterpret_cast<const char *>(&x), sizeof(x));
    	out.write(reinterpret_cast<const char *>(&y), sizeof(y));
    	out.write(reinterpret_cast<const char *>(&z), sizeof(z));
    	out.write(reinterpret_cast<const char *>(&vox), sizeof(vox));
    	out.write(reinterpret_cast<const char *>(&s), sizeof(s));
    };

    void loadFromFile(std::ifstream & in) {
    	in.read(reinterpret_cast<char *>(zZ.data()), sizeof(zZ));
    	in.read(reinterpret_cast<char *>(&x), sizeof(x));
    	in.read(reinterpret_cast<char *>(&y), sizeof(y));
    	in.read(reinterpret_cast<char *>(&z), sizeof(z));
    	in.read(reinterpret_cast<char *>(&vox), sizeof(vox));
    	in.read(reinterpret_cast<char *>(&s), sizeof(s));
    };
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
