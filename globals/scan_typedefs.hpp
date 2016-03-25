#ifndef SCAN_TYPEDEFS_HPP
#define SCAN_TYPEDEFS_HPP

#include <eigen3/Eigen/Eigen>

namespace Eigen {
	typedef Matrix< float, 1344, 1 > Vector1344f;
} // Eigen

typedef struct {
	Eigen::Vector1344f descriptor;
	Eigen::Vector3d position;
} SHOT1344WithXYZ;

namespace scan {
	typedef struct {
		Eigen::Vector3f point;
		double intensity;
		char r, g, b;
	} PointXYZRGBA;
	typedef struct {
		Eigen::Vector3f point;
		unsigned char rgb [3];
	} PointXYZRGB;
} // scan


#endif // SCAN_TYPEDEFS_HPP
