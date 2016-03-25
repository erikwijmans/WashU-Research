#ifndef PLACE_SCAN_PANORAMA_MATCHER_H
#define PLACE_SCAN_PANORAMA_MATCHER_H

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/StdVector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


namespace pano {

	double compare( const cv::Mat & panoA, const cv::Mat & panoB,
		const Eigen::Matrix3d & RA, const Eigen::Matrix3d & RB, 
		const std::vector<Eigen::Vector3d> & pointsInA,
		const Eigen::Vector3d & translation);

	void voxelToWorld(std::vector<Eigen::Vector3d> & points, 
		const Eigen::Matrix3d & R, const Eigen::Vector3i & zeroZero, 
		const double metersPerVoxel);

} // pano


#endif // PLACE_SCAN_PANORAMA_MATCHER_H
