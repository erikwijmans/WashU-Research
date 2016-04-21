#ifndef GET_ROTATIONS_H
#define GET_ROTATIONS_H

#include "preprocessor.h"

void getRotations(const pcl::PointCloud<NormalType>::Ptr & cloud_normals,
	const std::string & outName);


#endif // GET_ROTATIONS_H
