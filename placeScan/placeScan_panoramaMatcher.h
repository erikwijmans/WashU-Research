#pragma once
#ifndef PLACE_SCAN_PANORAMA_MATCHER_H
#define PLACE_SCAN_PANORAMA_MATCHER_H

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/StdVector>
#include <etw_utils.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace pano {

void compareNCC2(place::Panorama &panoA, place::Panorama &panoB,
                 const Eigen::Matrix3d &RA, const Eigen::Matrix3d &RB,
                 const Eigen::Vector3d &aToB, const Eigen::Vector3d &bToA,
                 place::edge &e);

} // pano

#endif // PLACE_SCAN_PANORAMA_MATCHER_H
