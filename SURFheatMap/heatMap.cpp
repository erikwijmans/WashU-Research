#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/StdVector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/core/cuda.hpp"
// #include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"

namespace Eigen {
	typedef Matrix<double, Dynamic, Dynamic, RowMajor> RowMatrixXd;
} // Eigen


int main() {
	cv::Mat grayA, grayB;
	grayA = cv::imread("A.png", CV_LOAD_IMAGE_GRAYSCALE);
	grayB = cv::imread("B.png", CV_LOAD_IMAGE_GRAYSCALE);

	const int rows = std::min(grayA.rows,
		grayB.rows);
	const int cols = std::min(grayA.cols, grayB.cols);

	std::vector<cv::KeyPoint> kpA, kpB;

	std::ifstream in ("data.dat", std::ios::in | std::ios::binary);
	int numKP;
	in.read(reinterpret_cast<char *>(&numKP), sizeof(int));
	kpA.resize(numKP);
	for (auto & kp : kpA)
		in.read(reinterpret_cast<char *>(&kp), sizeof(cv::KeyPoint));

	kpB.resize(numKP);
	for (auto & kp : kpB)
		in.read(reinterpret_cast<char *>(&kp), sizeof(cv::KeyPoint));

	std::cout << "Uploading" << std::endl;
	cv::cuda::GpuMat gpu_imgA, gpu_imgB;
	gpu_imgA.upload(grayA);
	gpu_imgB.upload(grayB);

	cv::Mat maskA = cv::Mat(grayA.size(), CV_8UC1, cv::Scalar(255));
	cv::Mat maskB = cv::Mat(grayB.size(), CV_8UC1, cv::Scalar(255));

	cv::cuda::GpuMat gpu_maskA, gpu_maskB;
	gpu_maskA.upload(maskA);


	cv::cuda::SURF_CUDA detector (200, 4, 2, true);

	cv::cuda::GpuMat gpu_kpA, gpu_kpB;
	cv::cuda::GpuMat gpu_descripA, gpu_descripB;
	cv::Mat descripA, descripB;
	detector.uploadKeypoints(kpA, gpu_kpA);
	detector(gpu_imgA, gpu_maskA, gpu_kpA, gpu_descripA, true);
	gpu_descripA.download(descripA);

	gpu_maskA.release();
	gpu_imgA.release();
	gpu_kpA.release();
	gpu_descripA.release();

	gpu_maskB.upload(maskB);
	detector.uploadKeypoints(kpB, gpu_kpB);
	detector(gpu_imgB, gpu_maskB, gpu_kpB, gpu_descripB, true);
	gpu_descripB.download(descripB);

	int descripSize = detector.descriptorSize();

	std::cout << descripA.rows << "?=" << kpA.size() << std::endl;
	std::cout << descripB.rows << "?=" << kpB.size() << std::endl;

	assert(descripA.rows == kpA.size());
	assert(descripB.rows == kpB.size());
	assert(descripA.rows == descripB.rows);
	Eigen::RowMatrixXd heatMap = Eigen::RowMatrixXd::Zero(rows, cols);
	for (int i = 0; i < kpA.size(); ++i) {
		cv::Mat A = descripA.row(i);
		cv::Mat B = descripB.row(i);
		int col = kpA[i].pt.x;
		int row = kpA[i].pt.y;
		double norm = cv::norm(A, B, cv::NORM_L2);
		heatMap(row, col) = boost::math::isfinite(norm) &&
			Eigen::numext::isfinite(norm) ? norm : 0;
	}

	const double * dataPtr = heatMap.data();
	double average = 0, sigma = 0;
	int count = 0;
	for (int i = 0; i < heatMap.size(); ++i) {
		if (*(dataPtr + i)) {
			average += *(dataPtr + i);
			++count;
		}
	}
	average/= count;

	for (int i = 0; i < heatMap.size(); ++i) {
		const double tmp = *(dataPtr + i) - average;
		if (tmp) {
			sigma += tmp*tmp;
		}
	}
	sigma /= count;
	sigma = sqrt(sigma);

	cv::Mat out (rows, cols, CV_8UC3, cv::Scalar::all(0));
	for (int j = 0; j < rows; ++j) {
		uchar * dst = out.ptr<uchar>(j);
		for (int i = 0; i < cols; ++i) {
			if (heatMap(j ,i)) {
				const int gray = cv::saturate_cast<uchar>(
	        255.0 * ((heatMap(j, i) - average)/(1.0 * sigma) + 1.0)/2.0);
	      int red, green, blue;
	      if (gray < 128) {
	        red = 0;
	        green = 2 * gray;
	        blue = 255 - green;
	      } else {
	        blue = 0;
	        red = 2 * (gray - 128);
	        green = 255 - red;
	      }
	      dst[i*3] = blue;
	      dst[i*3 +1] = green;
	      dst[i*3 + 2] = red;
			}
		}
	}
	cvNamedWindow("A", CV_WINDOW_NORMAL);
	cv::imshow("A", grayA);

	cvNamedWindow("B", CV_WINDOW_NORMAL);
	cv::imshow("B", grayB);

	cv::imwrite("heatMap.png", out);
	cvNamedWindow("SIFT Heat Map", CV_WINDOW_NORMAL);
	cv::imshow("SIFT Heat Map", out);
	cv::waitKey(0);

}