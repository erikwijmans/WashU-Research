#ifndef DT_HPP_
#define DT_HPP_

#include <eigen3/Eigen/Eigen>


namespace dt{
	Eigen::MatrixXd distanceTransform(const Eigen::MatrixXd &);

	void dt(Eigen::MatrixXd & image);

	Eigen::VectorXf dt(Eigen::VectorXf &);
}


#endif