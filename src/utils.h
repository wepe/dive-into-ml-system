#ifndef __UTILS_H__
#define __UTILS_H__

#include <eigen3/Eigen/Dense>


class Utils{
public:
	// sigmod function, depend on <cmath> library
	static double sigmod(double x);
	static double crossEntropyLoss(Eigen::VectorXd y,Eigen::VectorXd h);
    static double accuracy(Eigen::VectorXd y,Eigen::VectorXd pred);
};


#endif
