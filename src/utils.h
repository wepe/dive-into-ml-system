#ifndef __UTILS_H__
#define __UTILS_H__

#include <eigen3/Eigen/Dense>


class Utils{
public:
　　　　// sigmod function, depend on <cmath> library
    static double sigmod(double x);
    static double crossEntropyLoss(Eigen::VectorXd y,Eigen::VectorXd h);
    static double accuracy(Eigen::VectorXd y,Eigen::VectorXd pred);
    static double accuracy(double* y,double* pred,int size);
    static double* VectorXd_to_double_array(Eigen::VectorXd pred);
    static int* VectorXi_to_int_array(Eigen::VectorXi y);
    static Eigen::MatrixXd slice(Eigen::MatrixXd X,int start_idx,int end_idx);
    static Eigen::VectorXd slice(Eigen::VectorXd X,int start_idx,int end_idx);
};


#endif
