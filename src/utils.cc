#include <cmath>
#include "utils.h"
#include <iostream>

double Utils::sigmod(double x){
    return 1.0/(1.0+exp(-x));
}


double Utils::crossEntropyLoss(Eigen::VectorXd y,Eigen::VectorXd y_pred){
    int n = y.size();
    double loss = 0.0;
    #pragma omp parallel for reduction(+: loss)
    for(int i=0;i<n;i++){
        double yi_prob = y_pred(i);
        yi_prob = std::min(std::max(yi_prob,0.0001),0.9999);
        loss -= (y(i)*log2(yi_prob)+(1-y(i))*log2(1-yi_prob));
    }
    return loss/n;
}


double Utils::accuracy(Eigen::VectorXd y, Eigen::VectorXd pred){
    Eigen::VectorXi y_ = y.cast<int>();
    int n = y_.size();
    double hit = 0.0;
    #pragma omp parallel for reduction(+: hit)
    for(int i=0;i<n;i++){
        if(y_(i)==(pred(i)>0.5?1:0)){
            hit += 1.0;
        }
    }
    return hit/n;
}

double Utils::accuracy(double* y,double* pred,int size){
    double hit = 0.0;
    #pragma omp parallel for reduction(+: hit)
    for(int i=0;i<size;i++){
        if(y[i]==(pred[i]>0.5?1.0:0.0)){
            hit += 1.0;
        }
    }
    return hit/size;
}

Eigen::MatrixXd Utils::slice(Eigen::MatrixXd X,int start_idx,int end_idx){
    Eigen::MatrixXd ret(end_idx-start_idx+1,X.cols());
    #pragma omp parallel for
    for(int i=start_idx;i<=end_idx;i++){
        ret.row(i-start_idx) = X.row(i);
    }
    return ret;
}

Eigen::VectorXd Utils::slice(Eigen::VectorXd y,int start_idx,int end_idx){
    Eigen::VectorXd ret(end_idx-start_idx+1);
    #pragma omp parallel for
    for(int i=start_idx;i<=end_idx;i++){
        ret(i-start_idx) = y(i);
    }
    return ret;
}


int* Utils::VectorXi_to_int_array(Eigen::VectorXi y){
    int size = y.size();
    int* ret = new int[size];
    #pragma omp parallel for
    for(int i=0;i<size;i++){
        ret[i] = y(i);
    }
    return ret;
}  

double* Utils::VectorXd_to_double_array(Eigen::VectorXd pred){
    int size = pred.size();
    double* ret = new double[size];
    #pragma omp parallel for
    for(int i=0;i<size;i++){
        ret[i] = pred(i);
    }
    return ret;
}
