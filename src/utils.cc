#include <cmath>
#include "utils.h"
#include <iostream>

double Utils::sigmod(double x){
	return 1.0/(1.0+exp(-x));
}


double Utils::crossEntropyLoss(Eigen::VectorXd y,Eigen::VectorXd y_pred){
	int n = y.size();
	double loss;
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
    for(int i=0;i<n;i++){
        if(y_(i)==(pred(i)>0.5?1:0)){
            hit += 1.0;
        }
    }
    return hit/n;
}
