#include <cmath>
#include "utils.h"
#include <iostream>

double Utils::sigmod(double x){
	return 1.0/(1.0+exp(-x));
}


double Utils::crossEntropyLoss(Eigen::VectorXi y,Eigen::VectorXd y_pred){
	Eigen::VectorXd y_d = y.cast<double>();
	int n = y_d.size();
	double loss;
	for(int i=0;i<n;i++){
        double yi_prob = y_pred(i);
        yi_prob = std::min(std::max(yi_prob,0.0001),0.9999);
		loss -= (y_d(i)*log2(yi_prob)+(1-y_d(i))*log2(1-yi_prob));
	}
	return loss/n;
}


double Utils::accuracy(Eigen::VectorXi y, Eigen::VectorXd pred){
    int n = y.size();
    double hit = 0.0;
    for(int i=0;i<n;i++){
        if(y(i)==(pred(i)>0.5?1:0)){
            hit += 1.0;
        }
    }
    return hit/n;
}
