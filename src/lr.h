#ifndef __LR_H__
#define __LR_H__

#include <eigen3/Eigen/Dense>
#include <string>
#include "utils.h"


class LR{
public:
    LR();
	LR(int max_iter,double alpha,double lambda,double tolerance);
    ~LR();
	void fit(Eigen::MatrixXd X,Eigen::VectorXd y,int batch_size,int early_stopping_round,double (*metric)(double* y,double* y_pred,int size)=Utils::accuracy);
	Eigen::VectorXd getW();
	Eigen::VectorXd predict_prob(Eigen::MatrixXd X);
	Eigen::VectorXi predict(Eigen::MatrixXd X);
	void saveWeights(std::string filename);
	void loadWeights(std::string filename);
private:
	Eigen::VectorXd W;
	int max_iter;
	double lambda;  //l2 regulization
	double tolerance;  // error tolence
	double alpha; //learning rate
};



#endif
