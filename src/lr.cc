#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <boost/format.hpp>
#include "lr.h"
#include "utils.h"
#include <omp.h>
#include <ctime>

using namespace std;
using namespace Eigen;

LR::LR(){
    LR(100,0.01,0.01,0.001);
}

LR::LR(int max_iter,double alpha,double lambda,double tolerance){
    this->lambda = lambda; 
    this->max_iter = max_iter;
    this->tolerance = tolerance;
    this->alpha = alpha;
}

LR::~LR(){}

void LR::fit(MatrixXd X,VectorXd y,int batch_size,int early_stopping_round,double (*metric)(double* y,double* pred,int size)){
    //learn VectorXd W, consider reg,max_iter,tol.
    srand(time(NULL));
    W = VectorXd::Random(X.cols()+1);  //the last column of weight represent bias
    MatrixXd X_new(X.rows(),X.cols()+1);
    X_new<<X,MatrixXd::Ones(X.rows(),1);  //last column is 1.0

    MatrixXd X_batch;
    VectorXd y_batch;
    MatrixXd X_new_batch;
    //perform early stopping
    double best_acc = -1.0;
    int become_worse_round = 0;
    for(int iter=0;iter<max_iter;iter++){
        //index of this batch samples
        int start_idx = (batch_size*iter)%(static_cast<int>(X.rows()));
        int end_idx = min(start_idx+batch_size,static_cast<int>(X.rows()));
        
        X_batch = Utils::slice(X,start_idx,end_idx-1);
        y_batch = Utils::slice(y,start_idx,end_idx-1);
        X_new_batch = Utils::slice(X_new,start_idx,end_idx-1);
        
        //
        VectorXd y_pred = predict_prob(X_batch);
        VectorXd E = y_pred - y_batch;

        //W:= (1-lambda/n_samples)W-alpha*X^T*E
        //reference : http://blog.csdn.net/pakko/article/details/37878837
        W = (1.0-lambda/batch_size)*W - alpha*X_new_batch.transpose()*E;
	    
        //calculate the logloss and accuracy after this step
        y_pred = predict_prob(X_batch);
        
        double loss = Utils::crossEntropyLoss(y_batch,y_pred);
        double acc = metric(Utils::VectorXd_to_double_array(y_batch),Utils::VectorXd_to_double_array(y_pred),end_idx-start_idx);
        cout<<boost::format("Iteration: %d, logloss:%.5f, accuracy:%.5f") %iter %loss %acc<< endl;
		
        //when loss<tolerance, break
        if(loss<=tolerance) break;

        //perform early stopping
        if(acc<best_acc){
            become_worse_round += 1;
        }else{
            become_worse_round = 0;
            best_acc = acc;
        }
        if(become_worse_round>=early_stopping_round){
            cout<<"Early stopping."<<endl;
            break;
        }
    }
}


VectorXd LR::predict_prob(MatrixXd X){
    //predict the probability (of label 1) for given data X
    MatrixXd X_new(X.rows(),X.cols()+1);
    X_new<<X,MatrixXd::Ones(X.rows(),1);
    int num_samples = X_new.rows();
    VectorXd y_pred_prob = VectorXd::Zero(num_samples);
    #pragma omp parallel for
    for(int num=0;num<num_samples;num++){
        y_pred_prob(num) = Utils::sigmod(X_new.row(num).dot(W));
    }
    return y_pred_prob;
}


VectorXi LR::predict(MatrixXd X){
    //predict the label for given data X
    VectorXd y_pred_prob = predict_prob(X);
    VectorXi y_pred(y_pred_prob.size());
    #pragma omp parallel for
    for(int num=0;num<y_pred_prob.size();num++){
        y_pred(num) = y_pred_prob(num)>0.5?1:0;
    }
    return y_pred;
}


Eigen::VectorXd LR::getW(){
    return W;
}

void LR::saveWeights(std::string fpath){
    //save the model (save the weight ). 
    std::ofstream ofile;
    ofile.open(fpath.c_str());
    if (!ofile.is_open()){
        std::cerr<<"Can not open the file when call LR::saveWeights"<<std::endl;
        return;
    }
    //W write into the file
    for(int i=0;i<W.size()-1;i++){
        ofile<<W(i)<<" ";
    }
    ofile<<W(W.size()-1);
    ofile.close();
}


void LR::loadWeights(std::string fpath){
    //load the model (load the weight ) from filename.
    std::ifstream ifile;
    ifile.open(fpath.c_str());
    if (!ifile.is_open()){
        std::cerr<<"Can not open the file when call LR::loadWeights"<<std::endl;
        return;
    }
    //read the weights into vector<double>
    std::string line;
    std::vector<double> weights;
    getline(ifile,line);    //only one line
    std::stringstream ss(line); 
    double tmp;
    while(!ss.eof()){
        ss>>tmp;
        weights.push_back(tmp);
    }
    //initialize VectorXd with std::vector
    W = VectorXd::Map(weights.data(),weights.size());
    ifile.close();
}
