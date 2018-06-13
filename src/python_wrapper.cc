#include <iostream>
#include "lr.h"
#include <ctime>

using namespace Eigen;
using namespace std;

void gen_random(char *s, int len) {
    srand (time(NULL));
    static const char alphanum[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    for (int i = 0; i < len; ++i) {
        s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
    }
    s[len] = 0;
}

extern "C" void fit(double** features,int* labels,int row,int col,int max_iter,double alpha,double lambda,double tolerance,int early_stopping_round,int batch_size,char* ret,double (*metric)(double* y,double* pred,int size)=Utils::accuracy){
    //initialize data of Eigen type
    MatrixXd X(row,col);
    VectorXd y(row);
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            X(i,j) = features[i][j];
        }
        y(i) = labels[i];
    }

    //train the logistic regression model
    LR clf = LR(max_iter,alpha,lambda,tolerance);
    clf.fit(X,y,batch_size,early_stopping_round,metric);

    //save the model weights
    char* fmodel = new char[21];
    gen_random(fmodel,20);
    string model_path = "/tmp/"+string(fmodel);
    clf.saveWeights(model_path);
    strcpy(ret,model_path.c_str());
}


extern "C" void predict_prob(double** features,int row,int col,char* fmodel,double* ret){
    LR clf = LR();
    clf.loadWeights(fmodel);
    MatrixXd X(row,col);
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            X(i,j) = features[i][j];
        }
    }
    VectorXd pred = clf.predict_prob(X);

    for(int i=0;i<row;i++){
        ret[i] = pred(i);
    }
}


extern "C" void predict(double** features,int row,int col,char* fmodel,int* ret){
    double* prob = new double[row];
    predict_prob(features,row,col,fmodel,prob);
    for(int i=0;i<row;i++){
        ret[i] = prob[i]>0.5?1:0;
    }
}

int main(){


    int row=10,col=2;
    double** features = new double *[row];
    for(int i=0;i<row;i++){
        features[i] = new double[col];
    }
    int* labels = new int[row];
    
    double features_value[row*col] = {1.0,0.8,2.0,1.7,3.0,2.5,4.0,3.6,5.0,4.9,1.0,1.2,2.0,2.5,3.0,3.4,4.0,4.5,5.0,6.0};
    int labels_value[row] = {0,0,0,0,0,1,1,1,1,1};
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            features[i][j] = features_value[i*col+j];
        }
        labels[i] = labels_value[i];
    }

    char* ret = new char[26];
    fit(features,labels,row,col,200,0.01,0.0,0.01,10,64,ret);
    cout<<ret<<endl;

    int* pred = new int[row];
    predict(features,row,col,ret,pred);
    for(int i=0;i<row;i++){
        cout<<pred[i]<<",";
    }
}

