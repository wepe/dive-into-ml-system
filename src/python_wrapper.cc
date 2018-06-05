#include <iostream>
#include "lr.h"

using namespace Eigen;
using namespace std;


void gen_random(char *s, int len) {
    static const char alphanum[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    for (int i = 0; i < len; ++i) {
        s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
    }
    s[len] = 0;
}

extern "C" char* fit(double** features,int* labels,int row,int col,int max_iter,double alpha,double lambda,double tolerance){
    //initialize data of Eigen type
    MatrixXd X(row,col);
    VectorXi y(row);
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            X(i,j) = features[i][j];
        }
        y(i) = labels[i];
    }

    //train the logistic regression model
    LR clf = LR(max_iter,alpha,lambda,tolerance);
    clf.fit(X,y);

    //save the model weights
    char* fmodel = new char[20];
    gen_random(fmodel,20);
    string model_path = "/tmp/"+string(fmodel);
    clf.saveWeights(model_path);
    char *ret = new char[model_path.length()+1];
    strcpy(ret,model_path.c_str());
    return ret;
}


extern "C" double* predict_prob(double** features,int row,int col,char* fmodel){
    LR clf = LR();
    clf.loadWeights(fmodel);
    MatrixXd X(row,col);
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            X(i,j) = features[i][j];
        }
    }
    VectorXd pred = clf.predict_prob(X);

    double* ret = new double[row];
    for(int i=0;i<row;i++){
        ret[i] = pred(i);
    }

    return ret;
}


extern "C" int* predict(double** features,int row,int col,char* fmodel){
    double* prob = predict_prob(features,row,col,fmodel);
    int* ret = new int[row];
    for(int i=0;i<row;i++){
        ret[i] = prob[i]>0.5?1:0;
    }
    return ret;
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

    char* ret = fit(features,labels,row,col,200,0.01,0.0,0.01);
    cout<<ret<<endl;
}

