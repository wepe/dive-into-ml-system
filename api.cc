#include <iostream>
#include "lr.h"

using namespace Eigen;
using namespace std;

extern "C" double* train(double** features,int* labels,int row,int col){

    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            cout<<i<<","<<j<<","<<features[i][j]<<endl;
        }
        cout<<"label:"<<labels[i]<<endl;
    }

    MatrixXd X(row,col);
    VectorXi y(row);
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            X(i,j) = features[i][j];
        }
        y(i) = labels[i];
    }

    LR clf = LR(200,0.01,0.05,0.01);
    clf.fit(X,y);

    VectorXd W = clf.getW();
    double* ret = new double[col+1];
    for(int i=0;i<=col;i++){
        ret[i] = W(i);
    }
    return ret;
}

extern "C" void test_func(){
    cout<<"this is test function!"<<endl;
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

    double* ret = train(features,labels,row,col);
    for(int i=0;i<=col;i++){
        cout<<ret[i]<<endl;
    }
}

