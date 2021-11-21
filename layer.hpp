#pragma once


#include <iostream>
#include <vector>
using namespace std;

typedef vector<vector<double> > matrix;

struct Layer{
    int layer_id;
    matrix inputs; //inputs del layer
    matrix weights; //pesos que ingresan al layer
    matrix outputs; //inputs del sgte layer || output luego de activ func
    Layer(int, matrix, matrix, matrix);
    Layer(int, matrix);
};

Layer::Layer(int layer_id, matrix inputs, matrix weights, matrix outputs){
    this->layer_id = layer_id;
    this->inputs = inputs;
    this->outputs = outputs;
    this->weights = weights;
}

Layer::Layer(int layer_id, matrix inputs){
    this->layer_id = layer_id;
    this->inputs = inputs;
}
