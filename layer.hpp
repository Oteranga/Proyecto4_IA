#pragma once


#include <iostream>
using namespace std;

typedef vector<vector<double> > matrix;

class Layer{
    int num_h_nodes;
    matrix inputs;
    Layer(int num_h_nodes, matrix inputs);
};

Layer::Layer(int num_h_nodes, matrix inputs){
    this->num_h_nodes = num_h_nodes;
    this->inputs = inputs;
}