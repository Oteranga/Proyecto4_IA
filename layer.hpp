#pragma once


#include <iostream>
#include <vector>
using namespace std;

typedef vector<vector<double> > matrix;

struct Layer{
    int layer_id;
    int num_h_nodes;
    matrix inputs;
    Layer(int layer_id, int num_h_nodes, matrix inputs);
};

Layer::Layer(int layer_id, int num_h_nodes, matrix inputs){
    this->layer_id = layer_id;
    this->num_h_nodes = num_h_nodes;
    this->inputs = inputs;
}