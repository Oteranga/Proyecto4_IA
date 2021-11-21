 #pragma once


#include <iostream>
#include <math.h>
#include <cmath>
#include <vector>
using namespace std;

typedef vector<vector<double> > matrix;

class ActivFunc{
    public:
        matrix derivatives(matrix input); 
        matrix activation(matrix input);
        ActivFunc(string type_name, int num_classes);
		ActivFunc(){};
    private:
        string type;
        int num_classes;
        double sigmoid(double data);
		double tanh(double data);
		double relu(double data);
        vector<double> softmax(vector<double> vec);
};

ActivFunc::ActivFunc(string type_name, int num_classes){
    this->type = type_name;
    this->num_classes = num_classes;
}

double ActivFunc::sigmoid(double data){
	double result = 1 / (1 + exp(-data));
	return result;
}

double ActivFunc::tanh(double data){
	double result = (1 - exp(-2 * data)) / (1 + exp(-2 * data));
	return result;
}

double ActivFunc::relu(double data){
	if(data > 0)
		return data;
	else if(data <= 0)
		return 0;
}

matrix ActivFunc::derivatives(matrix input){
	for(int i = 0; i < input.size(); i++){
		if(this->type == "sigmoid")
			//input[i][0] = sigmoid(input[i][0]) * (1 - sigmoid(input[i][0]));
			input[i][0] = input[i][0] * (1 - input[i][0]);
		else if(this->type == "tanh")
			//input[i][0] = 1 - pow(tanh(input[i][0]), 2);
			input[i][0] = 1 - pow(input[i][0], 2);
		else
			input[i][0] = relu(input[i][0]);
	}
	return input;
}

matrix ActivFunc::activation(matrix input){
	for(int i = 0; i < input.size(); i++){
		if(this->type == "sigmoid")
			input[i][0] = sigmoid(input[i][0]);
		else if(this->type == "tanh")
			input[i][0] = tanh(input[i][0]);
		else if(this->type == "relu")
			input[i][0] = relu(input[i][0]);
	}
	return input;
}

vector<double> ActivFunc::softmax(vector<double> vec){
	double exponential = 0;
	double sum = 0;
	vector<double> result;
	for(int i = 0; i < vec.size(); i++){
		exponential = exp(vec[i]);
		for(int j = 0; j < this->num_classes; j++){
			sum += exp(vec[j]);
		}
		result.push_back(floor(exponential / sum));
		sum = 0;
		exponential = 0;
	}
	return result;
}
