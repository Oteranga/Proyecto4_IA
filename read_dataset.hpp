#pragma once

#include <math.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <sstream>
#include <vector>

using namespace std;

typedef vector<vector<double> > matrix;

class neural_network{
	private:
		vector<string> headers;
		int num_classes = 24; //24 letters of the alphabet without J and Z
		int num_features = 784;
		int layers = 2;
		matrix dataset;
		vector<int> labels;
		matrix x_values;

		matrix read_dataset(string file_name);
		double sigmoid(double data);
		double tanh(double data);
		double relu(double data);
		vector<double> softmax(vector<double> vec);
		vector<double> matrix_mul(matrix hidden, vector<double> input);
		vector<double> activation(vector<double> input);
		matrix weights(int size, int input_size);
		void back_propagation();
		void print_matrix(vector<double> mat);
	public:
		neural_network();
		neural_network(string file_name);
		void feed_forward(int hidden_nodes, int output_nodes);

};

neural_network::neural_network(string file_name){
	this->dataset = this->read_dataset(file_name);
}


matrix neural_network::read_dataset(string file_name){
    fstream file;
    file.open(file_name, ios::in);
    matrix data;
	string line;

	getline(file, line);
	istringstream ss_header(line);
	string token;

	while(getline(ss_header, token, ',')) {
		headers.push_back(token);
	}

	while(getline(file, line)){
		vector<double> single_row;
		stringstream ss(line);
		string value;

		while(getline(ss, value, ',')) {
			single_row.push_back(stod(value));
		}
		data.push_back(single_row);
		labels.push_back(single_row[0]);
		single_row.erase(single_row.begin());
		single_row.push_back(1.0);
		x_values.push_back(single_row);
	}
	file.close();
	return data;
}

void neural_network::feed_forward(int hidden_nodes, int output_nodes = 24){
	matrix w = weights(hidden_nodes, x_values[0].size());
	vector<double> result = matrix_mul(w, x_values[0]);
	//Activation method
	vector<double> result_ih = activation(result);

	matrix w_o = weights(output_nodes, result_ih.size());
	vector<double> result_h = matrix_mul(w_o, result_ih);
	vector<double> result_ho = activation(result_h);
	print_matrix(result_ho);
}

matrix neural_network::weights(int hidden_layer_size, int input_size){
	random_device rd;
    default_random_engine eng(rd());
    uniform_real_distribution<double> distr(-1, 2);
	matrix w;
	for(int i = 0; i < hidden_layer_size; i++){
		vector<double> temp_vector;
		//bias is generated as an additional weight
		for(int j = 0; j < input_size; j++){
			temp_vector.push_back(distr(eng));
		}
		w.push_back(temp_vector);
	}

	return w;
}

vector<double> neural_network::matrix_mul(matrix weights, vector<double> input){
	if(weights[0].size() != input.size())
		throw invalid_argument("Matrices columns and rows do not match");
	
	vector<double> result;
	for(int i = 0; i < weights.size(); i++){
		double sum = 0.0;
		for(int j = 0; j < weights[i].size(); j++){
			sum += (weights[i][j] * input[j]);
		}
		result.push_back(sum);
	}

	return result;
}

double neural_network::sigmoid(double data){
	double result = 1 / (1 + exp(-data));
	return result;
}

double neural_network::tanh(double data){
	double result = (1 - exp(-2 * data)) / (1 + exp(-2 * data));
	return result;
}

double neural_network::relu(double data){
	if(data > 0)
		return data;
	else if(data <= 0)
		return 0;
}

vector<double> neural_network::softmax(vector<double> vec){
	double exponential = 0;
	double sum = 0;
	vector<double> result;
	for(int i = 0; i < vec.size(); i++){
		exponential = exp(vec[i]);
		for(int j = 0; j < num_classes; j++){
			sum += exp(vec[j]);
		}
		result.push_back(floor(exponential / sum));
		sum = 0;
		exponential = 0;
	}
	return result;
}


vector<double> neural_network::activation(vector<double> input){
	for(int i = 0; i < input.size(); i++){
		input[i] = sigmoid(input[i]);
	}
	return input;
}



void neural_network::back_propagation(){
	
}

void neural_network::print_matrix(vector<double> mat){
	double max_value = 0;
	int index = 0;
	for(int i = 0; i < mat.size(); i++){
		if(mat[i] > max_value){
			max_value = mat[i];
			index = i;
		}
		cout << mat[i] << endl;
	}
	cout << "-------------------------------"<< endl;
	cout << "Max value: " << max_value << endl;
	cout << "With index: " << index << endl;
}