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
		int num_o_nodes; //num of classes, 24 letters of the alphabet without J and Z
		int num_features;
		int num_h_layers;
		int num_h_nodes;
		string activ_func;
		matrix dataset;
		vector<int> labels;
		matrix x_values;

		matrix read_dataset(string file_name);
		double sigmoid(double data);
		double tanh(double data);
		double relu(double data);
		vector<double> softmax(vector<double> vec);
		matrix matrix_mul(matrix hidden, matrix input);
		matrix activation(matrix input);
		matrix weights(int size, int input_size);
		void feed_forward(int input_index, int hidden_nodes, int output_nodes);
		void back_propagation();
		void error(int input_index);
		void print_matrix(matrix mat);
		matrix array_to_mat(vector<double> mat);
	public:
		neural_network();
		neural_network(int, int, int, string, int, int);
		void train(string file_name);

};

neural_network::neural_network(int num_features, int num_h_layers, int num_h_nodes, string activ_func, int num_o_nodes, int learning_rate){
	this->num_h_layers = num_h_layers;
	this->num_features = num_features;
	this->num_h_nodes = num_h_nodes;
	this->num_o_nodes = num_o_nodes;
	this->activ_func = activ_func;
}

void neural_network::train(string file_name){
	this->dataset = this->read_dataset(file_name);
	//int input_size = this->x_values.size();
	int input_size = 1;
	for(int i=0; i<input_size; i++){
		feed_forward(i, this->num_h_nodes,this->num_o_nodes);
	}
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

void neural_network::feed_forward(int input_index, int hidden_nodes, int output_nodes){
	matrix w = weights(hidden_nodes, x_values[input_index].size());
	// FIRST LAYER
	matrix input = array_to_mat(x_values[input_index]);
	matrix result = matrix_mul(w, input);
	matrix result_ih = activation(result);
	// REST OF LAYERS
	if(this->num_h_layers > 1){ //CHANGE FOR MATRIX INSTEAD OF VECTOR
		for(int i=0; i<this->num_h_layers-1; i++){
			matrix result = matrix_mul(w, result_ih);
			matrix result_ih = activation(result);
		}
	}
	// OUTPUT LAYER
	matrix w_o = weights(output_nodes, result_ih.size());
	matrix result_h = matrix_mul(w_o, result_ih);
	matrix result_ho = activation(result_h);
	print_matrix(result_ho);
}

matrix neural_network::weights(int num_hid_nodes, int input_size){
	random_device rd;
    default_random_engine eng(rd());
    uniform_real_distribution<double> distr(-1, 2);
	matrix w;
	for(int i = 0; i < num_hid_nodes; i++){
		vector<double> temp_vector;
		//bias is generated as an additional weight
		for(int j = 0; j < input_size; j++){
			temp_vector.push_back(distr(eng));
		}
		w.push_back(temp_vector);
	}

	return w;
}

matrix neural_network::matrix_mul(matrix weights, matrix input){
	if(weights[0].size() != input.size())
		throw invalid_argument("Matrices columns and rows do not match");
	
	matrix result;
	for(int i = 0; i < weights.size(); i++){
		vector<double> temp_sum;
		double sum = 0.0;
		for(int j = 0; j < weights[i].size(); j++){
			sum += (weights[i][j] * input[0][j]);
		}
		temp_sum.push_back(sum);
		result.push_back(temp_sum);
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
		for(int j = 0; j < this->num_o_nodes; j++){
			sum += exp(vec[j]);
		}
		result.push_back(floor(exponential / sum));
		sum = 0;
		exponential = 0;
	}
	return result;
}


matrix neural_network::activation(matrix input){
	for(int i = 0; i < input.size(); i++){
		if(this->activ_func == "sigmoid")
			input[0][i] = sigmoid(input[0][i]);
		else if(this->activ_func == "tanh")
			input[0][i] = tanh(input[0][i]);
		else
			input[0][i] = relu(input[0][i]);
	}
	return input;
}



void neural_network::back_propagation(){
	
}

void neural_network::print_matrix(matrix mat){
	double max_value = 0;
	int index = 0;
	for(int i = 0; i < mat[0].size(); i++){
		if(mat[0][i] > max_value){
			max_value = mat[0][i];
			index = i;
		}
		cout << mat[0][i] << endl;
	}
	cout << "-------------------------------"<< endl;
	cout << "Max value: " << max_value << endl;
	cout << "With index: " << index << endl;
}

matrix neural_network::array_to_mat(vector<double> mat){
	matrix result;
	for(int i = 0; i < mat.size(); i++){
		vector<double> temp;
		for(int j = 0; j < 1; j++){
			temp.push_back(mat[i]);
		}
		result.push_back(temp);
	}

	return result;
}