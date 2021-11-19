 #pragma once


#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <sstream>
#include <vector>
#include "activ_func.hpp"

using namespace std;

typedef vector<vector<double> > matrix;

class NeuralNetwork{
	private:
		vector<string> headers;
		vector<int> labels;
		vector<int> labels_T;
		vector<matrix> layers;
		int num_o_nodes; //num of classes, 24 letters of the alphabet without J and Z
		int num_features;
		int num_h_layers;
		int num_h_nodes;
		int num_classes;
		double learning_rate;
		string activ_func;
		matrix dataset;
		matrix dataset_T;
		matrix x_values;
		ActivFunc activ_func;

		matrix read_dataset(string file_name, string type);
		matrix weights(int size, int input_size);
		matrix feed_forward(int input_index, int hidden_nodes, int output_nodes);
		void back_propagation(matrix result_ho, int i);
		matrix get_target_arr(int input_index, string type); 
		void print_matrix(matrix mat);
		matrix array_to_mat(vector<double> mat);
		matrix matrix_mul(matrix hidden, matrix input);
		matrix substract_matrices(matrix result, matrix result_i);
		matrix scalar_mul(matrix mat);
		matrix transpose(matrix mat);
	public:
		NeuralNetwork();
		NeuralNetwork(int, int, int, string, double);
		void train(string file_name);
		void test(string file_name);
};

NeuralNetwork::NeuralNetwork(int num_features, int num_h_layers, int num_h_nodes, string activ_func_type, double learning_rate){
	this->num_h_layers = num_h_layers;
	this->num_features = num_features;
	this->num_h_nodes = num_h_nodes;
	this->num_o_nodes = 24;
	this->num_classes = num_o_nodes+1;
	this->activ_func = activ_func;
	this->learning_rate = learning_rate;
	this->activ_func = ActivFunc(activ_func_type,num_classes);
}

void NeuralNetwork::train(string file_name){
	this->dataset = this->read_dataset(file_name, " ");
	//int input_size = this->x_values.size();
	int num_tests = 10000;
	int input_size = 1;
	for(int i=0; i<input_size; i++){
		for(int j = 0; j < num_tests; j++){
			matrix result_ho = feed_forward(i, this->num_h_nodes,this->num_o_nodes);
			back_propagation(result_ho, i);
		}
	}
}

void NeuralNetwork::test(string file_name){
	this->dataset_T = this->read_dataset(file_name, "testing");
	//int input_size = this->x_values.size();
	int input_size = 1;
	for(int i=0; i<input_size; i++){
		matrix targets = get_target_arr(i,"testing");
		matrix result = feed_forward(i, this->num_h_nodes, this->num_o_nodes);
		print_matrix(result);
	}
}

matrix NeuralNetwork::get_target_arr(int input_index, string type){
	vector<double> targets(this->num_classes, 0.0);
	if(type != "testing")
		targets[this->labels[input_index]] = 1.0;
	else
		targets[this->labels_T[input_index]] = 1.0;
	matrix target_mat = array_to_mat(targets);
	return target_mat;
}

matrix NeuralNetwork::read_dataset(string file_name, string type){
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
		if(type != "testing"){
			data.push_back(single_row);
			labels.push_back(single_row[0]);
			single_row.erase(single_row.begin());
			single_row.push_back(1.0);
			x_values.push_back(single_row);
		} else {
			labels_T.push_back(single_row[0]);
			single_row.erase(single_row.begin());
			single_row.push_back(1.0);
			data.push_back(single_row);
		}
	}
	file.close();
	return data;
}

matrix NeuralNetwork::feed_forward(int input_index, int hidden_nodes, int output_nodes){
	matrix w = weights(hidden_nodes, x_values[input_index].size());
	// FIRST LAYER
	matrix input = array_to_mat(x_values[input_index]);
	matrix result = matrix_mul(w, input);
	matrix result_ih = activ_func.activation(result);
	Layer layer = 
	
	//HIDDEN LAYERS
	if(this->num_h_layers > 1){
		for(int i=0; i<this->num_h_layers-1; i++){
			matrix new_w = weights(hidden_nodes, hidden_nodes);
			result = matrix_mul(new_w, result_ih);
			result_ih = activ_func.activation(result);
		}
	}

	// OUTPUT LAYER
	matrix w_o = weights(output_nodes, hidden_nodes);
	matrix result_h = matrix_mul(w_o, result_ih);
	matrix result_ho = activ_func.activation(result_h);
	//print_matrix(result_ho);
	return result_ho;
}

matrix NeuralNetwork::weights(int num_hid_nodes, int input_size){
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


void NeuralNetwork::back_propagation(matrix result_ho, int input_index){
	matrix targets = get_target_arr(input_index,"training");
	matrix errors = substract_matrices(result_ho,targets);
	matrix rate_mat = scalar_mul(errors);
	matrix mat_T = transpose(result_ih);
	matrix deriv = derivatives(result_ho);
	matrix mult_mat = matrix_mul(rate_mat, deriv);
	matrix final = matrix_mul(mult_mat, mat_T);
}

void NeuralNetwork::print_matrix(matrix mat){
	double max_value = 0;
	int index = 0;
	for(int i = 0; i < mat.size(); i++){
		if(mat[i][0] > max_value){
			max_value = mat[i][0];
			index = i;
		}
		cout << mat[i][0] << endl;
	}
	cout << "-------------------------------"<< endl;
	cout << "Max value: " << max_value << endl;
	cout << "With index: " << index << endl;
}

matrix NeuralNetwork::matrix_mul(matrix weights, matrix input){
	if(weights[0].size() != input.size())
		throw invalid_argument("Matrices columns and rows do not match");
	
	matrix result;
	for(int i = 0; i < weights.size(); i++){
		vector<double> temp_sum;
		double sum = 0.0;
		for(int j = 0; j < weights[i].size(); j++){
			sum += (weights[i][j] * input[j][0]);
		}
		temp_sum.push_back(sum);
		result.push_back(temp_sum);
	}
	return result;
}

matrix NeuralNetwork::array_to_mat(vector<double> arr){
	matrix result;
	for(int i = 0; i < arr.size(); i++){
		vector<double> temp;
		temp.push_back(arr[i]);
		result.push_back(temp);
	}
	return result;
}

matrix NeuralNetwork::substract_matrices(matrix result_ho, matrix targets){
	matrix errors; 
	for(int j = 0; j < result_ho.size(); j++){
		vector<double> temp;
		temp.push_back(targets[j][0] - result_ho[j][0]);
		errors.push_back(temp);
	}
	return errors;
}

matrix NeuralNetwork::scalar_mul(matrix mat){
	for(int i = 0; i < mat.size(); i++){
		for(int j = 0; j < mat[i].size(); j++){
			mat[i][j] = mat[i][j] * this->learning_rate;
		}
	}

	return mat;
}

matrix NeuralNetwork::transpose(matrix mat){
	matrix new_mat;
	for(int i = 0; i < mat[0].size(); i++){
		vector<double> temp;
		for(int j = 0; j < mat.size(); j++){
			temp.push_back(mat[j][i]);
		}
		new_mat.push_back(temp);
	}

	return new_mat;
}