 #pragma once


#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <sstream>
#include <vector>
#include "activ_func.hpp"
#include "layer.hpp"

using namespace std;

typedef vector<vector<double> > matrix;

class NeuralNetwork{
	private:
		vector<string> headers;
		vector<int> labels;
		vector<int> labels_T;
		vector<Layer> layers;
		int num_o_nodes; //num of classes, 24 letters of the alphabet without J and Z
		int num_features;
		int num_h_layers;
		int num_h_nodes;
		int num_classes;
		double learning_rate;
		matrix dataset;
		matrix dataset_T;
		matrix x_values;
		ActivFunc activ_func;

		matrix read_dataset(string file_name, string type);
		matrix weights(int size, int input_size);
		matrix feed_forward(int input_index);
		matrix variation_forward(int input_index);
		void back_propagation(matrix result_ho, int i);
		matrix get_target_arr(int input_index, string type); 
		void print_matrix(matrix mat);
		matrix array_to_mat(vector<double> mat);
		matrix matrix_mul(matrix hidden, matrix input);
		matrix set_mat(int rows, int columns);
		matrix substract_matrices(matrix result, matrix result_i);
		matrix scalar_mul(matrix mat);
		matrix transpose(matrix mat);
		matrix element_mul(matrix error, matrix derivative);
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
	this->learning_rate = learning_rate;
	this->activ_func = ActivFunc(activ_func_type,num_classes);
}

//MAIN FUNCTIONS

void NeuralNetwork::train(string file_name){
	this->dataset = this->read_dataset(file_name, " ");
	//int input_size = this->x_values.size();
	int num_tests = 5;
	int input_size = 1;
	//First feed_forward
	
	for(int i=0; i<input_size; i++){
		matrix result_ho = feed_forward(i);
		back_propagation(result_ho, i);
		for(int j = 0; j < num_tests; j++){
			matrix new_result_ho = variation_forward(i);
			back_propagation(new_result_ho, i);
		}
	}
}

void NeuralNetwork::test(string file_name){
	this->dataset_T = this->read_dataset(file_name, "testing");
	//int input_size = this->x_values.size();
	int input_size = 1;
	for(int i=0; i<input_size; i++){
		matrix targets = get_target_arr(i,"testing");
		matrix result = variation_forward(i);
		print_matrix(result);
	}
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

// SECONDARY FUNCTIONS

matrix NeuralNetwork::feed_forward(int input_index){
	// LAYER I TO H0
	int x_size = x_values[input_index].size();
	matrix w = weights(this->num_h_nodes, x_size);
	matrix input = array_to_mat(x_values[input_index]);
	matrix result = matrix_mul(w, input);
	matrix result_activ = activ_func.activation(result);

	Layer layer_h0(0,input,w,result_activ);
	this->layers.push_back(layer_h0);
	
	// LAYER H0 TO HN
	int i=1;
	if(this->num_h_layers > 1){
		for(; i<this->num_h_layers; i++){
			Layer layer_hn(i,result_activ);
			matrix new_w = weights(this->num_h_nodes, this->num_h_nodes);
			result = matrix_mul(new_w, result_activ);
			result_activ = activ_func.activation(result);
			layer_hn.weights = new_w;
			layer_hn.outputs = result_activ;
			this->layers.push_back(layer_hn);
		}
	}

	// OUTPUT LAYER
	matrix w_o = weights(this->num_o_nodes, this->num_h_nodes);
	matrix result_h = matrix_mul(w_o, result_activ);
	matrix result_ho = activ_func.activation(result_h);
	Layer layer_ho(i,result_activ,w_o,result_ho);
	this->layers.push_back(layer_ho);

	return result_ho;
}

matrix NeuralNetwork::variation_forward(int input_index){
	// LAYER I TO H0
	int x_size = x_values[input_index].size();
	matrix w = this->layers[0].weights;
	matrix input = array_to_mat(x_values[input_index]);
	matrix result = matrix_mul(w, input);
	matrix result_activ = activ_func.activation(result);

	this->layers[0].outputs = result_activ;
	
	// LAYER H0 TO HN
	int i=1;
	if(this->num_h_layers > 1){
		for(; i<this->num_h_layers; i++){
			this->layers[i].inputs = result_activ;
			matrix new_w = this->layers[i].weights;
			result = matrix_mul(new_w, result_activ);
			result_activ = activ_func.activation(result);
			this->layers[i].outputs = result_activ;
		}
	}

	// OUTPUT LAYER
	matrix w_o = this->layers[i].weights;
	matrix result_h = matrix_mul(w_o, result_activ);
	matrix result_ho = activ_func.activation(result_h);
	this->layers[i].inputs = result_activ;
	this->layers[i].outputs = result_ho;

	return result_ho;
}

void NeuralNetwork::back_propagation(matrix result_ho, int input_index){
	matrix output_errors;
	matrix temp_errors;
	for(int i=this->num_h_layers; i>=0; i--){
		matrix delta_values;
		if(i==this->num_h_layers){
			matrix targets = get_target_arr(input_index,"training");
			output_errors = substract_matrices(result_ho,targets);
			// 1. learning rate * errors
			matrix rate_mat = scalar_mul(output_errors); //24x1

			matrix deriv = activ_func.derivatives(result_ho); //24x1

			// 2. learning rate * errors * derivatives
			matrix mult_mat = element_mul(rate_mat, deriv); //24x1

			// 3. learning rate * errors * derivatives * hidden_input
			matrix mat_H = transpose(this->layers[i].inputs); //1x10
			delta_values = matrix_mul(mult_mat, mat_H);
			temp_errors = output_errors;
		} else {
			matrix weights_T = transpose(this->layers[i+1].weights);
			matrix hidden_errors = matrix_mul(weights_T,temp_errors);
			// 1. learning rate * errors
			matrix rate_mat = scalar_mul(hidden_errors);

			matrix mat_H = transpose(this->layers[i].inputs);
			matrix deriv = activ_func.derivatives(this->layers[i].outputs);

			// 2. learning rate * errors * derivatives
			matrix mult_mat = element_mul(rate_mat, deriv);
			// 3. learning rate * errors * derivatives * hidden_input
			delta_values = matrix_mul(mult_mat, mat_H);
			temp_errors = hidden_errors;
		}
		this->layers[i].weights = substract_matrices(delta_values, this->layers[i].weights); 
	}
}

// AUX FUNCTIONS
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

matrix NeuralNetwork::get_target_arr(int input_index, string type){
	vector<double> targets(this->num_o_nodes, 0.0);
	if(type != "testing")
		targets[this->labels[input_index]] = 1.0;
	else
		targets[this->labels_T[input_index]] = 1.0;
	matrix target_mat = array_to_mat(targets);
	return target_mat;
}

// MATRIX FUNCTIONS

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
	
	int cols = input[0].size();
	int rows = weights.size();
	matrix result(rows,vector<double>(cols,0));
	for(int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++){
			for(int k = 0; k < weights[0].size(); k++){
				result[i][j] += weights[i][k] * input[k][j];
			}
		}
	}
	return result;
}

matrix NeuralNetwork::set_mat(int rows, int columns){
	matrix output;
	return output;
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
	for(int i = 0; i < targets.size(); i++){
		vector<double> temp;
		for(int j = 0; j < targets[0].size(); j++){
			temp.push_back(targets[i][j] - result_ho[i][j]);
		}
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

matrix NeuralNetwork::element_mul(matrix error, matrix derivative){
	matrix new_mat;
	for(int i = 0; i < error.size(); i++){
		vector<double> temp;
		for(int j = 0; j < error[0].size(); j++){
			temp.push_back(error[i][j] * derivative[i][j]);
		}
		new_mat.push_back(temp);
	}
	return new_mat;
}
