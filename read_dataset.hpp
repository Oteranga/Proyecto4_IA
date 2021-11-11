#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

using namespace std;

typedef vector<vector<int> > matrix;

class neuronal_network{
	private:
		vector<string> headers;
		int num_classes;
		matrix dataset;
	public:
		neuronal_network();
		neuronal_network(string file_name);
		matrix read_dataset(string file_name);
		double sigmoid(double data);
		double tanh(double data);
		double relu(double data);
		double softmax(double vec_value, vector<double> vec);

};

neuronal_network::neuronal_network(string file_name){
	this->dataset = read_dataset(file_name);
}

matrix neuronal_network::read_dataset(string file_name){
    fstream file;
    file.open(file_name, ios::in);
    vector<vector<int> > data;
	string line;

	getline(file, line);
	istringstream ss_header(line);
	string token;

	while(getline(ss_header, token, ',')) {
		headers.push_back(token);
	}

	while(getline(file, line)){
		vector<int> single_row;
		stringstream ss(line);
		string value;

		while(getline(ss, value, ',')) {
			single_row.push_back(stoi(value));
		}
		data.push_back(single_row);
	}
	file.close();
	return data;
}

double neuronal_network::sigmoid(double data){
	double result = 1 / (1 + exp(-data));
	return result;
}

double neuronal_network::tanh(double data){
	double result = (1 - exp(-2 * data)) / (1 + exp(-2 * data));
	return result;
}

double neuronal_network::relu(double data){
	if(data > 0)
		return data;
	else if(data <= 0)
		return 0;
}

double neuronal_network::softmax(double vec_value, vector<double> vec){
	double result = exp(vec_value);
	double sum = 0;
	for(int i = 0; i < num_classes; i++){
		sum += exp(vec[i]);
	}
	return result / sum;
}