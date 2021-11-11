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
		//matrix dataset;
	public:
		matrix dataset;
		neuronal_network();
		neuronal_network(string file_name);
		matrix read_dataset(string file_name);
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