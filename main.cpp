#include "read_dataset.hpp"

using namespace std;

int main(){
    neuronal_network neuronal("sign_mnist_train.csv");
    
    for(int i = 0; i < neuronal.dataset.size(); i++){
        for(int j = 0; j < neuronal.dataset[i].size(); j++){
            cout << neuronal.dataset[i][j] << ' ';
        }
        cout << endl;
    }

    return 0;
}