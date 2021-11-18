#include "neural_network.hpp"

using namespace std;

int main(){
    neural_network neural(784,1,10,"sigmoid",24,0.05);
    neural.train("sign_mnist_train.csv");
    return 0;
}