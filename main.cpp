#include "neural_network.hpp"

using namespace std;

int main(){
    NeuralNetwork neural(784,2,10,"sigmoid",0.05);
    neural.train("sign_mnist_train.csv");
    neural.test("sign_mnist_test.csv");
    return 0;
}