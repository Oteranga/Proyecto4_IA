#include "read_dataset.hpp"

using namespace std;

int main(){
    neural_network neural("sign_mnist_train.csv");
    neural.feed_forward(10);

    return 0;
}