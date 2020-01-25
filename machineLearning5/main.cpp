#include "globaldef.h"
#include "NeuralNetwork.h"
#include "Neuron.h"

#include <iostream>


void printNeuron(nn::Neuron& neuron)
{
    std::cout << neuron.getName() << " Bias:" << neuron.getBias() << std::endl;
}

int main()
{
    std::array<size_t, 2> rows = { 2, 4 };
    std::array<std::vector<double>, 2> biases =
    {
        { 
            {1.0, 2.0} ,
            {1.0, 2.0, 3.0, 4.0}
        }
    };


    nn::NeuralNetwork network(rows, biases);


    for (std::vector<nn::Neuron> columns : network.getNeuronVectors())
    {
        std::for_each(columns.begin(), columns.end(), printNeuron);
    }

    std::vector<double> inputValues = { 1, 0 };

    std::vector<double> returnValues = network.calculate(inputValues);

    for (double& val : returnValues)
    {
        std::cout << val << std::endl;
    }

    return 0;
}