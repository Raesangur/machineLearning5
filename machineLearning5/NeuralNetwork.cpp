#include "globaldef.h"
#include "NeuralNetwork.h"

namespace nn
{
    double getRandomNumber()
    {
        /* Create random number generator */
        static std::ranlux48 randomGenerator(0);

        return (double)randomGenerator();
    }



    NeuralNetwork::NeuralNetwork(NeuralNetwork& oldNetwork,
                                 double biasVariation, double weightVariation)
    {
        /* Increase iteration number */
        m_iterationNumber = oldNetwork.m_iterationNumber + 1;

        /* For every columns */
        for (size_t i = 0; i < oldNetwork.getNumberOfColumns(); i++)
        {
            /* Get the number of rows in current column */
            size_t rowsInColumn = oldNetwork.getRowsInColumn(i);

            /* Create a vector associated with that column */
            std::vector<Neuron> columnNeurons;

            for (size_t j = 0; j < rowsInColumn; j++)
            {
                /* Load old neuron and its connections */
                Neuron& oldNeuron = oldNetwork[i][j];
                std::vector<double> oldConnectionsWeights;
                std::for_each(oldNeuron.getInputConnectionsList().begin(),
                              oldNeuron.getInputConnectionsList().end(),
                    [&](Connection& oldConnection)
                    {
                        oldConnectionsWeights.push_back(oldConnection.getWeight());
                    });


                /* Generate new random bias for each neuron */
                double randomBias = getRandomNumber() * biasVariation;

                /* Create neurons */
                Neuron neuron(*this, j, i, randomBias, oldConnectionsWeights, weightVariation);

                /* Add the new neurons to the vector */
                columnNeurons.push_back(neuron);
            }

            /* Add the new column to the class' columns */
            m_neurons.push_back(columnNeurons);
        }
    }


    std::vector<double> NeuralNetwork::calculate(std::vector<double> input)
    {
        /* Check vector sizes */
        if (input.size() != m_neurons[0].size())
        {
            throw;
        }

        /* Forward the input vector to each neurons of the first layer */
        for (size_t i = 0; i < m_neurons[0].size(); i++)
        {
            m_neurons[0][i].setInput(input[i]);
        }

        /* For each other layer (each column is a layer) */
        for (size_t layer = 0; layer < m_neurons.size(); layer++)
        {
            /* For each row in the layer */
            for (size_t row = 0; row < m_neurons[layer].size(); row++)
            {
                m_neurons[layer][row].calculate();
            }
        }

        std::vector<double> outputValues;

        /* Calculations are complete, get the output values from the last layer */
        for (Neuron& lastLayerNeuron : m_neurons[m_neurons.size() - 1])
        {
            outputValues.push_back(lastLayerNeuron.getOutput());
        }

        return outputValues;
    }
}
