#pragma once

#include "globaldef.h"

#include "Neuron.h"
#include "Connection.h"

namespace nn
{
    /* Public namespace function */
    double getRandomNumber();


    /* Forward class declaration */
    class Neuron;


    class NeuralNetwork
    {
        /*****************************************************************************/
        /* Public methods ---------------------------------------------------------- */
    public:

        template<std::size_t numberOfColumns>
        NeuralNetwork(std::array<size_t, numberOfColumns> rowsInColumns);

        template<std::size_t numberOfColumns>
        NeuralNetwork(std::array<size_t, numberOfColumns> rowsInColumns,
            std::array<std::vector<double>, numberOfColumns> biases);

        NeuralNetwork(NeuralNetwork& oldNetwork, double biasVariation = 0.0, double weightVariation = 0.0);

        ~NeuralNetwork() = default;

        std::vector<double> calculate(std::vector<double> input);

        Neuron& getNeuron(size_t column, size_t row) { return m_neurons[column][row]; }
        std::vector<std::vector<Neuron>> getNeuronVectors() { return m_neurons; }
        size_t getRowsInColumn(size_t column) { return m_neurons[column].size();  }
        size_t getNumberOfColumns() { return m_neurons.size(); }

        std::vector<Neuron>& operator [](size_t column) { return m_neurons[column]; }


        /*****************************************************************************/
        /* Private methods --------------------------------------------------------- */
    private:

#pragma endregion



#pragma region Variables
        /*****************************************************************************/
        /* Public variables -------------------------------------------------------- */
    public:

        /*****************************************************************************/
        /* Private variables ------------------------------------------------------- */
    private:
        std::vector<std::vector<Neuron>> m_neurons;

        uint64_t m_iterationNumber = 0;
#pragma endregion
    };


    /*****************************************************************************/
    /* Templates --------------------------------------------------------------- */
    template<std::size_t numberOfColumns>
    NeuralNetwork::NeuralNetwork(std::array<size_t, numberOfColumns> rowsInColumns)
    {
        /* For every columns */
        for (size_t i = 0; i < numberOfColumns; i++)
        {
            /* Get the number of rows in current column */
            size_t rowsInColumn = rowsInColumns[i];

            /* Create a vector associated with that column */
            std::vector<Neuron> columnNeurons;

            for (size_t j = 0; j < rowsInColumn; j++)
            {
                /* Create neurons */
                Neuron neuron(*this, j, i);

                /* Add the new neurons to the vector */
                columnNeurons.push_back(neuron);
            }

            /* Add the new column to the class' columns */
            m_neurons.push_back(columnNeurons);
        }
    }


    template<std::size_t numberOfColumns>
    NeuralNetwork::NeuralNetwork(std::array<size_t, numberOfColumns> rowsInColumns,
        std::array<std::vector<double>, numberOfColumns> biases)
    {

        /* For every columns */
        for (size_t i = 0; i < numberOfColumns; i++)
        {
            /* Get the number of rows in current column */
            size_t rowsInColumn = rowsInColumns[i];

            /* Check the size of the biases vector */
            if (biases[i].size() != rowsInColumn)
            {
                throw;
            }

            /* Create a vector associated with that column */
            std::vector<Neuron> columnNeurons;

            for (size_t j = 0; j < rowsInColumn; j++)
            {
                /* Create neurons */
                Neuron neuron(*this, j, i, biases[i][j]);

                /* Add the new neurons to the vector */
                columnNeurons.push_back(neuron);
            }

            /* Add the new column to the class' columns */
            m_neurons.push_back(columnNeurons);
        }
    }
}