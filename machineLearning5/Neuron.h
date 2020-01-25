#pragma once
#include "globaldef.h"

#include "Connection.h"
#include "NeuralNetwork.h"

namespace nn
{
    /* Importing class */
    class Connection;
    class NeuralNetwork;

    class Neuron
    {
        /*****************************************************************************/
        /* Public methods ---------------------------------------------------------- */
    public:
        Neuron(NeuralNetwork& network, size_t row, size_t column,
            double bias = 0);
        Neuron(NeuralNetwork& network, size_t row, size_t column,
            const std::string& name, double bias = 0.0);
        Neuron(NeuralNetwork& network, size_t row, size_t column, double bias,
            std::vector<double>& previousWeights, double weightVariation = 0.0);
        ~Neuron() = default;

        double calculate();

        size_t getRow() { return m_row; }
        size_t getColumn() { return m_column; }
        double getBias() { return m_bias; } 
        double getOutput() { return m_output; } 
        std::string& getName() { return m_name; } 

        std::vector<Connection> getInputConnectionsList()
        {
            std::vector<Connection> tempVector;
            std::for_each(m_inputConnections.begin(), m_inputConnections.end(),
                [&](std::shared_ptr<Connection> connectionPtr)
                {
                    tempVector.push_back(*connectionPtr);
                });

            return tempVector;
        }
        std::vector<Connection> getOutputConnectionsList()
        {
            std::vector<Connection> tempVector;
            std::for_each(m_outputConnections.begin(), m_outputConnections.end(),
                [&](std::shared_ptr<Connection> connectionPtr)
                {
                    tempVector.push_back(*connectionPtr);
                });

            return tempVector;
        }

        void setBias(double bias) { m_bias = bias; }
        void setInput(double input) { m_input = input; }

        /*****************************************************************************/
        /* Private methods --------------------------------------------------------- */
    private:
        void getInputConnections(void);
        void getInputConnections(double weightVariation, std::vector<double>& previousWeights);
        double getInputValues(void);
        double activationFunction(double value);

        /*****************************************************************************/
        /* Public variables -------------------------------------------------------- */
    public:
        size_t m_row;
        size_t m_column;

        /*****************************************************************************/
        /* Private variables ------------------------------------------------------- */
    private:
        double m_bias = 0.0;
        double m_output = 0.0;
        double m_input = 0.0;

    protected:
        double m_randomWeightVariation = 0.0;

        std::string m_name;

        std::list<std::shared_ptr<Connection>> m_inputConnections;
        std::list<std::shared_ptr<Connection>> m_outputConnections;

        NeuralNetwork& m_network;
    };

}
