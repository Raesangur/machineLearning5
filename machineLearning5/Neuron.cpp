#include "globaldef.h"
#include "Neuron.h"

namespace nn
{
    Neuron::Neuron(NeuralNetwork& network,
        size_t row, size_t column,
        double bias) :
        m_row(row),
        m_column(column),
        m_bias(bias),
        m_network(network)
    {
        m_name = "Neuron [" + std::to_string(column) + "][" + std::to_string(row) + "]";

        getInputConnections();
    }

    Neuron::Neuron(NeuralNetwork& network,
        size_t row, size_t column,
        const std::string& name,
        double bias) :
        m_row(row),
        m_column(column),
        m_bias(bias),
        m_name(name),
        m_network(network)
    {
        getInputConnections();
    }

    Neuron::Neuron(NeuralNetwork& network,
                   size_t row, size_t column,
                   double bias,
                   std::vector<double>& previousWeights, double weightVariation) :
        m_row(row),
        m_column(column),
        m_bias(bias),
        m_network(network)
    {
        m_name = "Neuron [" + std::to_string(column) + "][" + std::to_string(row) + "]";

        getInputConnections(weightVariation, previousWeights);
    }

    double Neuron::calculate()
    {
        double input;

        /* Get given input if on the first layer */
        if (m_column == 0)
        {
            input = m_input;
        }
        /* Otherwise, gather input values from connections with preceding neurons */
        else
        {
            input = getInputValues();
        }

        m_output = activationFunction(input);

#ifdef DEBUG
        std::cout << m_name << " outputs: " << m_output << std::endl;
#endif

        return m_output;
    }

    void Neuron::getInputConnections(void)
    {
        if (m_column == 0)
        {
            /* first column, there are no inputs */
            return;
        }

        /* For each neuron in the previous column */
        for (size_t i = 0; i < m_network[m_column - 1].size(); i++)
        {
            Neuron& neuron = m_network[m_column - 1][i];

            /* Create connection */
            std::shared_ptr<Connection> connectionPtr = 
                std::make_shared<Connection>(Connection(neuron, *this));

            /* Add connection to respective vectors */
            neuron.m_outputConnections.push_back(connectionPtr);
            this->m_inputConnections.push_back(connectionPtr);
        }
    }

    void Neuron::getInputConnections(double weightVariation,
                                     std::vector<double>& previousWeights)
    {
        if (m_column == 0)
        {
            /* first column, there are no inputs */
            return;
        }

        /* For each neuron in the previous column */
        for (size_t i = 0; i < m_network[m_column - 1].size(); i++)
        {
            Neuron& neuron = m_network[m_column - 1][i];

            /* Generate a random new weight */
            double weight = getRandomNumber() *
                            weightVariation   *
                            previousWeights[i];

            /* Create connection */
            std::shared_ptr<Connection> connectionPtr = 
               std::make_shared<Connection>(Connection(neuron, *this, weight));

            /* Add connection to respecive vectors */
            neuron.m_outputConnections.push_back(connectionPtr);
            this->m_inputConnections.push_back(connectionPtr);
        }
    }

    double Neuron::getInputValues(void)
    {
        double total = 0.0;

        /* For each input connection */
        for (std::shared_ptr<Connection> connectionPtr : m_inputConnections)
        {
            /* Get input neuron */
            Neuron& inputNeuron = connectionPtr->getInputNeuron();

            /* Neuron's output * connection's weight */
            double weightedValue = inputNeuron.getOutput() *
                                   connectionPtr->getWeight();

            total += weightedValue;
        }

        /* Add this neuron's bias */
        total *= m_bias;

        return total;
    }

    double Neuron::activationFunction(double value)
    {
        /* Modified Simoid function:
         * 2 × ╭ ____1____  _  _1_  ╮
         *     ╰  1 + e^x       2   ╯
         */
        double eX = exp(value);	 /* e^x */

        double output = 1 / (1 + eX);
        output -= 0.5;
        output *= 2;

        return output;
    }


    //static double NormalSigmoid(double value)
    //{
    //    /* Sigmoid function: */
    //    /* _____1_____       */
    //    /*   1 + e^x	     */
    //    double eX = exp(value);	 /* e^x */

    //    double output = 1 / (1 + eX);

    //    return output;
    //}
}