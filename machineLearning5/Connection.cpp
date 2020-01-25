#include "globaldef.h"
#include "Connection.h"

namespace nn
{
    Connection::Connection(Neuron& inputNeuron, Neuron& outputNeuron, double weight) :
        m_weight(weight),
        m_inputNeuron(inputNeuron), m_outputNeuron(outputNeuron)
    {
        m_name = m_inputNeuron.getName() + " -> " + m_outputNeuron.getName();
    }
}