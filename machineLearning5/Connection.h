#pragma once
#include "globaldef.h"

#include "Neuron.h"

namespace nn
{
    /* Importing class */
    class Neuron;

    class Connection
    {
#pragma region Methods
        /*****************************************************************************/
        /* Public methods ---------------------------------------------------------- */
    public:
        Connection(Neuron& inputNeuron, Neuron& outputNeuron, double weight = 1.0);
        ~Connection() = default;

        Neuron& getInputNeuron() { return m_inputNeuron; }
        Neuron& getOutputNeuron() { return m_outputNeuron; }

        double getWeight() { return m_weight; }
        std::string& getName() { return m_name; }

        void setWeight(double weight) { m_weight = weight; }

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
        double m_weight;

        Neuron& m_inputNeuron;
        Neuron& m_outputNeuron;

        std::string m_name;
#pragma endregion
    };

}
