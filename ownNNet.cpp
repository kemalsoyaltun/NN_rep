#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <algorithm>

struct Connection
{
    double weight;
    double deltaweight;
    int from;
};

class Neuron
{
    public:
        Neuron(std::vector<unsigned> connectionIndexs, unsigned myIndex, unsigned layerIndex);
        Neuron(unsigned prevLayerSize, unsigned myIndex, unsigned layerIndex);

        void forward(const std::vector<Neuron> prevLayer);
        void forward(const std::vector<double> inputs);
        int getOutputVal(void) const {return outputVal;}
    private:
        static double transferFunction(double x) { return 1.0 / (1.0 + exp(-x)); }
        void setActivationFunction(double ftype = 0);
        // void setConnections(unsigned connectionIndexs);

        std::vector<Connection> connections; 
        std::vector<unsigned> index;
        double outputVal;
};

Neuron::Neuron(std::vector<unsigned> connectionIndexs, unsigned myIndex, unsigned layerIndex)
{
    for(unsigned c : connectionIndexs)
    {
        connections.push_back(Connection());
        connections.back().weight = rand() / double(RAND_MAX);
        connections.back().from = connectionIndexs[c];
    }
    index.push_back(myIndex);
    index.push_back(layerIndex);
    outputVal = 0.0;
}

Neuron::Neuron(unsigned prevLayerSize, unsigned myIndex, unsigned layerIndex)
{
    for(unsigned c = 0; c < prevLayerSize; c++)
    {
        connections.push_back(Connection());
        connections.back().weight = rand() / double(RAND_MAX);
        connections.back().from = c;
    }
    index.push_back(myIndex);
    index.push_back(layerIndex);
    outputVal = 0.0;
}

void Neuron::forward(const std::vector<Neuron> prevLayer)
{
    for (Connection i : connections)
    {
        outputVal += prevLayer[i.from].outputVal * i.weight;
    }
    outputVal = Neuron::transferFunction(outputVal);
}

void Neuron::forward(const std::vector<double> inputs)
{
    for (unsigned i = 0; i < inputs.size(); i++)
    {
        outputVal += inputs[i];
    }
    outputVal = Neuron::transferFunction(outputVal);
}

typedef std::vector<Neuron> Layer;

class Net
{
    public:
        Net(const std::vector<unsigned> &topology);
        Net(const std::vector<unsigned> &topology, const std::vector<unsigned> &connectionTopology);
        
        void learning(const std::vector<double> targetVals, const std::vector<double> inputs);
        void feedForward(const std::vector<double> &inputVals);
        std::vector<double> getResults() const {return myOutputs;}
    private:
        std::vector<Layer> myLayers;
        std::vector<double> myOutputs;
        std::vector<Connection> Myweights;
};

Net::Net(const std::vector<unsigned> &topology)
{
    unsigned numLayers = topology.size();
    myLayers.push_back(Layer());
    
    for(unsigned Neuron_n = 0; Neuron_n <= topology[0]; ++Neuron_n){
        myLayers.back().push_back(Neuron(0, Neuron_n, 0));
    }
    for(unsigned Layer_n = 1; Layer_n < numLayers; Layer_n++)
    {
        myLayers.push_back(Layer());
        for(unsigned Neuron_n = 0; Neuron_n <= topology[Layer_n]; ++Neuron_n){
            myLayers.back().push_back(Neuron(topology[Layer_n-2], Neuron_n, Layer_n));
        }
    }
}

Net::Net(const std::vector<unsigned> &topology, const std::vector<unsigned> &connectionTopology)
{
    unsigned numLayers = topology.size();
    myLayers.push_back(Layer());
    
    for(unsigned Neuron_n = 0; Neuron_n <= topology[0]; ++Neuron_n){
        myLayers.back().push_back(Neuron(0, Neuron_n, 0));
    }
    for(unsigned Layer_n = 1; Layer_n < numLayers; Layer_n++)
    {
        myLayers.push_back(Layer());
        for(unsigned Neuron_n = 0; Neuron_n <= topology[Layer_n]; ++Neuron_n){
            myLayers.back().push_back(Neuron(topology[Layer_n-2], Neuron_n, Layer_n));
        }
    }
}

void Net::feedForward(const std::vector<double> &inputVals)
{
    for (unsigned i = 0; i < inputVals.size(); i++)
    {
        myLayers[0][i].forward(inputVals);
    }

    for (unsigned Layen_n = 1; Layen_n < myLayers.size(); ++Layen_n)
    {
        for (unsigned Node_n = 0; Node_n < myLayers[Layen_n].size() -1; ++Node_n)
        {
            myLayers[Layen_n][Node_n].forward(myLayers[Layen_n-1]);
        }
    }
    for (unsigned i = 0; i < myLayers.back().size(); i++)
    {
        myOutputs.push_back(myLayers.back()[i].getOutputVal());
    }
}

int main()
{
    std::vector<unsigned> topology;
    std::vector<unsigned> connectionTopology;

    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(1);

    Net myNet(topology);

    std::vector<double> targetvals;
    std::vector<double> inputs;

    // targetvals = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    inputs = {1.0, 1.0, 1.0};

    myNet.feedForward(inputs);

    std::cout << "Output: " << myNet.getResults()[0] << std::endl;
    return 0;
}

// genel taslak oluşturuldu
// şuanki hata net veya nöron oluşturma sırasında bir for döngüsünde liste dışına taşmadan oluşuyor