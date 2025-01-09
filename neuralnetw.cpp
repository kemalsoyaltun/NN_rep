#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <algorithm>
#include "MNISTread.h"

struct Connection
{
    double weight;
    double deltaweight;
};

class Neuron;

typedef std::vector<Neuron> Layer;
////////////////////////// Neuron //////////////////////////
class Neuron
{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val;}
    double getOutputVal(void) const {return m_outputVal;}
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);
    std::vector<Connection> m_outputWeights;
private:
    static double eta; // [0.0..1.0] overall net training rate
    static double alpha; // [0.0..n] multiplier of last weight change (momentum)
    // static double transferFunction(double x) { return tanh(x); }
    // static double transferFunctionDerivative(double x) { return 1.0 - x * x; }
    static double transferFunction(double x) { return 1.0 / (1.0 + exp(-x)); }
    static double transferFunctionDerivative(double x) { return x * (1.0 - x); }
    // static double randomWeight(void) { return rand() / double(RAND_MAX); }
    static double randomWeight(void) { return rand() / double(RAND_MAX) * 2.0 - 1.0; }
    double sumDOW(const Layer &nextLayer) const;
    double m_outputVal;
    unsigned m_myIndex;
    double m_gradient;
};

double Neuron::eta = 0.15; // overall net learning rate, [0.0..1.0]
double Neuron::alpha = 0.5; // momentum, multiplier of last deltaWeight, [0.0..n]

void Neuron::updateInputWeights(Layer &prevLayer)
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer

    for (unsigned n = 0; n < prevLayer.size(); ++n)
    {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaweight;

        double newDeltaWeight =
            // Individual input, magnified by the gradient and train rate:
            eta
            * neuron.getOutputVal()
            * m_gradient
            // Also add momentum = a fraction of the previous delta weight
            + alpha
            * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaweight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed

    for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
    {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;

    // Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.

    for (unsigned n = 0; n < prevLayer.size() -1; ++n)
    {
        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for(unsigned c = 0; c < numOutputs; c++)
    {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = rand() / double(RAND_MAX);
        // std::cout << "Made a New Connection: "  << m_outputWeights.back().weight << std::endl;
    }
    m_myIndex = myIndex;
}
////////////////////////// Net //////////////////////////
class Net
{
public:
    Net(const std::vector<unsigned> &topology);
    void feedForward(const std::vector<double> &inputVals);
    void backProp(const std::vector<double> &targetVals);
    void getResults(std::vector<double> &resultVals) const;
    void learning(const std::vector<double> &targetVals,const std::vector<std::vector<double>> &inputs);
private:
    std::vector<Layer> m_Layers; // m_Layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;
};

void Net::learning(const std::vector<double> &targetVals, const std::vector<std::vector<double>> &inputs)
{
    std::vector<double> resultVals;
    for (unsigned i = 0; i < targetVals.size(); i++)
    {
        // for (int e = 0; e < 784; ++e) {
        //     std::cout << static_cast<double>(inputs[i][e]) << " ";
        //     if ((e + 1) % 27 == 0) std::cout << std::endl;
        // }
        // std::cout << "targetvals: " << targetVals[i][0] << ", " << targetVals[i][1] << ", " << targetVals[i][2] << ", " << targetVals[i][3] << ", " << targetVals[i][4] << ", " << targetVals[i][5] << ", " << targetVals[i][6] << ", " << targetVals[i][7] << ", " << targetVals[i][8] << ", " << targetVals[i][9] <<std::endl;
        std::cout << "targetvals: " << targetVals[i] << std::endl;
        feedForward(inputs[i]);
        backProp(targetVals);
        getResults(resultVals);
        // std::cout << "Output: " << resultVals[0] << ", " <<resultVals[1] <<", " <<resultVals[2] <<", " <<resultVals[3] <<", " <<resultVals[4] <<", " <<resultVals[5] <<", " <<resultVals[6] <<", " <<resultVals[7] <<", " <<resultVals[8] <<", " <<resultVals[9] <<std::endl;
        std::cout << "Output: " << resultVals[0] << std::endl;

        // auto maxElementTar = std::max_element(targetVals[i].begin(), targetVals[i].end());
        // auto maxElementRes = std::max_element(resultVals.begin(), resultVals.end());

        // int indexTar = std::distance(targetVals[i].begin(), maxElementTar);
        // int indexRes = std::distance(resultVals.begin(), maxElementRes);

        // if (indexRes == indexTar)
        // {
        //     std::cout << "True" << ", Confidence:" << *maxElementRes <<std::endl <<std::endl <<std::endl <<std::endl;
        // }
        // else
        // {
        //     std::cout << "False" << ", Confidence:" << *maxElementRes <<std::endl <<std::endl <<std::endl <<std::endl;
        // }
        // std::cout << i+1 << ". propogation" <<std::endl;
        // for (unsigned l = 0; l < m_Layers.size() - 1; l++)
        // {
        //     std::cout << "Layer" << l << " :"<<std::endl;
        //     for (unsigned j = 0; j < m_Layers[l].size() - 1; j++)
        //     {
        //         std::cout << "     Neuron" << j << " weights:"<<std::endl;

        //         for (unsigned k = 0; k < m_Layers[l][j].m_outputWeights.size(); k++)
        //         {
        //         std::cout << "          " <<m_Layers[l][j].m_outputWeights[k].weight << std::endl;
        //         }
        //     }
        //     std::cout << "---------------------"<<std::endl;
        // }
        // std::cout <<std::endl<<std::endl<<std::endl<<std::endl<<std::endl;
    }
}

void Net::getResults(std::vector<double> &resultVals) const
{
    resultVals.clear();

    for (unsigned n = 0; n < m_Layers.back().size() - 1; ++n)
    {
        resultVals.push_back(m_Layers.back()[n].getOutputVal());
    }
}

void Net::backProp(const std::vector<double> &targetVals)
{
    // Calculate overall net error (RMS of output neuron errors)
    Layer &outputLayer = m_Layers.back();
    m_error = 0.0;

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
    {
        double error = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += error * error;
    }
    m_error /= outputLayer.size() - 1; // get average error squared
    m_error = sqrt(m_error); // RMS

    // Implement a recent average measurement:

    m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);

    // Calculate output layer gradients

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
    {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // Calculate gradients on hidden layers

    for (unsigned layerNum = m_Layers.size() - 2; layerNum > 0; --layerNum)
    {
        Layer &hiddenLayer = m_Layers[layerNum];
        Layer &nextLayer = m_Layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n)
        {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // For all layers from outputs to first hidden layer,
    // update connection weights

    for (unsigned layerNum = m_Layers.size() - 1; layerNum > 0; --layerNum)
    {
        Layer &layer = m_Layers[layerNum];
        Layer &prevLayer = m_Layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n)
        {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Net::feedForward(const std::vector<double> &inputVals){
    assert(inputVals.size() == m_Layers[0].size() - 1);
    for (unsigned i = 0; i < inputVals.size(); i++){
        m_Layers[0][i].setOutputVal(inputVals[i]);
    }

    for (int Layen_n = 1; Layen_n < m_Layers.size(); ++Layen_n){
        Layer &prevLayer = m_Layers[Layen_n-1];
        for (int Node_n = 0; Node_n < m_Layers[Layen_n].size() -1; ++Node_n){
            m_Layers[Layen_n][Node_n].feedForward(prevLayer);
        }
    }
}

Net::Net(const std::vector<unsigned> &topology)
{
    unsigned numLayers = topology.size();
    for(unsigned Layer_n = 0; Layer_n < numLayers; Layer_n++)
    {
        m_Layers.push_back(Layer());
        unsigned numOutputs = Layer_n == topology.size()-1 ? 0 : topology[Layer_n + 1];
        for(unsigned Neuron_n = 0; Neuron_n <= topology[Layer_n]; ++Neuron_n){
            m_Layers.back().push_back(Neuron(numOutputs, Neuron_n));
            // std::cout << "Made a New Neuron" << std::endl;
        }
        
        // Force the bias node's output value to 1.0. It's the last neuron created above
        m_Layers.back().back().setOutputVal(1.0);
    }
}

int main()
{
    std::vector<unsigned> topology;

    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(1);

    Net myNet(topology);

    std::vector<double> targetvals;
    std::vector<std::vector<int>> inputs;

    targetvals = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    inputs = {{}}
    // int epoch = 1;
    // // e.g., { 3, 2, 1 }
    // std::vector<unsigned> topology;
    // topology.push_back(784);
    // topology.push_back(64);
    // topology.push_back(10);

    // Net myNet(topology);

    // // Görüntü dosyasını oku
    // std::string imageFilePath = "/home/yorgundemokrat/cpptr/Dataset/train-images.idx3-ubyte"; 
    // std::vector<unsigned char> imageData = readUByteFile(imageFilePath);

    // // Etiket dosyasını oku
    // std::string labelFilePath = "/home/yorgundemokrat/cpptr/Dataset/train-labels.idx1-ubyte";
    // std::vector<unsigned char> labelData = readUByteFile(labelFilePath);
    
    // int magicNumber = (imageData[0] << 24) | (imageData[1] << 16) | (imageData[2] << 8) | imageData[3];
    // int numberOfImages = (imageData[4] << 24) | (imageData[5] << 16) | (imageData[6] << 8) | imageData[7];
    // int numberOfRows = (imageData[8] << 24) | (imageData[9] << 16) | (imageData[10] << 8) | imageData[11];
    // int numberOfColumns = (imageData[12] << 24) | (imageData[13] << 16) | (imageData[14] << 8) | imageData[15];
    
    // int imageSize = numberOfRows * numberOfColumns;

    // for(int e = 0; e < epoch; e++)
    //     {
    //         for(int i = 0; i < 500; i++)
    //         {
    //             // Görüntü verisini double vektörüne dönüştür
    //             std::vector<double> inputVals(imageData.begin() + 16 + i * imageSize, 
    //                                         imageData.begin() + 16 + (i + 1) * imageSize);

    //             // inputVals'in değerlerini 0-1 aralığına çek
    //             for (double &val : inputVals) {
    //                 val = val / 255.0;
    //             }

    //             int label = labelData[8 + i];

    //             std::vector<double> targetVals(10, 0.0); // 10 düğümlü hedef vektörü, başlangıçta tüm elemanlar 0.0
                
    //             targetVals[label] = 1.0; // Hangi düğümün aktif olduğunu göster
    //             std::cout << "Label: " << label << std::endl;
    //             // Eğitimi gerçekleştirin
    //             myNet.learning({targetVals}, {inputVals});
    //         }
    //     }
        return 0;
}