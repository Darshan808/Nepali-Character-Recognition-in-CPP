#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <vector>
#include <algorithm>
#include <Utils.h>
#include <bits/stdc++.h>
using namespace std;

class LayerParent
{
public:
    std::vector<std::vector<double>> weights, biases, rawOutputs, activations, dC_dWs, dC_dBs;
    // Layer(int inputCount, int outputCount);
    int getNumberOfNeurons();
    std::vector<std::vector<double>> getWeights();
    void setWeights(std::vector<std::vector<double>> updated_weights);
    void setBiases(std::vector<std::vector<double>> updated_biases);
    std::vector<std::vector<double>> getBiases();
};

class Layer : public LayerParent
{
    public:
        Layer(int inputCount, int outputCount);
};

class DropOut: public LayerParent{
    private:
    double dropOutRate;
    DropOut(double dropOutRate);
    void dropOut();
};

class BatchNormalization: public LayerParent{
    void normalize();
};

class NeuralNetwork
{
private:
    // Hyperparameters
    double alpha;
    // BackProp Variables
    std::vector<double> prev_dC_das;
    std::vector<double> y_hats;
    std::vector<std::vector<double>> activations;
    void leakyRelu(std::vector<std::vector<double>> &mat);
public:
    void softmax(vector<vector<double>> &matOrg);
    vector<Layer *> layers;
    NeuralNetwork(vector<int> neuronCount, double alpha, string fileName = "na");
    vector<vector<double>> feedForward(vector<vector<double>> input);
    vector<vector<double>> deriv_LeakyReLU(const vector<vector<double>> &mat);
    void backward_prop(vector<vector<double>> X, vector<int> Y);
    void update_params();
    vector<int> get_predictions(const vector<vector<double>> &P);
    double getCrossEntropyLoss(vector<int> &Y, vector<vector<double>> &predictions);
    void saveLosses();
    void saveAccuracies();
    void gradient_descent(vector<vector<double>> &X, vector<int> &Y, int iterations, bool verbose = false);
    void saveLayer(vector<vector<double>> &weights, vector<vector<double>> &biases, std::string &filename, bool reset = false);
    bool loadLayer(ifstream &inFile, vector<vector<double>> &W, vector<vector<double>> &B);
    void saveNetwork(string fileName);
    void train(vector<vector<double>> X_train, vector<int> Y_train, vector<vector<double>> X_val, vector<int> Y_val, vector<vector<double>> X_val_2, vector<int> Y_val_2, int iterations, bool verbose = false, string fileName = "model.txt", bool save = false);
    pair<int, double> predict(vector<vector<double>> X);
};

#endif