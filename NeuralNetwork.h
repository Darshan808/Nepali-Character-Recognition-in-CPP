#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <bits/stdc++.h>
#include "Utils.h"

using namespace std;

class Layer
{
public:
    vector<vector<float>> weights, biases, rawOutputs, activations, dC_dWs, dC_dBs;
    Layer(int inputCount, int outputCount)
    {
        for (int i = 0; i < outputCount; i++)
        {
            weights.push_back(Utils::generateRandomVector(inputCount, -0.5, 0.5));
            biases.push_back(Utils::generateRandomVector(1, -1, 1));
        }
    }
    int getNumberOfNeurons()
    {
        return biases.size();
    }
    vector<vector<float>> getWeights()
    {
        return weights;
    }
    void setWeights(vector<vector<float>> updated_weights)
    {
        weights = updated_weights;
    }
    void setBiases(vector<vector<float>> updated_biases)
    {
        biases = updated_biases;
    }
    vector<vector<float>> getBiases()
    {
        return biases;
    }
};

class NeuralNetwork
{
private:
    // Hyperparameters
    float alpha = 0.5;
    // BackProp Variables
    vector<float> prev_dC_das;
    vector<float> y_hats;
    vector<vector<float>> activations;
    void leakyRelu(vector<vector<float>> &mat)
    {
        for (auto &row : mat)
        {
            for (auto &val : row)
            {
                val = max(0.1f * val, val);
            }
        }
    }

public:
    void softmax(vector<vector<float>> &matOrg)
    {
        vector<vector<float>> mat = Utils::transposeMatrix(matOrg);
        for (auto &row : mat)
        {
            vector<double> exp_values(row.size());
            double max_el = *max_element(row.begin(), row.end());
            for (size_t i = 0; i < row.size(); ++i)
            {
                exp_values[i] = exp(row[i] - max_el);
            }
            double sum_exp_values = accumulate(exp_values.begin(), exp_values.end(), 0.0);
            for (size_t i = 0; i < row.size(); ++i)
            {
                row[i] = exp_values[i] / sum_exp_values;
            }
        }
        matOrg = Utils::transposeMatrix(mat);
    }
    vector<Layer *> layers;
    NeuralNetwork(vector<int> neuronCount, string fileName="na")
    {
        for (int i = 0; i < neuronCount.size() - 1; i++)
        {
            layers.push_back(new Layer(neuronCount[i], neuronCount[i + 1]));
        }
        
        if(fileName != "na"){
            ifstream inFile(fileName);
            for (int i = 0; i < this->layers.size(); i++)
            {
                this->loadLayer(inFile,layers[i]->weights,layers[i]->biases);
            }
            inFile.close();
        }
    }
    vector<vector<float>> feedForward(vector<vector<float>> input)
    {
        for (int i = 0; i < layers.size(); i++)
        {
            input = Utils::matMul(layers[i]->getWeights(), input);
            input = Utils::matAdd(input, layers[i]->getBiases());
            layers[i]->rawOutputs = input;
            if (i == layers.size() - 1)
            {
                // cout<<"Before sofmax: "<<input[0][0]<<", "<<input[1][0]<<endl;
                softmax(input);
                // cout<<"After sofmax: "<<input[0][0]<<", "<<input[1][0]<<endl;
            }
            else
                leakyRelu(input);
            layers[i]->activations = input;
        }
        return layers[layers.size() - 1]->activations;
    }
    vector<vector<float>> deriv_LeakyReLU(const vector<vector<float>> &mat)
    {
        vector<vector<float>> derivative(mat.size(), vector<float>(mat[0].size()));
        for (size_t i = 0; i < mat.size(); ++i)
        {
            for (size_t j = 0; j < mat[i].size(); ++j)
            {
                derivative[i][j] = (mat[i][j] > 0) ? 1.0f : 0.0f;
            }
        }
        return derivative;
    }
    void backward_prop(vector<vector<float>> X, vector<int> Y)
    {
        int m = Y.size();
        vector<vector<float>> one_hot_Y = Utils::one_hot(Y);
        vector<vector<float>> prev_dZs;
        size_t numLayers = layers.size();
        // For output layer
        //  cout << "Activations last layer: " << layers[numLayers - 1]->activations[0][0] << ", " << layers[numLayers - 1]->activations[1][0] << endl;
        prev_dZs = Utils::matrixSubtraction(layers[numLayers - 1]->activations, one_hot_Y);
        // cout << "prev_dZs last layer: " << prev_dZs[0][0]<<", "<<prev_dZs[0][1]<<endl;
        layers[numLayers - 1]->dC_dWs = Utils::multiplyScalarToMatrix((1.0f / m), (Utils::matMul(prev_dZs, Utils::transposeMatrix(layers[numLayers - 2]->activations))));
        layers[numLayers - 1]->dC_dBs = Utils::multiplyScalarToMatrix((1.0f / m), Utils::sumRows(prev_dZs));
        for (int i = numLayers - 2; i >= 0; i--)
        {
            prev_dZs = Utils::multiplyCorrespondingElements(Utils::matMul(Utils::transposeMatrix(layers[i + 1]->getWeights()), prev_dZs), deriv_LeakyReLU(layers[i]->rawOutputs));
            layers[i]->dC_dWs = Utils::multiplyScalarToMatrix(1.0f / m, Utils::matMul(prev_dZs, Utils::transposeMatrix(i == 0 ? X : layers[i - 1]->activations)));
            layers[i]->dC_dBs = Utils::multiplyScalarToMatrix(1.0f / m, Utils::sumRows(prev_dZs));
            // cout << "prev_dZs " << prev_dZs[0][0] << ", " << prev_dZs[0][1] << endl;
        }
    }
    void update_params()
    {
        for (int i = 0; i < layers.size(); i++)
        {
            // cout<<"Layer "<<i<<" dW[0][0]: "<<layers[i]->dC_dWs[0][0]<<", dW[0][1] "<<layers[i]->dC_dWs[0][1]<<endl;
            // cout<<"Before Update: Layer "<<i<<" W[0][0]: "<<layers[i]->getWeights()[0][0]<<", dW[0][1] "<<layers[i]->getWeights()[0][1]<<endl;
            layers[i]->setWeights(Utils::matAdd(layers[i]->getWeights(), Utils::multiplyScalarToMatrix(-1 * alpha, layers[i]->dC_dWs)));
            // cout<<"After update: Layer "<<i<<" W[0][0]: "<<layers[i]->getWeights()[0][0]<<", dW[0][1] "<<layers[i]->getWeights()[0][1]<<endl;
            // cout<<"Before update: Layer "<<i<<" B[0]: "<<layers[i]->getBiases()[0][0]<<", B[1] "<<layers[i]->getBiases()[1][0]<<endl;
            layers[i]->setBiases(Utils::matAdd(layers[i]->getBiases(), Utils::multiplyScalarToMatrix(-1 * alpha, layers[i]->dC_dBs)));
            // cout<<"After update: Layer "<<i<<" B[0]: "<<layers[i]->getBiases()[0][0]<<", B[1] "<<layers[i]->getBiases()[1][0]<<endl;
        }
    }
    vector<int> get_predictions(const vector<vector<float>> &P)
    {
        if (P.empty() || P[0].empty())
        {
            throw runtime_error("The probability matrix is empty.");
        }

        // Get the number of classes and the number of training samples
        size_t num_classes = P.size();
        size_t num_samples = P[0].size();

        // Initialize a vector to store the predictions
        vector<int> predictions(num_samples, 0);

        // Iterate over each training sample (each column)
        for (size_t j = 0; j < num_samples; ++j)
        {
            float max_prob = P[0][j];
            int max_index = 0;

            // Find the class with the highest probability for the current sample
            for (size_t i = 1; i < num_classes; ++i)
            {
                if (P[i][j] > max_prob)
                {
                    max_prob = P[i][j];
                    max_index = i;
                }
            }

            // Store the index of the class with the highest probability
            predictions[j] = max_index;
        }

        return predictions;
    }

    double getCrossEntropyLoss(vector<int> &Y, vector<vector<float>> &predictions)
    {
        double loss = 0;
        int batch_size = Y.size();
        int index;
        for (int j = 0; j < batch_size; j++)
        {
            index = Y[j];
            loss -= (log(predictions[index][j]) / batch_size);
        }
        return loss;
    }

    void saveLoss(double loss, string type, string filename){
        std::ios_base::openmode mode = std::ios::app;
        std::ofstream outFile(filename, mode);
        if (!outFile)
        {
            std::cerr << "Error opening file for writing Loss!" << std::endl;
            return;
        }
        if (type == "train")
        {
            outFile << "train_loss: " << loss << std::endl;
        }
        else if(type == "val")
        {
            outFile << "val_loss: " << loss << std::endl;
        }
        else if(type == "val_2")
        {
            outFile << "val_2_loss: " << loss << std::endl;
        }
        else
        {
            cout<<"Unknown Mode of Loss!"<<endl;
        }
        outFile.close();
    }

    void saveAccuracy(float accuracy, string type, string filename){
        std::ios_base::openmode mode = std::ios::app;
        std::ofstream outFile(filename, mode);
        if (!outFile)
        {
            std::cerr << "Error opening file for writing Loss!" << std::endl;
            return;
        }
        if (type == "train")
        {
            outFile << "train_accuracy: " << accuracy << std::endl;
        }
        else if(type == "val")
        {
            outFile << "val_accuracy: " << accuracy << std::endl;
        }
        else if(type == "val_2")
        {
            outFile << "val_2_accuracy: " << accuracy << std::endl;
        }
        else
        {
            cout<<"Unknown Mode of Accuracy!"<<endl;
        }
        outFile.close();
    }

    void gradient_descent(vector<vector<float>> &X, vector<int> &Y, int iterations)
    {
        int ll = layers.size() - 1;
        double loss = 0;
        int batch_size = X[0].size();
        for (int i = 0; i < iterations; i++)
        {
            vector<vector<float>> predictions = feedForward(X);
            double loss_i = getCrossEntropyLoss(Y, predictions);
            loss += loss_i;
            cout<<"Training data Loss: "<<loss_i<<endl; //Training Loss!
            backward_prop(X, Y);
            update_params();
            if ((i + 1) % 1 == 0)
            {
                cout << "Iteration: " << i + 1 << endl;
                float accuracy = Utils::getAccuracy(get_predictions(layers[ll]->activations), Y);
                cout << "Accuracy: " << accuracy << endl;
            }
        }
        saveLoss(loss,"train","metrics/loss.txt");
        float accuracy = Utils::getAccuracy(get_predictions(layers[ll]->activations), Y);
        saveAccuracy(accuracy,"train","metrics/accuracy.txt");
    }

    void saveLayer(vector<vector<float>> &weights, vector<vector<float>> &biases, std::string &filename, bool reset = false)
    {
        std::ios_base::openmode mode = reset ? std::ios::out : std::ios::app;
        std::ofstream outFile(filename, mode);

        if (!outFile)
        {
            std::cerr << "Error opening file for writing!" << std::endl;
            return;
        }

        // Write weights dimensions and values
        if (!weights.empty())
        {
            outFile << 'W' << ' ' << weights.size() << ' ' << weights[0].size() << std::endl;
            for (const auto &row : weights)
            {
                for (const auto &weight : row)
                {
                    outFile << weight << ' ';
                }
                outFile << std::endl;
            }
        }

        // Write biases dimensions and values
        if (!biases.empty())
        {
            outFile << 'B' << ' ' << biases.size() << ' ' << 1 << std::endl;
            for (const auto &bias : biases)
            {
                outFile << bias[0] << ' ';
            }
            outFile << std::endl;
        }

        outFile.close();
    }
    bool loadLayer(ifstream &inFile, vector<vector<float>> &W, vector<vector<float>> &B)
    {
        string line;
        int row, col;
        char type;
        float a;
        int i, j;
        while (getline(inFile, line))
        { // Read line by line until the end of the file
            istringstream iss(line);
            if (line[0] == 'W' | line[0] == 'B')
            {
                iss >> type >> row >> col; // Read the first character and integer from the line
                if (type == 'W')
                    W.resize(row, std::vector<float>(col));
                if (type == 'B')
                    B.resize(row, std::vector<float>(col));
                i = 0, j = 0;
                continue;
            }
            if (type == 'W')
            {
                while (iss >> a)
                { // Read remaining integers from the line
                    W[i][j] = a;
                    j++;
                }
                i++;
                j = 0;
            }
            else if (type == 'B')
            {
                while (iss >> a)
                { // Read remaining integers from the line
                    B[j][0] = a;
                    j++;
                }
                return true;
            }
            else
            {
                throw runtime_error("Invalid type or type is not set");
                return false;
            }
        }
    }
    void train(vector<vector<float>> X_train, vector<int> Y_train, vector<vector<float>> X_val, vector<int> Y_val, vector<vector<float>> X_val_2, vector<int> Y_val_2, int iterations, string fileName = "model.txt", bool save=false)
    {
        gradient_descent(X_train, Y_train, iterations);
        //Get validation loss here! (for one batch)
        vector<vector<float>> predictions = this->feedForward(X_val);
        double loss = this->getCrossEntropyLoss(Y_val,predictions);
        cout<<"Validation data loss: "<<loss<<endl;
        saveLoss(loss,"val","metrics/loss.txt");
        float accuracy = Utils::getAccuracy(get_predictions(layers[layers.size()-1]->activations), Y_val);
        saveAccuracy(accuracy, "val", "metrics/accuracy.txt");

        predictions = this->feedForward(X_val_2);
        loss = this->getCrossEntropyLoss(Y_val_2,predictions);
        cout << "Validation data 2 loss: " << loss << endl << endl;
        saveLoss(loss,"val_2","metrics/loss.txt");
        accuracy = Utils::getAccuracy(get_predictions(layers[layers.size()-1]->activations), Y_val_2);
        saveAccuracy(accuracy, "val_2", "metrics/accuracy.txt");

        if(save){
            this->saveLayer(this->layers[0]->weights,this->layers[0]->biases,fileName,true);
            for(int i=1;i<this->layers.size();i++){
                this->saveLayer(this->layers[i]->weights,this->layers[i]->biases,fileName);
            }
        }
    }
    pair<int,float> predict(vector<vector<float>> X){
        vector<vector<float>> predictions = feedForward(X);
        int index = 0;
        float confidence = predictions[index][0];
        for(int i=1;i<predictions.size();i++){
            if(predictions[i][0]>predictions[index][0]){
                index = i;
                confidence = predictions[i][0];
            }
        }
        return {index,confidence};
    }
};

#endif // MYCLASSES_H
