#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <vector>
#include <algorithm>
#include <Utils.h>
#include <bits/stdc++.h>
using namespace std;

class Layer
{
public:
    std::vector<std::vector<double>> weights, biases, rawOutputs, activations, dC_dWs, dC_dBs;

    Layer(int inputCount, int outputCount);

    int getNumberOfNeurons();

    std::vector<std::vector<double>> getWeights();

    void setWeights(std::vector<std::vector<double>> updated_weights);

    void setBiases(std::vector<std::vector<double>> updated_biases);

    std::vector<std::vector<double>> getBiases();
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
    void softmax(std::vector<std::vector<double>> &matOrg);

    vector<Layer *> layers;
    NeuralNetwork(vector<int> neuronCount, double alpha, string fileName = "na");
    vector<vector<double>> feedForward(vector<vector<double>> input)
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
    vector<vector<double>> deriv_LeakyReLU(const vector<vector<double>> &mat)
    {
        vector<vector<double>> derivative(mat.size(), vector<double>(mat[0].size()));
        for (size_t i = 0; i < mat.size(); ++i)
        {
            for (size_t j = 0; j < mat[i].size(); ++j)
            {
                derivative[i][j] = (mat[i][j] > 0) ? 1.0f : 0.0f;
            }
        }
        return derivative;
    }
    void backward_prop(vector<vector<double>> X, vector<int> Y)
    {
        int m = Y.size();
        vector<vector<double>> one_hot_Y = Utils::one_hot(Y);
        vector<vector<double>> prev_dZs;
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
    vector<int> get_predictions(const vector<vector<double>> &P)
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
            double max_prob = P[0][j];
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

    double getCrossEntropyLoss(vector<int> &Y, vector<vector<double>> &predictions)
    {
        double loss = 0;
        int batch_size = Y.size();
        int index;
        for (int j = 0; j < batch_size; j++)
        {
            index = Y[j];
            loss -= (log(predictions[index][j] + 0.000001) / batch_size);
        }
        return loss;
    }

    // void saveLoss(double loss, string type, string filename){
    //     std::ios_base::openmode mode = std::ios::app;
    //     std::ofstream outFile(filename, mode);
    //     if (!outFile)
    //     {
    //         std::cerr << "Error opening file for writing Loss!" << std::endl;
    //         return;
    //     }
    //     if (type == "train")
    //     {
    //         outFile << "train_loss: " << loss << std::endl;
    //     }
    //     else if(type == "val")
    //     {
    //         outFile << "val_loss: " << loss << std::endl;
    //     }
    //     else if(type == "val_2")
    //     {
    //         outFile << "val_2_loss: " << loss << std::endl;
    //     }
    //     else
    //     {
    //         cout<<"Unknown Mode of Loss!"<<endl;
    //     }
    //     outFile.close();
    // }
    void saveLosses()
    {
        std::ios_base::openmode mode = std::ios::app;
        std::ofstream outFile("C:\\Users\\Darshan\\Desktop\\OOP project\\model\\saved\\alpha.txt", mode);
        if (!outFile)
        {
            std::cerr << "Error opening file for writing Loss!" << std::endl;
            return;
        }
        outFile << "Train loss: ";
        for (const auto &loss : train_loss)
        {
            outFile << loss << ' ';
        }
        outFile << std::endl;

        outFile << "Validation loss: ";
        for (const auto &loss : val_loss)
        {
            outFile << loss << ' ';
        }
        outFile << std::endl;
        outFile << "Validation 2 loss: ";
        for (const auto &loss : val_2_loss)
        {
            outFile << loss << ' ';
        }
        outFile << std::endl;
        outFile.close();
    }
    void saveAccuracies()
    {
        std::ios_base::openmode mode = std::ios::app;
        std::ofstream outFile("C:\\Users\\Darshan\\Desktop\\OOP project\\model\\metrics\\loss.txt", mode);
        if (!outFile)
        {
            std::cerr << "Error opening file for writing Loss!" << std::endl;
            return;
        }
        outFile << "Train accuracy: ";
        for (const auto &accuracy : train_accuracy)
        {
            outFile << accuracy << ' ';
        }
        outFile << std::endl;

        outFile << "Validation accuracy: ";
        for (const auto &accuracy : val_accuracy)
        {
            outFile << accuracy << ' ';
        }
        outFile << std::endl;
        outFile << "Validation 2 accuracy: ";
        for (const auto &accuracy : val_2_accuracy)
        {
            outFile << accuracy << ' ';
        }
        outFile << std::endl;
        outFile.close();
    }
    void gradient_descent(vector<vector<double>> &X, vector<int> &Y, int iterations, bool verbose = false)
    {
        int ll = layers.size() - 1;
        double loss = 0;
        int batch_size = X[0].size();
        double accuracy;
        for (int i = 0; i < iterations; i++)
        {
            vector<vector<double>> predictions = feedForward(X);
            backward_prop(X, Y);
            update_params();
            double loss_i = getCrossEntropyLoss(Y, predictions);
            loss += loss_i;
            if (verbose)
            {
                cout << "Training data Loss: " << loss_i << endl; // Training Loss!
                if ((i + 1) % 1 == 0)
                {
                    // cout << "Iteration: " << i + 1 << endl;
                    accuracy = Utils::getAccuracy(get_predictions(layers[ll]->activations), Y);
                    cout << "Training data Accuracy: " << accuracy << endl;
                }
            }
        }
        train_loss.push_back(loss);
        train_accuracy.push_back(accuracy);
        // saveLoss(loss,"train","metrics/loss.txt");
        // saveAccuracy(accuracy,"train","metrics/accuracy.txt");
    }

    void saveLayer(vector<vector<double>> &weights, vector<vector<double>> &biases, std::string &filename, bool reset = false)
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
    bool loadLayer(ifstream &inFile, vector<vector<double>> &W, vector<vector<double>> &B)
    {
        string line;
        int row, col;
        char type;
        double a;
        int i, j;
        while (getline(inFile, line))
        { // Read line by line until the end of the file
            istringstream iss(line);
            if (line[0] == 'W' | line[0] == 'B')
            {
                iss >> type >> row >> col; // Read the first character and integer from the line
                if (type == 'W')
                    W.resize(row, std::vector<double>(col));
                if (type == 'B')
                    B.resize(row, std::vector<double>(col));
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
        return true;
    }
    void saveNetwork(string fileName)
    {
        this->saveLayer(this->layers[0]->weights, this->layers[0]->biases, fileName, true);
        for (int i = 1; i < this->layers.size(); i++)
        {
            this->saveLayer(this->layers[i]->weights, this->layers[i]->biases, fileName);
        }
        saveLosses();
        saveAccuracies();
    }
    void train(vector<vector<double>> X_train, vector<int> Y_train, vector<vector<double>> X_val, vector<int> Y_val, vector<vector<double>> X_val_2, vector<int> Y_val_2, int iterations, bool verbose = false, string fileName = "model.txt", bool save = false)
    {
        gradient_descent(X_train, Y_train, iterations, verbose);
        // Get validation loss here! (for one batch)
        vector<vector<double>> predictions = this->feedForward(X_val);
        double loss = this->getCrossEntropyLoss(Y_val, predictions);
        // saveLoss(loss,"val","metrics/loss.txt");
        double accuracy = Utils::getAccuracy(get_predictions(layers[layers.size() - 1]->activations), Y_val);
        val_loss.push_back(loss);
        val_accuracy.push_back(accuracy);
        if (verbose)
        {
            cout << "Validation data loss: " << loss << endl;
            cout << "Validation data accuracy: " << accuracy << endl;
        }
        // saveAccuracy(accuracy, "val", "metrics/accuracy.txt");

        predictions = this->feedForward(X_val_2);
        loss = this->getCrossEntropyLoss(Y_val_2, predictions);
        // saveLoss(loss,"val_2","metrics/loss.txt");
        val_2_loss.push_back(loss);
        accuracy = Utils::getAccuracy(get_predictions(layers[layers.size() - 1]->activations), Y_val_2);
        val_2_accuracy.push_back(accuracy);
        if (verbose)
        {
            cout << "Validation data 2 loss: " << loss << endl;
            cout << "Validation data 2 accuracy: " << accuracy << endl
                 << endl;
        }
        // saveAccuracy(accuracy, "val_2", "metrics/accuracy.txt");

        if (save)
        {
            saveNetwork(fileName);
        }
    }
    pair<int, double> predict(vector<vector<double>> X)
    {
        vector<vector<double>> predictions = feedForward(X);
        int index = 0;
        double confidence = predictions[index][0];
        for (int i = 1; i < predictions.size(); i++)
        {
            if (predictions[i][0] > predictions[index][0])
            {
                index = i;
                confidence = predictions[i][0];
            }
        }
        return {index, confidence};
    }
};

#endif // MYCLASSES_H