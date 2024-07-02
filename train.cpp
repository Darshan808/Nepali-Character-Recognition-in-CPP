#include <bits/stdc++.h>
#include "NeuralNetwork.h"
#include "Utils.h"
#include "Danfe.h"
using namespace std;

int main()
{
    // vector<const wchar_t *> labels = {
    //     L"क", L"ख", L"ग", L"घ", L"ङ", L"च", L"छ", L"ज", L"झ", L"ञ",
    //     L"ट", L"ठ", L"ड", L"ढ", L"ण", L"त", L"थ", L"द", L"ध", L"न",
    //     L"प", L"फ", L"ब", L"भ", L"म", L"य", L"र", L"ल", L"व", L"श",
    //     L"ष", L"स", L"ह", L"क्ष", L"त्र", L"ज्ञ",
    //     L"O", L"१", L"२", L"३", L"४", L"५", L"६", L"७", L"८", L"९"};
    const std::vector<std::string> romanizedLabels = {
        "ka", "kha", "ga", "gha", "nga", "cha", "chha", "ja", "jha", "anda",
        "ta", "tha", "da", "dha", "3na", "ta2", "tha2", "da2", "dha2", "4na",
        "pa", "pha", "ba", "bha", "ma", "ya", "ra", "la", "va", "sa",
        "2sa", "3sa", "ha", "kshya", "tra", "gya",
        "o", "1", "2", "3", "4", "5", "6", "7", "8", "9"
    };

    std::vector<std::vector<float>> train_data;
    std::vector<std::vector<float>> val_data;
    std::vector<std::vector<float>> val_2_data;
    Danfe::loadData(val_data, "dataset/cv_data/cv.txt");
    Danfe::loadData(val_2_data, "dataset/cv_data/cv_2.txt");

    vector<vector<float>> X_train;
    vector<int> Y_train;
    vector<vector<float>> X_val;
    vector<int> Y_val;
    vector<vector<float>> X_val_2;
    vector<int> Y_val_2;
    Danfe::xySplit(val_data, X_val, Y_val);
    Danfe::xySplit(val_2_data, X_val_2, Y_val_2);

    // Training Network
    NeuralNetwork *NN = new NeuralNetwork({1024, 512, 256, 46});
    int NUM_BATCHES = 1;
    bool save;
    for(int i=0;i<NUM_BATCHES;i++){
        cout<<"Running Batch "<<i+1<<endl;
        Danfe::loadData(train_data,"dataset/train_batches/train_batch_"+to_string(i)+".txt");
        // Splitting data
        Danfe::xySplit(train_data,X_train,Y_train);
        // NeuralNetwork *NN = new NeuralNetwork({1024, 10, 47}, "my_model.txt");
        save = i == NUM_BATCHES-1;
        NN->train(X_train, Y_train, X_val, Y_val, X_val, Y_val ,1,"model/just.txt",save);
        X_train.clear();
        Y_train.clear();
        train_data.clear();
    }
}