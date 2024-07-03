#include<bits/stdc++.h>
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
        "o", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
    // std::vector<std::vector<float>> data;
    // Danfe::loadData(data,"nepali_mini_dataset",true);

    // // Splitting data
    // vector<vector<float>> X_train;
    // vector<int> Y_train;
    // Danfe::xySplit(data,X_train,Y_train,true);

    //Training Network
    NeuralNetwork *NN = new NeuralNetwork({1024, 10, 10 46}, 0.5, "model/10_10_0.5.txt");
    // NN->train(X_train, Y_train,30,"my_model.txt");
    //Inside train, use gradient descent and save weights + biases
    //Learn about batch,epochs,type of gradient descent
    // NN->train(X_train, Y_train, 100,"my_model.txt");
    // vector<vector<float>> predict_X;
    // Danfe::extractColumn(X_train,480,predict_X,true);
    // Danfe::extractPixels("sample_5.txt",predict_X);
    vector<vector<float>> predict_X;
    Danfe::getLatestImgPixels(predict_X);
    Utils::normalizeData(predict_X);
    pair<int,float> p = NN->predict(predict_X);
    cout<<"Predicted ";
    // // std::locale::global(std::locale(""));
    // // std::wcout.imbue(std::locale());
    cout<<romanizedLabels[p.first];
    cout<<" with confidence "<<p.second<<endl;

    //To do
    //My labels starts with 1 but 1 is ka
    //Make it start from 0, change architecture of neural network
    //train again
    //Find a way to automate using batch script
    //Try using python to convert clicked images to 32*32 //Done!
}