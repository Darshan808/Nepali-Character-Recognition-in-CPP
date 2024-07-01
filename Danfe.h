#ifndef MYCLASSES_H
#define MYCLASSES_H

#include <bits/stdc++.h>
#include "Utils.h"

class Danfe{
    public:
        static void loadData(std::vector<std::vector<float>> &data, string fileName, bool verbose = false) {
            std::ifstream file(fileName); // Open the CSV file

            if (!file.is_open())
            {
                std::cerr << "Failed to open the file." << std::endl;
            }
            std::string line;
            // Read the header line if there's one and ignore it
            // if (std::getline(file, line))
            // {
            //     // Do nothing with the header or process it if needed
            // }
            while (std::getline(file, line))
            { // Read each line
                std::vector<float> row;
                std::stringstream ss(line);
                std::string cell;

                while (std::getline(ss, cell, ' '))
                {                                   // Split by comma
                    row.push_back(std::stof(cell)); // Convert to int and add to row
                }

                data.push_back(row); // Add row to the data vector
            }
            if(verbose){
                std::cout << "Size of training data: " << data.size() << ',' << data[0].size() << std::endl;
            }
            file.close();
        }
        static void xySplit(vector<vector<float>> &data, vector<vector<float>> &X_train, vector<int> &Y_train, bool verbose = false)
        {
            for (const auto &sample : data)
            {
                vector<float> pixels(sample.begin() + 1, sample.end());
                X_train.push_back(pixels);
                Y_train.push_back(static_cast<int>(sample[0]));
            }
            X_train = Utils::transposeMatrix(X_train);
            Utils::normalizeData(X_train);
            if(verbose){
                std::cout << "Size of X_train: " << X_train.size() << ',' << X_train[0].size() << std::endl;
                std::cout << "Size of Y_train: " << Y_train.size() << endl;
            }
        }
        static void extractColumn(vector<vector<float>> &data, int index, vector<vector<float>> &single,bool save=false){
            int rows = data.size();
            int cols = data[0].size();
            if(index>=cols){
                throw runtime_error("Index out of bound, cannot extract column!");
            }
            single.resize(rows,vector<float>(1));
            for(int i=0;i<rows;i++){
                single[i][0] = data[i][index];
            }
            if(save){
                std::ofstream outFile("prediction.txt", std::ios::out);
                if (!outFile)
                {
                    std::cerr << "Error opening file for writing!" << std::endl;
                    return;
                }
                for(int i=0;i<rows;i++){
                    outFile<<static_cast<int>(single[i][0]*255)<<" ";
                }
                outFile.close();
            }
        }
        static void extractPixels(string fname, vector<vector<float>> &data){
            ifstream inFile("data/"+fname);
            string line;
            getline(inFile, line);
            istringstream iss(line);
            int pv;
            data.resize(32*32,vector<float>(1));
            for(int i=0;i<32*32;i++){
                iss >> data[i][0];
            }
            inFile.close();
        }
        static void getLatestImgPixels(vector<vector<float>> &data){
            int index = 1;
            std::ifstream checkFile;
            while (true)
            {
                std::string filename = "data/sample_" + std::to_string(index++) + ".txt";
                checkFile.open(filename);

                if (!checkFile) break;
                checkFile.close();
            }
            extractPixels("sample_"+to_string(index-2)+".txt",data);
        }
};

#endif // MYCLASSES_H
