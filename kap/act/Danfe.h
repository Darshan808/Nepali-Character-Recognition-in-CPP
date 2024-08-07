#ifndef MYCLASSES_H
#define MYCLASSES_H

#include <bits/stdc++.h>
#include "Utils.h"

class Danfe
{
public:
    static void loadData(std::vector<std::vector<double>> &data, string fileName, bool verbose = false)
    {
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
            std::vector<double> row;
            std::stringstream ss(line);
            std::string cell;

            while (std::getline(ss, cell, ' '))
            {                                   // Split by comma
                row.push_back(std::stof(cell)); // Convert to int and add to row
            }

            data.push_back(row); // Add row to the data vector
        }
        if (verbose)
        {
            std::cout << "Size of training data: " << data.size() << ',' << data[0].size() << std::endl;
        }
        file.close();
    }
    static void xySplit(vector<vector<double>> &data, vector<vector<double>> &X_train, vector<int> &Y_train, int lower_range, bool verbose = false)
    {
        for (const auto &sample : data)
        {
            vector<double> pixels(sample.begin() + 1, sample.end());
            X_train.push_back(pixels);
            Y_train.push_back(static_cast<int>(sample[0]) - lower_range);
        }
        X_train = Utils::transposeMatrix(X_train);
        Utils::normalizeData(X_train);
        if (verbose)
        {
            std::cout << "Size of X_train: " << X_train.size() << ',' << X_train[0].size() << std::endl;
            std::cout << "Size of Y_train: " << Y_train.size() << endl;
        }
    }
    static void extractColumn(vector<vector<double>> &data, int index, vector<vector<double>> &single, bool save = false)
    {
        int rows = data.size();
        int cols = data[0].size();
        if (index >= cols)
        {
            throw runtime_error("Index out of bound, cannot extract column!");
        }
        single.resize(rows, vector<double>(1));
        for (int i = 0; i < rows; i++)
        {
            single[i][0] = data[i][index];
        }
        if (save)
        {
            std::ofstream outFile("prediction.txt", std::ios::out);
            if (!outFile)
            {
                std::cerr << "Error opening file for writing!" << std::endl;
                return;
            }
            for (int i = 0; i < rows; i++)
            {
                outFile << static_cast<int>(single[i][0] * 255) << " ";
            }
            outFile.close();
        }
    }
    // Function to change the file extension
    static std::string changeExtension(const std::string &filename, const std::string &newExtension)
    {
        std::size_t lastDot = filename.find_last_of('.');
        if (lastDot == std::string::npos)
        {
            // No dot found, just append the new extension
            return filename + newExtension;
        }
        else
        {
            // Replace the extension
            return filename.substr(0, lastDot) + newExtension;
        }
    }
    static void extractPixels(string fname, vector<vector<double>> &data)
    {
        fname = changeExtension(fname,".txt");
        ifstream inFile("C:\\Users\\Darshan\\Desktop\\OOP project\\assets\\pixels\\" + fname);
        if (!inFile.is_open())
        {
            std::cerr << "Error: Pixel File could not be opened or found!" << std::endl;
        }
        string line;
        getline(inFile, line);
        istringstream iss(line);
        int pv;
        data.resize(32 * 32, vector<double>(1));
        for (int i = 0; i < 32 * 32; i++)
        {
            iss >> data[i][0];
        }
        inFile.close();
    }
    static void getLatestImgPixels(vector<vector<double>> &data)
    {
        int index = 1;
        std::ifstream checkFile;
        while (true)
        {
            std::string filename = "C:\\Users\\Darshan\\Desktop\\OOP project\\assets\\pixels\\sample_" + std::to_string(index++) + ".txt";
            checkFile.open(filename);

            if (!checkFile)
                break;
            checkFile.close();
        }
        extractPixels("sample_" + to_string(index - 2) + ".txt", data);
    }
};

#endif // MYCLASSES_H
