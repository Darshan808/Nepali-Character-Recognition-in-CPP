#include "Danfe2.h"
#include <bits/stdc++.h>

void Danfe::loadData(std::vector<std::vector<double>> &data, std::string fileName, bool verbose)
{
    std::ifstream file(fileName); // Open the CSV file

    if (!file.is_open())
    {
        std::cerr << "Failed to open the file." << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ' '))
        {
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

void Danfe::xySplit(std::vector<std::vector<double>> &data, std::vector<std::vector<double>> &X_train, std::vector<int> &Y_train, int lower_range, bool verbose)
{
    for (const auto &sample : data)
    {
        std::vector<double> pixels(sample.begin() + 1, sample.end());
        X_train.push_back(pixels);
        Y_train.push_back(static_cast<int>(sample[0]) - lower_range);
    }

    X_train = Utils::transposeMatrix(X_train);
    Utils::normalizeData(X_train);

    if (verbose)
    {
        std::cout << "Size of X_train: " << X_train.size() << ',' << X_train[0].size() << std::endl;
        std::cout << "Size of Y_train: " << Y_train.size() << std::endl;
    }
}

void Danfe::extractColumn(std::vector<std::vector<double>> &data, int index, std::vector<std::vector<double>> &single, bool save)
{
    int rows = data.size();
    int cols = data[0].size();

    if (index >= cols)
    {
        throw std::runtime_error("Index out of bound, cannot extract column!");
    }

    single.resize(rows, std::vector<double>(1));
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

std::string Danfe::changeExtension(const std::string &filename, const std::string &newExtension)
{
    std::size_t lastDot = filename.find_last_of('.');
    if (lastDot == std::string::npos)
    {
        return filename + newExtension;
    }
    else
    {
        return filename.substr(0, lastDot) + newExtension;
    }
}

void Danfe::extractPixels(std::string fname, std::vector<std::vector<double>> &data)
{
    fname = changeExtension(fname, ".txt");
    std::ifstream inFile("C:\\Users\\Darshan\\Desktop\\OOP project\\assets\\pixels\\" + fname);

    if (!inFile.is_open())
    {
        std::cerr << "Error: Pixel File could not be opened or found!" << std::endl;
        return;
    }

    std::string line;
    std::getline(inFile, line);
    std::istringstream iss(line);
    int pv;
    data.resize(32 * 32, std::vector<double>(1));
    for (int i = 0; i < 32 * 32; i++)
    {
        iss >> data[i][0];
    }

    inFile.close();
}

void Danfe::getLatestImgPixels(std::vector<std::vector<double>> &data)
{
    int index = 1;
    std::ifstream checkFile;

    while (true)
    {
        std::string filename = "C:\\Users\\Darshan\\Desktop\\OOP project\\assets\\pixels\\sample_" + std::to_string(index++) + ".txt";
        checkFile.open(filename);

        if (!checkFile)
        {
            break;
        }

        checkFile.close();
    }

    extractPixels("sample_" + std::to_string(index - 2) + ".txt", data);
}
