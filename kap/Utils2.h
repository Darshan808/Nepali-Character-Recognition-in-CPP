#ifndef UTILS_H
#define UTILS_H
#include <bits/stdc++.h>
using namespace std;

class Utils
{
public:
    static std::vector<double> generateRandomVector(size_t size, double min, double max);
    static std::vector<std::vector<double>> matMul(const std::vector<std::vector<double>> &mat1, const std::vector<std::vector<double>> &mat2);
    static std::vector<std::vector<double>> matAdd(const std::vector<std::vector<double>> &mat1, const std::vector<std::vector<double>> &mat2);
    static std::vector<std::vector<double>> matrixSubtraction(const std::vector<std::vector<double>> &mat1, const std::vector<std::vector<double>> &mat2);
    static void printMatrix(const std::vector<std::vector<double>> &matrix);
    static std::vector<std::vector<double>> one_hot(const std::vector<int> &Y);
    static std::vector<std::vector<double>> transposeMatrix(const std::vector<std::vector<double>> &mat);
    static std::vector<std::vector<double>> multiplyScalarToMatrix(double scalar, const std::vector<std::vector<double>> &mat);
    static std::vector<std::vector<double>> sumRows(const std::vector<std::vector<double>> &mat);
    static std::vector<std::vector<double>> multiplyCorrespondingElements(const std::vector<std::vector<double>> &mat1, const std::vector<std::vector<double>> &mat2);
    static double getAccuracy(std::vector<int> predictions, std::vector<int> labels);
    static void normalizeData(std::vector<std::vector<double>> &data);
};

#endif 
