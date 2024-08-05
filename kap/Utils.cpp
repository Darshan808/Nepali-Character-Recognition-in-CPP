#include "Utils2.h"
#include <bits/stdc++.h>

std::vector<double> Utils::generateRandomVector(size_t size, double min, double max)
{
    std::mt19937 gen(static_cast<unsigned int>(std::time(0)));
    std::uniform_real_distribution<> distrib(min, max);
    std::vector<double> random_vector;
    random_vector.reserve(size);
    for (size_t i = 0; i < size; ++i)
    {
        random_vector.push_back(static_cast<double>(distrib(gen)));
    }
    return random_vector;
}

std::vector<std::vector<double>> Utils::matMul(const std::vector<std::vector<double>> &mat1, const std::vector<std::vector<double>> &mat2)
{
    if (mat1[0].size() != mat2.size())
    {
        throw std::runtime_error("Size of mat1 does not match size of mat2");
    }

    size_t rows = mat1.size();
    size_t cols = mat2[0].size();
    size_t common_dim = mat2.size();
    std::vector<std::vector<double>> result(rows, std::vector<double>(cols, 0.0));

    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            for (size_t k = 0; k < common_dim; ++k)
            {
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }

    return result;
}

std::vector<std::vector<double>> Utils::matAdd(const std::vector<std::vector<double>> &mat1, const std::vector<std::vector<double>> &mat2)
{
    bool isFeedForward = false;
    if (mat1.size() != mat2.size() || mat1[0].size() != mat2[0].size())
    {
        isFeedForward = true;
    }
    std::vector<std::vector<double>> result(mat1.size(), std::vector<double>(mat1[0].size()));

    for (size_t i = 0; i < mat1.size(); ++i)
    {
        for (size_t j = 0; j < mat1[0].size(); ++j)
        {
            if (!isFeedForward)
                result[i][j] = mat1[i][j] + mat2[i][j];
            else
                result[i][j] = mat1[i][j] + mat2[i][0];
        }
    }

    return result;
}

std::vector<std::vector<double>> Utils::matrixSubtraction(const std::vector<std::vector<double>> &mat1, const std::vector<std::vector<double>> &mat2)
{
    if (mat1.size() != mat2.size() || mat1[0].size() != mat2[0].size())
    {
        throw std::runtime_error("Matrix dimensions must match for subtraction.");
    }

    std::vector<std::vector<double>> res(mat1.size(), std::vector<double>(mat1[0].size()));

    for (size_t i = 0; i < mat1.size(); ++i)
    {
        for (size_t j = 0; j < mat1[0].size(); ++j)
        {
            res[i][j] = mat1[i][j] - mat2[i][j];
        }
    }

    return res;
}

void Utils::printMatrix(const std::vector<std::vector<double>> &matrix)
{
    for (const auto &row : matrix)
    {
        std::cout << "{ ";
        for (const auto &element : row)
        {
            std::cout << element << ", ";
        }
        std::cout << "}" << std::endl;
    }
}

std::vector<std::vector<double>> Utils::one_hot(const std::vector<int> &Y)
{
    size_t size = Y.size();
    int max_val = 35;
    std::vector<std::vector<double>> one_hot_Y(max_val + 1, std::vector<double>(size, 0.0));

    for (size_t i = 0; i < size; ++i)
    {
        one_hot_Y[Y[i]][i] = 1;
    }

    return one_hot_Y;
}

std::vector<std::vector<double>> Utils::transposeMatrix(const std::vector<std::vector<double>> &mat)
{
    if (mat.empty())
    {
        return {};
    }

    size_t rows = mat.size();
    size_t cols = mat[0].size();
    std::vector<std::vector<double>> transposed(cols, std::vector<double>(rows));

    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            transposed[j][i] = mat[i][j];
        }
    }

    return transposed;
}

std::vector<std::vector<double>> Utils::multiplyScalarToMatrix(double scalar, const std::vector<std::vector<double>> &mat)
{
    std::vector<std::vector<double>> result(mat.size(), std::vector<double>(mat[0].size()));

    for (size_t i = 0; i < mat.size(); ++i)
    {
        for (size_t j = 0; j < mat[0].size(); ++j)
        {
            result[i][j] = mat[i][j] * scalar;
        }
    }

    return result;
}

std::vector<std::vector<double>> Utils::sumRows(const std::vector<std::vector<double>> &mat)
{
    std::vector<std::vector<double>> result(mat.size(), std::vector<double>(1));

    for (size_t i = 0; i < mat.size(); ++i)
    {
        double sum = 0.0;
        for (size_t j = 0; j < mat[0].size(); ++j)
        {
            sum += mat[i][j];
        }
        result[i] = {sum};
    }
    return result;
}

std::vector<std::vector<double>> Utils::multiplyCorrespondingElements(const std::vector<std::vector<double>> &mat1, const std::vector<std::vector<double>> &mat2)
{
    if (mat1.size() != mat2.size() || mat1[0].size() != mat2[0].size())
    {
        throw std::runtime_error("Matrix dimensions must match for element-wise multiplication.");
    }

    std::vector<std::vector<double>> result(mat1.size(), std::vector<double>(mat1[0].size()));

    for (size_t i = 0; i < mat1.size(); ++i)
    {
        for (size_t j = 0; j < mat1[0].size(); ++j)
        {
            result[i][j] = mat1[i][j] * mat2[i][j];
        }
    }

    return result;
}

double Utils::getAccuracy(std::vector<int> predictions, std::vector<int> labels)
{
    if (predictions.size() != labels.size())
    {
        throw std::runtime_error("Predictions size does not match with labels size!");
    }
    int total_count = predictions.size();
    int correct_count = 0;
    for (int i = 0; i < total_count; i++)
    {
        if (predictions[i] == labels[i])
        {
            correct_count++;
        }
    }
    return 1.0f * correct_count / total_count;
}

void Utils::normalizeData(std::
