#ifndef UTILS_H
#define UTILS_H

#include <bits/stdc++.h>
using namespace std;

class Utils
{
public:
    static vector<double> generateRandomVector(size_t size, double min, double max)
    {
        mt19937 gen(static_cast<unsigned int>(time(0)));
        uniform_real_distribution<> distrib(min, max);
        vector<double> random_vector;
        random_vector.reserve(size);
        for (size_t i = 0; i < size; ++i)
        {
            random_vector.push_back(static_cast<double>(distrib(gen)));
        }
        return random_vector;
    }
    static vector<vector<double>> matMul(const vector<vector<double>> &mat1, const vector<vector<double>> &mat2)
    {
        if (mat1[0].size() != mat2.size())
        {
            throw runtime_error("Size of mat1 does not match size of mat2");
        }

        // Initialize the result matrix with appropriate dimensions
        size_t rows = mat1.size();
        size_t cols = mat2[0].size();
        size_t common_dim = mat2.size();
        vector<vector<double>> result(rows, vector<double>(cols, 0.0));

        // Perform matrix multiplication
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
    // Function to add two matrices element-wise
    static vector<vector<double>> matAdd(const vector<vector<double>> &mat1, const vector<vector<double>> &mat2)
    {
        // Check if both matrices have the same dimensions
        bool isFeedForward = false;
        if (mat1.size() != mat2.size() || mat1[0].size() != mat2[0].size())
        {
            isFeedForward = true;
        }
        // Initialize the result matrix with the same dimensions as mat1 and mat2
        vector<vector<double>> result(mat1.size(), vector<double>(mat1[0].size()));

        // Perform element-wise addition
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
    static vector<vector<double>> matrixSubtraction(const vector<vector<double>> &mat1, const vector<vector<double>> &mat2)
    {
        if (mat1.size() != mat2.size() || mat1[0].size() != mat2[0].size())
        {
            throw runtime_error("Matrix dimensions must match for subtraction.");
        }

        vector<vector<double>> res(mat1.size(), vector<double>(mat1[0].size()));

        for (size_t i = 0; i < mat1.size(); ++i)
        {
            for (size_t j = 0; j < mat1[0].size(); ++j)
            {
                res[i][j] = mat1[i][j] - mat2[i][j];
            }
        }

        return res;
    }
    static void printMatrix(const vector<vector<double>> &matrix)
    {
        for (const auto &row : matrix)
        {
            cout << "{ ";
            for (const auto &element : row)
            {
                cout << element << ", ";
            }
            cout << "}" << endl;
        }
    }
    static vector<vector<double>> one_hot(const vector<int> &Y)
    {
        // Find the size and max value in Y
        size_t size = Y.size();
        int max_val = 9;

        // Initialize the one-hot encoded matrix with zeros
        vector<vector<double>> one_hot_Y(max_val + 1, vector<double>(size, 0.0));

        // Fill in the one-hot encoded matrix
        for (size_t i = 0; i < size; ++i)
        {
            one_hot_Y[Y[i]][i] = 1;
        }

        return one_hot_Y;
    }
    static vector<vector<double>> transposeMatrix(const vector<vector<double>> &mat)
    {
        if (mat.empty())
        {
            return {};
        }

        size_t rows = mat.size();
        size_t cols = mat[0].size();
        vector<vector<double>> transposed(cols, vector<double>(rows));

        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                transposed[j][i] = mat[i][j];
            }
        }

        return transposed;
    }
    // Function to multiply a scalar to all elements of a matrix
    static vector<vector<double>> multiplyScalarToMatrix(double scalar, const vector<vector<double>> &mat)
    {
        // Create a result matrix with the same dimensions as the input matrix
        vector<vector<double>> result(mat.size(), vector<double>(mat[0].size()));

        // Iterate over each element and multiply by the scalar
        for (size_t i = 0; i < mat.size(); ++i)
        {
            for (size_t j = 0; j < mat[0].size(); ++j)
            {
                result[i][j] = mat[i][j] * scalar;
            }
        }

        return result;
    }
    // Function to sum all elements of each row and return a vector with one column
    static vector<vector<double>> sumRows(const vector<vector<double>> &mat)
    {
        // Create a result vector with the same number of rows as the input matrix
        vector<vector<double>> result(mat.size(), vector<double>(1));

        // Iterate over each row
        for (size_t i = 0; i < mat.size(); ++i)
        {
            // Sum the elements of the row
            double sum = 0.0;
            for (size_t j = 0; j < mat[0].size(); ++j)
            {
                sum += mat[i][j];
            }
            // Store the sum in the result vector
            result[i] = {sum};
        }
        return result;
    }
    // Function to multiply corresponding elements of two matrices
    static vector<vector<double>> multiplyCorrespondingElements(const vector<vector<double>> &mat1, const vector<vector<double>> &mat2)
    {
        // Check if both matrices have the same dimensions
        if (mat1.size() != mat2.size() || mat1[0].size() != mat2[0].size())
        {
            throw runtime_error("Matrix dimensions must match for element-wise multiplication.");
        }

        // Create a result matrix with the same dimensions as the input matrices
        vector<vector<double>> result(mat1.size(), vector<double>(mat1[0].size()));

        // Iterate over each element and multiply the corresponding elements of mat1 and mat2
        for (size_t i = 0; i < mat1.size(); ++i)
        {
            for (size_t j = 0; j < mat1[0].size(); ++j)
            {
                result[i][j] = mat1[i][j] * mat2[i][j];
            }
        }

        return result;
    }
    static double getAccuracy(vector<int> predictions, vector<int> labels)
    {
        if (predictions.size() != labels.size())
        {
            throw runtime_error("Predictions size does not match with labels size!");
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
    static void normalizeData(std::vector<std::vector<double>> &data)
    {
        for (auto &row : data)
        {
            for (auto &pixel : row)
            {
                pixel /= 255.0f;
            }
        }
    }
};

#endif