#ifndef DANFE_H
#define DANFE_H

#include <bits/stdc++.h>
#include "Utils.h"

class Danfe
{
public:
    static void loadData(std::vector<std::vector<double>> &data, std::string fileName, bool verbose = false);
    static void xySplit(std::vector<std::vector<double>> &data, std::vector<std::vector<double>> &X_train, std::vector<int> &Y_train, int lower_range, bool verbose = false);
    static void extractColumn(std::vector<std::vector<double>> &data, int index, std::vector<std::vector<double>> &single, bool save = false);
    static std::string changeExtension(const std::string &filename, const std::string &newExtension);
    static void extractPixels(std::string fname, std::vector<std::vector<double>> &data);
    static void getLatestImgPixels(std::vector<std::vector<double>> &data);
};

#endif
