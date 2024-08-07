#ifndef GRAPHICS_H
#define GRAPHICS_H

#include <raylib.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

using namespace std;

class Graphics
{
private:
    static const int hPad = 50;
    static const int nodeRadius = 30;
    static const vector<Color> colors;
public:
    static Color getRandomColor();
    static void addEvenLayer(int num, int X, vector<vector<pair<int, int>>> &myNetwork);
    static void addOddLayer(int num, int X, vector<vector<pair<int, int>>> &myNetwork);
    static void animateNetwork(vector<vector<pair<int, int>>> &myNetwork, bool isAnimating, bool madePrediction);
    static void drawNetwork(bool isAnimating, bool madePrediction);
    static std::string doubleToString(double value, int precision = 2);
};
#endif // GRAPHICS_H
