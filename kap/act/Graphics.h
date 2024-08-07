#include <raylib.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
using namespace std;

class Graphics{
    private:
    static const int hPad = 50;
    static const int nodeRadius = 30;

    static const vector<Color> colors;
    
    public:

    static Color getRandomColor()
    {
        int randomIndex = GetRandomValue(0, colors.size() - 1);
        return colors[randomIndex];
    }

    static void addEvenLayer(int num, int X, vector<vector<pair<int, int>>> &myNetwork)
    {
        int Y = 360 - (num / 2.0 - 0.5) * nodeRadius * 2 - (num / 2.0 - 0.5) * hPad;
        vector<pair<int, int>> layer;
        for (int i = 0; i < num; i++)
        {
            // DrawCircle(X,Y,nodeRadius,RED);
            layer.push_back({X, Y});
            Y += (hPad + 2 * nodeRadius);
        }
        myNetwork.push_back(layer);
    }

    static void addOddLayer(int num, int X, vector<vector<pair<int, int>>> &myNetwork)
    {
        int up = num / 2;
        int Y = 360 - (up)*nodeRadius * 2 - (up)*hPad;
        vector<pair<int, int>> layer;
        for (int i = 0; i < num; i++)
        {
            // DrawCircle(X,Y,nodeRadius,RED);
            layer.push_back({X, Y});
            Y += (hPad + 2 * nodeRadius);
        }
        myNetwork.push_back(layer);
    }

    static void animateNetwork(vector<vector<pair<int, int>>> &myNetwork, bool isAnimating, bool madePrediction)
    {

        for (int i = 0; i < myNetwork.size() - 1; i++)
        {
            int iSize = myNetwork[i].size();
            int jSize = myNetwork[i + 1].size();
            for (int c = 0; c < iSize; c++)
            {
                for (int d = 0; d < jSize; d++)
                {
                    DrawLine(myNetwork[i][c].first, myNetwork[i][c].second, myNetwork[i + 1][d].first, myNetwork[i + 1][d].second, isAnimating ? getRandomColor() : GREEN);
                }
            }
        }

        int p = 10;
        for (vector<pair<int, int>> layer : myNetwork)
        {
            int ls = layer.size();
            pair<int, int> s = layer[0];
            pair<int, int> e = layer[ls - 1];
            pair<int, int> tl = {s.first - nodeRadius - p, s.second - nodeRadius - p};
            pair<int, int> bl = {tl.first, e.second + nodeRadius + p};
            pair<int, int> tr = {s.first + nodeRadius + p, tl.second};
            pair<int, int> br = {e.first + nodeRadius + p, bl.second};
            DrawLine(tl.first, tl.second, bl.first, bl.second, RED);
            DrawLine(tl.first, tl.second, tr.first, tr.second, RED);
            DrawLine(tr.first, tr.second, br.first, br.second, RED);
            DrawLine(br.first, br.second, bl.first, bl.second, RED);
            for (pair<int, int> node : layer)
            {
                DrawCircle(node.first, node.second, nodeRadius, isAnimating ? getRandomColor() : madePrediction ? GOLD
                                                                                                                : BLUE);
            }
        }
    }

    static void drawNetwork(bool isAnimating, bool madePrediction)
    {
        std::vector<int> v = {6, 4, 2};
        int X = 320;   // ending 1060 //Gap 840
        int gap = 300; // 840/3
        vector<vector<pair<int, int>>> myNetwork;
        for (size_t i = 0; i < v.size(); i++)
        {
            if (v[i] % 2 == 0)
                addEvenLayer(v[i], X, myNetwork);
            else
                addOddLayer(v[i], X, myNetwork);
            X += gap;
        }
        animateNetwork(myNetwork, isAnimating, madePrediction);
    }
    static std::string doubleToString(double value, int precision = 2)
    {
        return to_string(value).substr(0,4);
    }
};

const vector<Color> Graphics::colors {BLUE, RED, GREEN, YELLOW, ORANGE, PURPLE, PINK, SKYBLUE, LIME, GOLD, VIOLET, DARKBLUE, DARKGREEN, DARKPURPLE, BEIGE, BROWN, GRAY, MAGENTA, MAROON, RAYWHITE};