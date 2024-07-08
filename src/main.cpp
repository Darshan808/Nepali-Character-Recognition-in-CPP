#include <raylib.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <include/Graphics.h>
#include <include/NeuralNetwork.h>
#include <include/Danfe.h>
#include <include/Utils.h>

using namespace std;

class myGraphics{
    private:
    Font customFont;
    Font customFont24;
    Color lightGreen;
    Image outImg;
    Texture uploadTexture;
    Rectangle uploadRect;
    Texture qmTexture;
    Rectangle qmRect;

    // Load image and create texture
    Texture2D inputTexture;
    vector<Texture2D> outputTextureNum;
    std::string filename;

    // Assuming Rectangle and Vector2 are defined types in your project
    Rectangle sourceRect;
    Rectangle destRect;
    Rectangle destRect2;
    Rectangle inputImgRect;
    Vector2 origin;

    std::string inputText;
    char key;

    Rectangle buttonRect; // x, y, width, height
    Color buttonColor;
    bool buttonClicked;

    bool isAnimating;
    double animationTime;
    double ANIMATION_DURATION;
    bool inputImageLoaded;
    bool predictionMade;
    int PREDICTION;
    float CONFIDENCE;
    NeuralNetwork *NN;
    vector<vector<double>> predict_X;

public:
    void initialize(){
        InitWindow(1280, 720, "Nepali Digital Recognition");
        SetTargetFPS(15);

        customFont = LoadFontEx("C:\\Users\\Darshan\\Desktop\\OOP project\\assets\\font\\Roboto-Regular.ttf", 32, NULL, 250);
        customFont24 = LoadFontEx("C:\\Users\\Darshan\\Desktop\\OOP project\\assets\\font\\Roboto-Regular.ttf", 24, NULL, 250);

        lightGreen = Color({20, 160, 133, 255});

        // Load misc images
        outImg = LoadImage("C:\\Users\\Darshan\\Desktop\\OOP project\\assets\\misc\\upload.png");
        uploadTexture = LoadTextureFromImage(outImg);
        uploadRect = {0, 0, static_cast<float>(uploadTexture.width),static_cast<float>(uploadTexture.height)};
        outImg = LoadImage("C:\\Users\\Darshan\\Desktop\\OOP project\\assets\\misc\\qm.png");
        qmTexture = LoadTextureFromImage(outImg);
        qmRect = {0, 0, static_cast<float>(qmTexture.width), static_cast<float>(qmTexture.height)};

        // Load image and create texture
        outputTextureNum.resize(10);
        for (int i = 0; i < 10; i++)
        {
            filename = "C:\\Users\\Darshan\\Desktop\\OOP project\\assets\\0_9\\" + std::to_string(i) + ".png";
            const char *filename_cstr = filename.c_str();
            outImg = LoadImage(filename_cstr);
            outputTextureNum[i] = LoadTextureFromImage(outImg);
            UnloadImage(outImg); // Unload image data from RAM as it's now in VRAM
        }

        // Assuming Rectangle and Vector2 are defined types in your project
        sourceRect = {0, 0, static_cast<float>(outputTextureNum[0].width), static_cast<float>(outputTextureNum[0].height)};
        destRect = {20, 285, 150, 150};
        destRect2 = {1080, 285, 150, 150};
        origin = {0, 0};

        buttonRect = {540, 650, 300, 50}; // x, y, width, height
        buttonColor = GRAY;
        buttonClicked = false;

        isAnimating = false;
        animationTime = 0.0;
        ANIMATION_DURATION = 3.0;
        inputImageLoaded = false;
        predictionMade = false;
        PREDICTION = 0;
        CONFIDENCE = -1;
        NN = new NeuralNetwork({1024, 10, 10}, 0.0001, "numbers.txt");

        while (WindowShouldClose() == false)
        {

            Vector2 mousePoint = GetMousePosition();
            bool mouseOverButton = CheckCollisionPointRec(mousePoint, buttonRect);

            if (mouseOverButton)
            {
                buttonColor = LIGHTGRAY;
                if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
                {
                    buttonClicked = true;
                    isAnimating = true;
                    animationTime = GetTime();
                    pair<int, double> p = NN->predict(predict_X);
                    PREDICTION = p.first;
                    CONFIDENCE =static_cast<float>(p.second);
                }
            }
            else
            {
                buttonColor = GRAY;
            }
            if (isAnimating && (GetTime() - animationTime >= ANIMATION_DURATION))
            {
                isAnimating = false;
                predictionMade = true;
            }
            // Get char pressed (unicode character) on the frame
            key = GetCharPressed();

            // Check if more characters have been pressed on the same frame
            while (key > 0)
            {
                if ((key >= 32) && (key <= 125))
                {
                    inputText += key;
                }
                key = GetCharPressed(); // Check next character in the queue
            }

            if (IsKeyPressed(KEY_BACKSPACE) && !inputText.empty())
            {
                inputText.pop_back();
            }
            if (IsKeyPressed(KEY_ENTER) && !inputText.empty())
            {
                string filename = "C:\\Users\\Darshan\\Desktop\\OOP project\\assets\\input\\" + inputText;
                const char *filename_cstr = filename.c_str();
                outImg = LoadImage(filename_cstr);
                inputTexture = LoadTextureFromImage(outImg);
                inputImgRect = {0, 0, static_cast<float>(inputTexture.width), static_cast<float>(inputTexture.height)};
                UnloadImage(outImg); // Unload image data from RAM as it's now in VRAM
                Danfe::extractPixels(inputText, predict_X);
                Utils::normalizeData(predict_X);
                inputImageLoaded = true;
            }
            if (IsKeyPressed(KEY_TAB))
            {
                isAnimating = false;
                predictionMade = false;
                inputImageLoaded = false;
                predict_X.clear();
                inputText.clear();
            }
            BeginDrawing();
            ClearBackground(lightGreen);
            Graphics::drawNetwork(isAnimating, predictionMade);
            DrawTexturePro(inputImageLoaded ? inputTexture : uploadTexture, inputImageLoaded ? inputImgRect : uploadRect, destRect, origin, 0.0f, WHITE);
            DrawTexturePro(isAnimating ? outputTextureNum[GetRandomValue(0, 9)] : predictionMade ? outputTextureNum[PREDICTION]
                                                                                                 : qmTexture,
                           isAnimating ? sourceRect : predictionMade ? sourceRect
                                                                     : qmRect,
                           destRect2, origin, 0.0f, WHITE);
            if (predictionMade)
                DrawTextEx(customFont, ("Confidence: " + Graphics::doubleToString(CONFIDENCE)).c_str(), Vector2({1040, 450}), 32, 1, WHITE);
            DrawTextEx(customFont24, "Image:", Vector2({20, 460}), 24, 1, WHITE);
            DrawTextEx(customFont24, inputText.c_str(), Vector2({20, 500}), 24, 1, WHITE);
            DrawLine(15, 530, 150, 530, WHITE);
            DrawRectangleRec(buttonRect, buttonColor);
            DrawRectangleLinesEx(buttonRect, 2, BLACK);
            DrawTextEx(customFont, "Make Prediction!", Vector2({buttonRect.x + 50, buttonRect.y + 10}), 32, 1, WHITE);

            if (buttonClicked)
            {
                buttonClicked = false; // Reset for next frame
            }

            EndDrawing();
        }
        for (Texture2D tex : outputTextureNum)
        {
            UnloadTexture(tex);
        }
        CloseWindow();
    }
};

int main()
{
    myGraphics gp;
    gp.initialize();
    return 0;
}