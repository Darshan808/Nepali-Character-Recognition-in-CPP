#include <raylib.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <include/bup/Graphics.h>
#include <include/bup/NeuralNetwork.h>
#include <include/bup/Danfe.h>
#include <include/bup/Utils.h>

using namespace std;

class myGraphics{
    private:
    string modelChoosen = "num";
    Font customFont;
    Font customFont24;
    Color lightGreen;
    Image outImg;
    Texture uploadTexture;
    Rectangle uploadRect;
    Texture qmTexture;
    Rectangle qmRect;
    Texture switchTexture;
    Rectangle switchRect;

    // Load image and create texture
    Texture2D inputTexture;
    vector<Texture2D> outputTextureNum;
    vector<Texture2D> outputTextureChar;
    std::string filename;

    // Assuming Rectangle and Vector2 are defined types in your project
    Rectangle sourceRect;
    Rectangle destRect;
    Rectangle destRect2;
    Rectangle swtichDestRect;
    Rectangle inputImgRect;
    Vector2 origin;

    std::string inputText;
    char key;

    Rectangle buttonRect; // x, y, width, height
    Rectangle modelButtonRect; // x, y, width, height
    Color buttonColor;
    Color modelButtonColor;
    bool buttonClicked;

    bool isAnimating;
    double animationTime;
    double ANIMATION_DURATION;
    bool inputImageLoaded;
    bool predictionMade;
    int PREDICTION;
    float CONFIDENCE;
    NeuralNetwork *NN_num, *NN_char;
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
        outImg = LoadImage("C:\\Users\\Darshan\\Desktop\\OOP project\\assets\\misc\\switch_small.png");
        switchTexture = LoadTextureFromImage(outImg);
        switchRect = {0, 0, static_cast<float>(switchTexture.width), static_cast<float>(switchTexture.height)};

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
        //Load characters
        outputTextureChar.resize(36);
        for(int i=0;i<36;i++){
            filename = "C:\\Users\\Darshan\\Desktop\\OOP project\\assets\\0_35\\" + std::to_string(i) + ".png";
            const char *filename_cstr = filename.c_str();
            outImg = LoadImage(filename_cstr);
            outputTextureChar[i] = LoadTextureFromImage(outImg);
            UnloadImage(outImg); // Unload image data from RAM as it's now in VRAM
        }

        // Assuming Rectangle and Vector2 are defined types in your project
        sourceRect = {0, 0, static_cast<float>(outputTextureNum[0].width), static_cast<float>(outputTextureNum[0].height)};
        destRect = {20, 285, 150, 150};
        destRect2 = {1080, 285, 150, 150};
        origin = {0, 0};
        

        buttonRect = {540, 650, 300, 50}; // x, y, width, height
        modelButtonRect = {928, 50, 300, 50}; // x, y, width, height
        buttonColor = GRAY;
        modelButtonColor = GRAY;
        buttonClicked = false;
        swtichDestRect = {modelButtonRect.x + modelButtonRect.width - 50 , modelButtonRect.y+5,modelButtonRect.height-10, modelButtonRect.height-10};

        isAnimating = false;
        animationTime = 0.0;
        ANIMATION_DURATION = 3.0;
        inputImageLoaded = false;
        predictionMade = false;
        PREDICTION = 0;
        CONFIDENCE = -1;
        NN_num = new NeuralNetwork({1024, 64, 32, 10}, 0.0001, "number.txt");
        NN_char = new NeuralNetwork({1024, 128, 64, 32, 36}, 0.001, "alphabet.txt");

        while (WindowShouldClose() == false)
        {

            Vector2 mousePoint = GetMousePosition();
            bool mouseOverButton = CheckCollisionPointRec(mousePoint, buttonRect);
            bool mouseOverModelButton = CheckCollisionPointRec(mousePoint, modelButtonRect);

            if (mouseOverModelButton)
            {
                modelButtonColor = LIGHTGRAY;
                if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
                {
                    //switch model and display accordingly
                    if(modelChoosen == "num")
                        modelChoosen = "char";
                    else
                        modelChoosen = "num";
                }
            }
            else
            {
                modelButtonColor = GRAY;
            }

            if (mouseOverButton)
            {
                buttonColor = LIGHTGRAY;
                if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
                {
                    buttonClicked = true;
                    isAnimating = true;
                    animationTime = GetTime();
                    pair<int, double> p;
                    if(modelChoosen == "num")
                        p = NN_num->predict(predict_X);
                    else
                        p = NN_char->predict(predict_X);
                        
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
                predictionMade = false;
                string filename = "C:\\Users\\Darshan\\Desktop\\OOP project\\assets\\input\\" + inputText + ".png";
                const char *filename_cstr = filename.c_str();
                outImg = LoadImage(filename_cstr);
                inputTexture = LoadTextureFromImage(outImg);
                inputImgRect = {0, 0, static_cast<float>(inputTexture.width), static_cast<float>(inputTexture.height)};
                UnloadImage(outImg); // Unload image data from RAM as it's now in VRAM
                Danfe::extractPixels(inputText, predict_X);
                // Utils::normalizeData(predict_X);
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
            DrawTexturePro(isAnimating ? modelChoosen == "num" ? outputTextureNum[GetRandomValue(0, 9)] : outputTextureChar[GetRandomValue(0,35)] : predictionMade ? modelChoosen == "num" ? outputTextureNum[PREDICTION] : outputTextureChar[PREDICTION]
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
            DrawRectangleRec(modelButtonRect, modelButtonColor);
            DrawRectangleLinesEx(modelButtonRect, 2, BLACK);
            DrawTextEx(customFont, "Make Prediction", Vector2({buttonRect.x + 50, buttonRect.y + 10}), 32, 1,!mouseOverButton ? WHITE : BLACK);
            DrawTextEx(customFont, modelChoosen == "num" ? "Digits Recognition" : "Letter Recognition", Vector2({modelButtonRect.x + 15, modelButtonRect.y + 10}), 32, 1, !mouseOverModelButton ? WHITE : BLACK);
            DrawTexturePro(switchTexture, switchRect, swtichDestRect, origin, 0.0f, WHITE);

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