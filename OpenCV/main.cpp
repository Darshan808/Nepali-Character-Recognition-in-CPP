#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <string>

int main()
{
    int index = 1;
    cv::Mat img, imgCheck, img_gray, img_resized;
    while(true && index<1000){
        // Load the JPG image
        img = cv::imread("C:\\Users\\Darshan\\Desktop\\OOP project\\assets\\input\\sample_"+std::to_string(index)+".jpg");
        if (img.empty())
        {
            if(index == 1) std::cerr << "Error: Image cannot be loaded!" << std::endl;
            break;
        }
        imgCheck = cv::imread("C:\\Users\\Darshan\\Desktop\\OOP project\\assets\\input\\sample_" + std::to_string(index) + ".png");
        if(!imgCheck.empty()){
            index++;
            continue;
        }
        // Convert to PNG format
        cv::imwrite("C:\\Users\\Darshan\\Desktop\\OOP project\\assets\\input\\sample_"+std::to_string(index)+".png", img);

        // Convert to grayscale
        cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

        // Resize the image to 32x32 using linear interpolation
        cv::resize(img_gray, img_resized, cv::Size(32, 32), 0, 0, cv::INTER_LINEAR);

        // Threshold the pixel values
        for (int i = 0; i < img_resized.rows; ++i)
        {
            for (int j = 0; j < img_resized.cols; ++j)
            {
                uchar &pixel_value = img_resized.at<uchar>(i, j);
                pixel_value = 255 - pixel_value;
                if (pixel_value < 150)
                {
                    pixel_value = 0;
                }
                else
                {
                    pixel_value = 255;
                }
            }
        }

        cv::imwrite("C:\\Users\\Darshan\\Desktop\\OOP project\\assets\\processed\\sample_"+std::to_string(index)+".png", img_resized);

        // Open the file to write pixel values
        std::ofstream outFile("C:\\Users\\Darshan\\Desktop\\OOP project\\assets\\pixels\\sample_"+std::to_string(index)+".txt");
        if (!outFile)
        {
            std::cerr << "Error: File cannot be created!" << std::endl;
            return -1;
        }

        // Write the pixel values to the file
        for (int i = 0; i < img_resized.rows; ++i)
        {
            for (int j = 0; j < img_resized.cols; ++j)
            {
                int pixel_value = img_resized.at<uchar>(i, j);
                outFile << pixel_value << " ";
            }
        }

        outFile.close();
        std::cout << "Pixel values have been written to sample_"+std::to_string(index)+".txt" << std::endl;
        index++;
    }
    return 0;
}

// cmake -B ./build
// cmake --build ./build
// ./build/Debug/OpenCVExecutable.exe

//To do:
//Make NeuralNetwork and raylib under same folder!
//Make this preprocess also under same folder!
//Use bat script to automate the data preprocessing and conversion to png for display
//Done!