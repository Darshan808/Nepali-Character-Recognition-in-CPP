# Nepali Handwritten Digits and Character Recognition

Nepali Handwritten digits and character recognition model programmed and trained using C++ following an object-oriented approach.

## Table of Contents

- [Preview](#preview)
- [Getting Started](#getting-started)
- [Features](#features)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Preview

![preview.png](./assets/preview.png?raw=true)

## Getting Started

To get started with this project, follow these steps:

1. **Clone or download this repository to your local machine:**
    ```bash
    git clone https://github.com/Darshan808/Nepali-Character-Recognition-in-CPP
    cd nepali-character-recognition-in-cpp
    ```

2. **Install the necessary dependencies for your platform:**
    - **OpenCV**: For image preprocessing
        - Download and install OpenCV from [OpenCV releases](https://opencv.org/releases/)
    - **Raylib**: For rendering graphics
        - Download and install Raylib from [Raylib](https://www.raylib.com/)

3. **Open the project in your preferred IDE or text editor.**

4. **Build the project:**
    - Use your IDE's build tools, or run the provided build script (if applicable).

5. **Run the program:**
    - On Windows, you can use the `start.bat` script:
        ```bash
        start.bat
        ```

## Features

- **Image Preprocessing**: Converts images to grayscale and normalizes pixel values using OpenCV.
- **Neural Network Training**: Implements a simple Artificial Neural Network (ANN) for recognizing Nepali characters and digits.
- **Real-time Recognition**: Processes input images and displays the recognition results using Raylib.
- **Modular Design**: Follows an object-oriented approach for easy maintenance and scalability.


## Usage

1. **Training the Model:**
    - Ensure your training dataset is placed in the `dataset/train/` directory.
    - Run the training module (details depend on your specific implementation in `model/train/main.cpp`).

2. **Testing the Model:**
    - Place your test images in the `assets/input/` directory.
    - Run the testing module to evaluate the model's performance.

3. **Real-time Recognition:**
    - Use the provided UI to upload an image and see the recognition results in real-time.

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenCV](https://opencv.org/) for providing powerful image processing libraries.
- [Raylib](https://www.raylib.com/) for an easy-to-use graphics library.
- Thanks to all contributors and the open-source community for their valuable support and contributions.


