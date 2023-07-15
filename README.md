# American Sign Language Detection

American Sign Language Detection is a deep learning project that aims to accurately recognize and interpret American Sign Language (ASL) gestures in real time. This repository provides a comprehensive solution for ASL detection using computer vision techniques and machine learning algorithms.

![ASL Detection Demo](demo.gif)

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)

## Introduction

American Sign Language (ASL) is a visual language used by the Deaf and hard-of-hearing community. ASL relies on hand gestures, facial expressions, and body movements to communicate. ASL Detection project leverages deep learning and computer vision techniques to recognize and interpret ASL gestures, providing a means of communication for individuals with hearing impairments.

This repository provides a pre-trained deep learning model that can detect and interpret ASL gestures in real time using a webcam or pre-recorded videos. It includes the necessary code to preprocess input data, train the model, and perform real-time inference.

## Installation

To install the ASL Detection project, follow these steps:

1. Clone this repository to your local machine:

```
git clone https://github.com/your-username/asl-detection.git
```

2. Change to the project directory:

```
cd asl-detection
```


3. Create a virtual environment (optional but recommended):

```
python -m venv env
source env/bin/activate # For Linux/Mac
env\Scripts\activate # For Windows
```

4. Install the required dependencies:

```
pip install -r requirements.txt
```


5. Download the pre-trained model weights:

[Download the model weights](https://example.com/model_weights.pth) and place them in the `weights/` directory.

## Usage

To use the ASL Detection project, follow these steps:

1. Ensure your webcam is connected and functional.

2. Run the following command to start the real-time ASL gesture detection:

```
python detect_asl.py
```


This will open a new window showing the webcam feed, with ASL gesture detection overlays.

3. Perform ASL gestures in front of the webcam, and the program will recognize and interpret them in real-time.

Alternatively, you can use pre-recorded videos by specifying the file path as an argument:

```
python detect_asl.py --video /path/to/video.mp4
```


This will process the specified video and display the ASL gesture detection results.

## Model Training

If you want to train the ASL Detection model on your own dataset, you can follow these steps:

1. Prepare your ASL gesture dataset, ensuring that it is properly labeled.

2. Place the dataset in the `data/` directory, following the required format.

3. Run the following command to start the model training:

```
python train.py
```


This will train the model on the provided dataset and save the trained weights in the `weights/` directory.

Please refer to the `data/` directory for more information on dataset preparation and format.

## Contributing

Contributions to the ASL Detection project are welcome! If you find any issues or want to add new features, please submit a pull request. Make sure to follow the contribution guidelines outlined in the `CONTRIBUTING.md` file.

## License

The ASL Detection project is made as Part of the University curriculum by
Mainak Chattopadhyay 21BAI1217, Parthiba Mukhopadhyay 21BAI1168, Pallav Gupta 21BAI1169.

