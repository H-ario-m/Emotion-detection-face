# Emotion Detection Project

This project is an emotion detection system that uses Convolutional Neural Networks (CNN) to classify emotions from facial expressions. The project is divided into two parts: training the model and using the trained model to detect emotions in real-time using a webcam.

## Table of Contents

1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Training the Model](#training-the-model)
4. [Real-Time Emotion Detection](#real-time-emotion-detection)
5. [Model Evaluation](#model-evaluation)
6. [License](#license)

## Installation

1. Clone this repository to your local machine.

   ```bash
   git clone https://github.com/yourusername/emotion-detection.git
   ```
   
2. Navigate to the project directory.

   ```bash
   cd emotion-detection
   ```
   
3. Install the required dependencies.

   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

1. Download the emotion dataset and split it into training and testing directories.
2. Place the training images in the `train` directory and the testing images in the `test` directory.

```
├── data
│   ├── train
│   │   ├── Angry
│   │   ├── Nothing
│   │   ├── Happy
│   │   ├── Sad
│   ├── test
│   │   ├── Angry
│   │   ├── Nothing
│   │   ├── Happy
│   │   ├── Sad
```

## Training the Model

The `new_improved.py` script is used to train the emotion detection model. You can change the path directory according to your dataset and train the model based on the dataset.

Run the script to start the training process:

```bash
python new_improved.py
```

The model will be trained and saved as `final_emotion_detection_model.keras`.

## Real-Time Emotion Detection

The `newcam.py` script is used for real-time emotion detection using a webcam. The script loads the trained model and detects emotions in real-time.

Run the script to start the emotion detection:

```bash
python newcam.py
```

Press `q` to quit the application.

## Model Evaluation

After training, the model is evaluated on the test set to calculate the test loss and test accuracy. The results will be printed on the console:

```python
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
```
