# Multimodal-Sentiment-Analysis

This project performs sentiment analysis on Twitter data using both text and image information. It uses a combination of pre-trained models and custom neural networks to classify tweets as positive, negative, or neutral.

## What's Inside

* `twitter-dataset-for-sentiment-analysis.zip`: The dataset (you'll need to download it from Kaggle).
* `twitter_dataset_for_sentiment_analysis/`: Extracted dataset folder.
* `multimodal_model.pth`: The saved weights of the multimodal model.
* `text_classifier/`: Saved text classification model.
* `img_classifier/`: Saved image classification model.
* `MultimodalPreprocessor.joblib`: The saved preprocessor.
* `MultimodalDataset.joblib`: The saved dataset.
* `requirements.txt`: A list of the Python packages needed to run the code.
* `Multimodal-Sentiment-Analysis.ipynb` file: The main python script.

## Getting Started

1.  **Open the ipynb file in Google Coalb**
    * Login to Colab and load the `Multimodal-Sentiment-Analysis.ipynb` file

1.  **Download requirements.txt**
    * Download the `requirements.txt` file in your environment

3. **Enable the GPU usage:**
    * From the appropriate section. Now the notebook is ready to use.

## Project Explanation

The project uses the following steps:

1.  **Data Download and Preparation:**
    * Downloads the Twitter dataset from Kaggle.
    * Loads and preprocesses the text and image data.

2.  **Text Classification:**
    * Uses a pre-trained BERT-tiny model to extract text features.
    * Trains a text classifier to predict sentiment from the text.

3.  **Image Classification:**
    * Uses a pre-trained MobileNetV2 model to extract image features.
    * Trains an image classifier to predict sentiment from the images.

4.  **Multimodal Fusion:**
    * Combines the text and image features using a custom neural network.
    * Trains the multimodal model to predict sentiment using both text and image information.

5.  **Evaluation:**
    * Evaluates the model's performance on a test dataset.
    * Generates a confusion matrix and classification report.

## Model Details

* **Text Model:** BERT-tiny
* **Image Model:** MobileNetV2
* **Multimodal Model:** Custom neural network with attention mechanism

## Important Notes

* Make sure you have a stable internet connection for downloading the dataset and pre-trained models.
* The code is designed to run on a machine with a GPU for faster training.

## Contributing

If you'd like to contribute to this project, please feel free to submit a pull request.

## License

MIT License

Copyright (c) 2025 Luca Alfano

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
