# YouTube Sentiment Analysis with T5

This project implements a sentiment analysis model using the T5 transformer. The model is trained to analyze the sentiment of YouTube video URLs.

## Project Overview

The project involves:
- Collecting data
- Preprocessing the data
- Training the T5 model
- Evaluating the model's accuracy
- Creating a Gradio interface for easy use

## Installation

To install the required packages, run:

```bash
pip install transformers datasets evaluate gradio youtube-transcript-api pydub


## **Data Preparation**
Ensure you have a CSV file named dataset.csv with the following columns:

URL: The YouTube video URL
Label: The sentiment label (e.g., "positive", "negative")

## **Training the Model**
The following script trains the T5 model on your dataset and evaluates its accuracy:
