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

## Data Preparation

Ensure you have a CSV file named dataset.csv with the following columns:

URL: The YouTube video URL
Label: The sentiment label (e.g., "positive", "negative")

## Training the Model

The following script trains the T5 model on your dataset and evaluates its accuracy:

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
import gradio as gr

# Load dataset
dataset = load_dataset('csv', data_files='dataset.csv')
tokenizer = T5Tokenizer.from_pretrained("t5-small")

def preprocess_function(examples):
    inputs = examples['URL']
    targets = examples['Label']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

train_dataset = dataset['train'].map(preprocess_function, batched=True)

model = T5ForConditionalGeneration.from_pretrained("t5-small")

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.replace('▁', ' ') for pred in decoded_preds]
    decoded_labels = [label.replace('▁', ' ') for label in decoded_labels]

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    result = accuracy_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return result

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

results = trainer.evaluate()
accuracy = results['eval_accuracy'] * 100
print(f"Model Accuracy: {accuracy:.2f}%")


## Gradio Interface

To create a Gradio interface for the sentiment analysis model, use the following code:

def sentiment_analysis(url):
    inputs = tokenizer(url, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    outputs = model.generate(**inputs)
    sentiment = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sentiment

iface = gr.Interface(
    fn=sentiment_analysis,
    inputs="text",
    outputs="text",
    title="YouTube Video Sentiment Analysis",
    description="Enter a YouTube video URL to get the sentiment analysis result."
)

iface.launch()

## **License

This project is licensed under the MIT License - see the LICENSE file for details.
