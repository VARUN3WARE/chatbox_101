# Chatbot with Intent Recognition

This is a chatbot project that uses Natural Language Processing (NLP) to identify user intents and provide relevant responses. The bot is built using TensorFlow and Keras for intent classification and trained on a custom dataset of user intents.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model](#model)
- [License](#license)

## Project Overview
The chatbot is capable of understanding user input by classifying it into predefined intents and generating appropriate responses based on the intent. The model uses a simple neural network trained with Keras, and the input is processed using a Tokenizer and Label Encoder. The bot runs interactively in the terminal, where users can chat with it, and it will respond based on the detected intent.

## Features
- Intent recognition using a trained neural network.
- Provides relevant responses to user queries based on classified intent.
- Simple and interactive terminal-based chatbot interface.
- Uses TensorFlow and Keras for model building.
- Customizable intents and responses via `intents.json`.

## Tech Stack
- Python
- TensorFlow and Keras
- NumPy
- Scikit-learn (for Label Encoding)
- Colorama (for terminal text coloring)

## Installation

### Prerequisites
Make sure you have the following installed:
- Python 3.x
- TensorFlow (`pip install tensorflow`)
- Scikit-learn (`pip install scikit-learn`)
- Colorama (`pip install colorama`)

### Preparing the Dataset
1. Place your `intents.json` file in the root directory (or modify the path in the script).
2. The `intents.json` file should have the following structure:
   ```json
   {
       "intents": [
           {
               "tag": "greeting",
               "patterns": ["Hi", "Hello", "How are you?"],
               "responses": ["Hello!", "Hi, how can I help you?"]
           },
           {
               "tag": "goodbye",
               "patterns": ["Bye", "See you later"],
               "responses": ["Goodbye!", "Have a nice day!"]
           }
       ]
   }
   ```

### Training the Model
1. Before running the chatbot, train the model by executing the training script (ensure your `intents.json` is ready):
   ```bash
   python model_train.py
   ```

2. The trained model will be saved as `chat_model.keras`, and the tokenizer and label encoder will be saved as `tokenizer.pickle` and `label_encoder.pickle`.

### Running the Chatbot
Once the model is trained and saved, you can interact with the chatbot by running:
```bash
python main.py
```
Start chatting with the bot! Type `quit` to end the conversation.

## Usage
- Run the bot and interact through the terminal.
- The bot recognizes intents from the training dataset (`intents.json`) and responds accordingly.

## Dataset
The `intents.json` file contains the labeled data used to train the model. Each intent consists of:
- **Tag**: The label for the intent.
- **Patterns**: The user inputs associated with the intent.
- **Responses**: The possible responses the bot can give.

You can modify the `intents.json` file to customize the chatbot's intents and responses.

## Model
The model is a simple neural network that uses an Embedding layer, a GlobalAveragePooling layer, and Dense layers to classify the user's input into one of the predefined intents. It is trained using the `sparse_categorical_crossentropy` loss function and `adam` optimizer.

### Model Architecture
- **Embedding Layer**: Turns words into dense vectors of fixed size.
- **GlobalAveragePooling1D**: Averages the embeddings over the input sequence.
- **Dense Layers**: Two fully connected layers with ReLU activation.
- **Output Layer**: Softmax activation to output a probability distribution over the possible intents.
