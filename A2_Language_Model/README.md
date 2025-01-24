# A2: Language Model

This repository contains the implementation of a language model using an LSTM network. The project is divided into three tasks: dataset acquisition, model training, and web application development. Below is a detailed explanation of each task and the project as a whole.

---

## Task 1: Dataset Acquisition

### Dataset Description
The dataset used in this project is the [Harry Potter Novels Dataset](https://github.com/Ginga1402/Harry-Potter-Dataset). It includes the complete text of the Harry Potter novels by J.K. Rowling and is suitable for training a language model due to its rich textual content.

> **Note:** If you chose a different dataset, replace this description with your dataset details and provide the source link.

### Dataset Credit
The dataset was sourced from [Ginga1402 on GitHub](https://github.com/Ginga1402/Harry-Potter-Dataset). Proper acknowledgment is given in compliance with the dataset usage guidelines.

---

## Task 2: Model Training

### Preprocessing Steps
1. **Text Cleaning**: The dataset was preprocessed to remove unnecessary characters, whitespace, and special symbols.
2. **Tokenization**: The text was split into tokens (words or characters) to facilitate the training process.
3. **Vocabulary Creation**: A vocabulary of unique tokens was built, and tokens were mapped to integer indices.
4. **Sequence Generation**: Fixed-length sequences were created for input to the LSTM model.

> **Flag:** Add any specific preprocessing steps you used or modify the above based on your implementation.

### Model Architecture
The language model was built using a Long Short-Term Memory (LSTM) network. The architecture includes:
- An embedding layer to convert tokens into dense vector representations.
- LSTM layers to capture context and dependencies in the text.
- A fully connected output layer with a softmax activation for predicting the next token.

> **Flag:** Add details on the number of layers, hidden units, or any specific hyperparameters you used.

### Training Process
The model was trained using the following steps:
1. The dataset was split into training and validation sets.
2. The training loop minimized the cross-entropy loss using an optimizer (e.g., Adam).
3. The model checkpoint with the best validation loss was saved as `st125338_best-val-lstm_lm.pt`.

---

## Task 3: Text Generation - Web Application Development

### Web Application
A simple Flask web application was developed to demonstrate the capabilities of the trained language model.

#### Features
1. **Input Box**: Users can input up to 5 words as a text prompt.
2. **Text Generation**: The model generates a continuation of the input text based on the context and style of the training dataset.
3. **Temperature Control**: For best results, the temperature for text generation is set to 1.0.

#### How It Works
1. The user provides a text prompt in the input box.
2. The application interfaces with the language model to generate the next sequence of text.
3. The generated text is displayed on the web page.

### Documentation
- The Flask app loads the trained model (`st125338_best-val-lstm_lm.pt`) into GPU memory (if available) for inference.
- The `generate_text` method of the model is used to create text continuations.
- Input validation ensures the prompt is limited to 5 words.

> **Flag:** Add any additional notes or custom features you implemented in the app.

---

## Acknowledgments
I would like to express my gratitude to:
- **Professor** for his guidance and support of this project.
- **Friends and seniors** for their valuable insights, assistance and encouragement.
- **Ginga1402** for providing the [Harry Potter Novels Dataset](https://github.com/Ginga1402/Harry-Potter-Dataset), which formed the basis of this project.

---

## Instructions to Run
1. Clone this repository.
2. Install the required Python packages: `pip install -r requirements.txt`.
3. Run the Flask app: `python app.py`.
4. Access the application at `http://127.0.0.1:5000`.
