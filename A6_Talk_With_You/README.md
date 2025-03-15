# NLP Assignment A6: Let's Talk with Yourself

## Student Info
**Name**: Zwe Htet  
**Student ID**: st125338  

---

## Acknowledgments
This project was developed under the guidance of **Professor Chaklam** as part of the **Machine Learning** course. Special thanks to my classmates and seniors for their valuable insights and support.

---

## Retrieval-Augmented Chatbot
This repository contains the implementation of the **"A6: Let's Talk with Yourself"** assignment. The objective is to develop a chatbot that utilizes **Retrieval-Augmented Generation (RAG)** techniques to answer questions about personal information based on provided documents.

---

## Table of Contents
1. [Overview](#overview)  
2. [Installation](#installation)  
3. [Usage](#usage)  
4. [Data Sources](#data-sources)  
5. [Prompt Design](#prompt-design)  
6. [Retrieval and Generation Models](#retrieval-and-generation-models)  
7. [Chatbot Implementation](#chatbot-implementation)  
8. [Web Application](#web-application)  
9. [Results](#results)  
10. [Future Improvements](#future-improvements)  
11. [Conclusion](#conclusion)  

---

## Overview
The project involves:
1. **Data Processing**: Extracting and splitting personal information from structured documents.
2. **Prompt Engineering**: Designing an effective prompt for chatbot responses.
3. **Model Selection**: Comparing different retriever and generator models to find the optimal combination.
4. **Web Application**: Deploying the chatbot using Gradio.

---

## Installation
### Prerequisites
- Python 3.8 or later
- Recommended Python libraries:
  ```bash
  torch
  transformers
  langchain
  faiss-cpu
  gradio
  ```

### Setup Instructions
1. Clone this repository:
   ```bash
   git clone https://github.com/zwecrab/NPU/tree/main/A6_Talk_With_You
   cd A6_Talk_With_You
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Data Sources
The chatbot retrieves information from the following document:
- **A6_Bio.pdf** - A structured document containing personal details used for chatbot responses.

---

## Prompt Design
The chatbot is guided by the following prompt:
```text
Answer the following question very shortly(no more than 50 words) based solely on the provided context.
If the answer is not present in the context, say 'I'm sorry, but I don't have enough information to answer that.'

Context:
{context}

Question:
{input}

Answer:
```

---

## Retrieval and Generation Models
### Embedding Model (Retriever)
- **Model Used**: `sentence-transformers/all-MiniLM-L6-v2`
- **Reason**: Small, fast, and effective for retrieval tasks.

### Text-Generation Models Explored

| Model | Parameters | Performance | Total Runtime | Inference Time |
|--------|-------------|-------------|---------------|----------------|
| `Qwen/Qwen2.5-Coder-0.5B-Instruct` | 0.5B | Good speed, best for structured outputs | 21.35s        | 14.35s         |
| `microsoft/Phi-4-instruct` | 3.84B | High coherence, slightly slower | 119.60s       | 99.85s         |
| `mistralai/Mistral-7B-Instruct-v0.2` | 7B | More detailed responses but requires more memory | 7.71s         | 2.91s          |

#### Final Choice:
The **Qwen2.5-Coder model** was chosen due to its efficiency and speed while maintaining accurate responses within limited computational resources.

---

## Chatbot Implementation
The chatbot pipeline consists of:
1. **Document Loader**: Loads PDFs for retrieval.
2. **Text Splitter**: Splits documents into smaller chunks.
3. **FAISS Retriever**: Stores and retrieves document embeddings.
4. **LLM Generator**: Uses a fine-tuned LLM to generate responses.
5. **Web Interface**: A Gradio-based web UI for user interaction.

---

### Prerequisites
```bash
pip install torch transformers langchain faiss-cpu gradio
```

### Setup
```bash
git clone https://github.com/zwecrab/NPU/tree/main/A6_Talk_With_You
cd A6_Talk_With_You
```

### Running the Application
```bash
python app.py
```
Access at: `http://127.0.0.1:7860`

---

The UI includes:
- A text input for user queries.
- A response box for chatbot-generated answers.
- Predefined sample questions for easy testing.

---

## Results
- High retrieval accuracy using sentence-transformers/all-MiniLM-L6-v2.
- Fast response times with Qwen2.5-Coder.
- Minimal hallucination when responses were constrained to the document context.
---
## **Screenshots**

**Home Page**
![Web Application UI](Screenshots/home.png)

**Reply for input with information**
![With Information Result](Screenshots/with_info.png)

**Reply for input without information**
![No Infomation Result](Screenshots/no_info.png)

---

## Future Improvements
- Experiment with fine-tuned models to further improve accuracy.
- Deploy on Hugging Face Spaces for a live version.
- Enhance UI with more interactive chatbot elements.

---

## Conclusion
This project successfully implemented a RAG-powered chatbot capable of answering personal questions based on retrieved documents. By leveraging FAISS for retrieval and Qwen2.5-Coder for text generation, the chatbot demonstrates a structured approach to handling user queries efficiently.