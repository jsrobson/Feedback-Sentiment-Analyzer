# Feedback Sentiment Analyzer

![Python](https://img.shields.io/badge/python-3.11-blue?logo=python)
![License](https://img.shields.io/badge/license-MIT-green)

A Python application for analyzing user feedback using **sentiment analysis** and **topic modeling**. The tool clusters text data, identifies topics and subtopics, performs sentiment analysis, and generates **human-readable summaries** using a large language model (LLM). Includes a **Tkinter-based UI** for easy interaction.

Motivated by every public engagement dataset that requires labourious hand-theming of each provided comment, collection into topics and sub-topics, identification of relevant verbatim comments, and downstream distillation of topics and sub-topics into summary text. This application automates this process using deterministic topic modelling (BERTopic) and text summaization using open-source large language modelling (Gemma 3).

**Note:** The Gemma 3 LLM used for summarization can be slow, especially for large datasets. Please allow extra time for processing when running the analysis.

---

## Features

- **CSV Input**: Load user feedback from CSV files.
- **Sentiment Analysis**: Classifies feedback as positive, negative, or neutral.
- **Topic Clustering**: Groups feedback into topics and subtopics using BERTopic.
- **Summarization**: Generates concise summaries for each topic and subtopic via a large language model.
- **GUI Interface**: User-friendly Tkinter interface for loading data, selecting columns, running analysis, and saving results.

---

## Installation
1. Clone the repository
```bash
git clone https://github.com/jsrobson/Feedback-Sentiment-Analyzer.git

cd Feedback-Sentiment-Analyzer
```
2. Create a virtual environment (recommended)
```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Set a [HuggingFace API](https://huggingface.co/docs/hub/en/security-tokens) key in an .env file:
```bash
HF_API_KEY=your_huggingface_api_key
```

---

## Usage
```bash
python app.py
```
1. Load a CSV containing user feedback.
2. Select the column containing feedback text within the CSV.
3. (Optional) Load a CSV of seed topics (single column) to guide clustering.
4. Specify the save location for the output CSV.
5. Click RUN to process the data. Progress will be shown in a popup window.
6. Click RESET to clear the selections and start over.

The output CSV will contain:
- General topic
- Subtopic
- Sentiment alignment
- Number of responses
- Summary of subtopic area

---

## Example Output

Below is an example of the generated subtopic output using a test file (100 rows) of product sentiment for the Apple Magic Mouse. The produced CSV contains four subtopics in the same format.

**General Topic:** Comfortable Responsive Apple Mouse

**Subtopic:** Comfortable Responsive Cursor Control

**Sentiment:** Very Positive

**Number of Responses:** 51

**Summary:** The feedback centers around the Apple Magic Mouse, a wireless mouse designed for use with Apple devices.
Users consistently praise its comfortable design, responsive touch surface, and seamless connectivity with both Macs and iPads,
highlighting features like intuitive gestures and long battery life. While some minor concerns were raised regarding
the charging port placement and price, the overall sentiment is overwhelmingly positive, with many recommending the mouse
as a valuable and stylish accessory for enhancing the Apple user experience.


---

## Project Structure
```bash
Feedback-Sentiment-Analyzer/
├── app.py                 # Entry point to launch the GUI
├── data/                  # Sample data and test CSVs
├── processor/             # Topic and subtopic logic, parser
├── tests/                 # Unit tests for processing modules
├── user_interface/        # Tkinter UI components
├── utils/                 # Helper modules (Sentiment, Summary, Cluster, CSVLoader)
├── requirements.txt       # Python dependencies
```

---

## Dependencies
Key libraries are:
- **Pandas**: Data manipulation
- **tkinter**: Simple graphical user interface
- **BERTopic**: Topic modelling
- **transformers**: Sentiment and summarization models
- **torch**: Backend for large language model invocation
- **sentence-transformers**: Embeddings for BERTopic
- **umap-learn, hdbscan, scikit-learn**: Clustering and dimensionality reduction

Full list available in **requirements.txt**.

---

## Testing
Run the unit tests:
```bash
pytest tests\
```
Note: UI is not covered by automated tests; these focus on processing logic.

---

## Learning
Building this project provided hands-on experience across the full lifecycle of a data-driven application, from raw input to user-facing output. Key learnings include:

- **End-to-end Natural Language Processing pipelines**  
  Designing and integrating a workflow that combines sentiment analysis, topic modeling, and LLM-based summarization, while managing data flow between each stage.

- **Practical use of transformer models**  
  Working with HuggingFace pipelines and pretrained models for multilingual sentiment analysis, including handling empty inputs and performance considerations.

- **Topic modeling with BERTopic**  
  Applying clustering techniques (UMAP + HDBSCAN) and class-based TF-IDF to extract interpretable topics and subtopics from unstructured text.

- **Separation of concerns in application design**  
  Structuring the project to clearly separate UI logic, processing logic, and utility components, making the codebase easier to test, extend, and maintain.

- **Testing ML-adjacent code**  
  Writing unit tests for non-deterministic and external dependencies by isolating logic and using mocking to avoid expensive model initialization during tests.

- **Building a desktop UI for data workflows**  
  Creating a Tkinter-based interface that supports file selection, progress feedback, background threading, and safe interaction with long-running tasks.
---

## Licence
[MIT Licence](https://choosealicense.com/licenses/mit/)
