# 🛡️ SpamShield — Spam Email Classifier

A machine learning project that classifies email text as **Spam** or **Not Spam** using Natural Language Processing (NLP) techniques and Scikit-learn.

SpamShield uses TF-IDF for text feature extraction and Logistic Regression for classification, with an interactive web interface built using Streamlit.

## 🚀 Features

- 📧 Classifies email text as Spam or Not Spam
- 🔤 Converts text into numerical features using TF-IDF
- 🤖 Uses Logistic Regression for spam classification
- 📊 Evaluates performance using accuracy, precision, recall, F1-score, and a confusion matrix
- 🔎 Displays the predicted class and estimated spam probability
- 🌐 Provides an interactive Streamlit web interface
- 💾 Saves the trained ML pipeline using Joblib for reuse during predictions

## 🛠️ Tech Stack

- **Language:** Python
- **Data Processing:** Pandas
- **Machine Learning:** Scikit-learn
- **Feature Extraction:** TF-IDF
- **Model:** Logistic Regression
- **Web Interface:** Streamlit
- **Model Serialization:** Joblib

## 📂 Project Structure

```text
Spam-Email/
│
├── Data/
│   └── your_filename.csv
│
├── models/
│   └── spam_classifier.pkl
│
├── train.py
├── predict.py
├── app.py
├── requirements.txt
└── README.md
```

> Replace `your_filename.csv` with your actual dataset filename.

## ⚙️ How It Works

1. **Data preprocessing:** Removes duplicate records and prepares email text and labels.
2. **Train-test split:** Splits the dataset into training and testing sets while preserving the class distribution.
3. **Text vectorization:** Uses TF-IDF to convert email text into numerical features.
4. **Model training:** Trains a Logistic Regression classifier on the extracted features.
5. **Model evaluation:** Evaluates the model on unseen test data using classification metrics.
6. **Prediction:** Loads the saved pipeline and classifies new email text.
7. **Web interface:** Lets users paste email text and view the prediction through Streamlit.

## 📊 Model Performance

The Logistic Regression model achieved **98.68% test accuracy** in the initial experiment on the selected dataset.

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

*Note: Results depend on the dataset and train-test split. Test accuracy does not guarantee the same performance on real-world emails.*

## 💻 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/IshaanSingla404/Spam-Email.git
cd Spam-Email
```


### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Place your CSV file inside the `Data/` directory. The dataset must contain these columns:

| Column | Description |
|---|---|
| `text` | Email text |
| `spam` | Target label: `0` = Not Spam, `1` = Spam |

Make sure `DATA_PATH` in `train.py` points to your actual CSV filename.

### 5. Train the model

```bash
python train.py
```

This trains the model, prints evaluation metrics, and saves the pipeline to `models/spam_classifier.pkl`.

### 6. Run the command-line predictor

```bash
python predict.py
```

Enter email text when prompted. Type `exit` to quit.

### 7. Launch the web application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to use SpamShield.

> The saved model file must exist before launching the app. Run `train.py` first if it is missing.

## 🧠 What I Learned

- Text preprocessing and feature extraction using TF-IDF
- Training and comparing machine learning classification models
- Evaluating classification performance beyond accuracy
- Building reusable Scikit-learn pipelines
- Saving and loading trained models with Joblib
- Integrating an ML model into a Streamlit application
- Structuring a machine learning project for GitHub

## 🔮 Future Improvements

- Test the model on more diverse and realistic email datasets
- Improve preprocessing and experiment with additional models
- Add a confidence threshold for uncertain predictions
- Deploy the application so it can be accessed online
- Improve the interface and add more detailed prediction insights

## ⚠️ Disclaimer

SpamShield is an educational machine learning project. Predictions are estimates and may be incorrect. Do not rely on this model alone for security-critical email filtering.

## 👨‍💻 Author

**KEERTHIVASAN V**

---

