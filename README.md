#  Patient-Centered Health AI

A machine learning-powered web app that predicts medical conditions based on user-inputted symptom descriptions. This project applies NLP and supervised learning to provide quick, patient-friendly condition suggestions.

##  Project Overview

This Streamlit-based MVP interface allows patients to enter symptom narratives (e.g., _"I’m feeling dizzy with a sore throat and chest pain"_) and receive a predicted disease name as output (e.g., _“Pneumonia”_).

##  Technologies Used

- Python (NLP & ML) – Text preprocessing, TF-IDF vectorization
- Scikit-learn – Naive Bayes classifier
- LDA (Latent Dirichlet Allocation) – Topic modeling (for analysis)
- Streamlit – Interactive web-based UI
- Pandas, NLTK – Data processing and stopword filtering

##  Project Structure
├── patient_ai_mvp.py # Streamlit frontend interface
├── nb_model.pkl # Trained Naive Bayes model (excluded from repo)
├── tfidf_vectorizer.pkl # TF-IDF vectorizer (excluded from repo)
├── original_symptom_dataset.xlsx # Cleaned dataset
└── README.md # Project documentation


##  Key Features

-  Predicts diseases from free-text symptom descriptions
-  Labeled over 1000 medical conditions
-  LDA topic visualization for symptom clusters
-  Designed for accessibility and usability by non-technical users

##  How It Works

1. User inputs symptom text
2. Text is cleaned, tokenized, and vectorized
3. Naive Bayes model predicts the disease label
4. Label is mapped to a human-readable disease name

##  Model Files

To keep the repo lightweight, the following file is excluded:
- `nb_model.pkl`


 These can be shared upon request or deployed alongside the app in a production setting.

##  How to Run Locally

1. Clone the repo
2. Install required packages:  
   ```bash
   pip install streamlit scikit-learn pandas nltk





