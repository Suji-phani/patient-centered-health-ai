import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Map numeric labels to condition names (example)
label_map = {
    35: "Common Cold",
    46: "Anal Fissure",
    72: "Arthritis",
    78: "Astigmatism",
    86: "Atypical Hyperplasia",
    116: "Vertigo",
    143: "Metabolic Syndrome",
    147: "Broken Heart Syndrome",
    149: "Pneumonia",
    167: "Carbon Monoxide Poisoning",
    168: "Cellulitis",
    181: "Toxic Imbalance",
    186: "Neurogenic Infection",
    193: "Inflammatory Condition",
    234: "Autoimmune Disorder",
    235: "Concussion",
    262: "Chronic Bronchitis",
    275: "Renal Dysfunction",
    284: "Ovarian Insufficiency",
    308: "Migraine",
    345: "Endocrine Imbalance",
    437: "Ulcerative Colitis",
    456: "Pancreatitis",
    459: "Neuropathic Pain",
    471: "Hepatic Encephalopathy",
    500: "Hypoglycemia",
    505: "Cystic Fibrosis",
    515: "Liver Cirrhosis",
    539: "Hyperparathyroidism",
    541: "Acanthosis Nigricans",
    545: "Autoimmune Hepatitis",
    552: "Psoriatic Arthritis",
    553: "Lupus Erythematosus",
    554: "Ankylosing Spondylitis",
    556: "Thalassemia",
    558: "Eosinophilic Esophagitis",
    559: "Pernicious Anemia",
    561: "Cerebral Palsy",
    563: "Rheumatic Fever",
    564: "Multiple Myeloma",
    565: "Allergic Rhinitis",
    566: "Hodgkin Lymphoma",
    567: "Crohn’s Disease",
    570: "Systemic Sclerosis",
    572: "Graves’ Disease",
    573: "Sarcoidosis",
    574: "Raynaud’s Syndrome",
    575: "Churg-Strauss Syndrome",
    577: "Whipple’s Disease",
    578: "Pulmonary Fibrosis",
    579: "Septic Arthritis",
    580: "Bronchiectasis",
    582: "Polyarteritis Nodosa",
    585: "Achalasia",
    586: "Sjogren’s Syndrome",
    587: "Addison’s Disease",
    588: "Amyloidosis",
    591: "Behçet’s Disease",
    593: "Idiopathic Thrombocytopenic Purpura",
    596: "Flu-like Illness",
    608: "Hyperthyroidism",
    650: "Trigeminal Neuralgia",
    666: "Chronic Fatigue Syndrome",
    708: "Interstitial Cystitis",
    736: "Myasthenia Gravis",
    760: "Lichen Planus",
    766: "Chikungunya",
    785: "Parvovirus Infection",
    795: "Fibromyalgia",
    798: "Ovarian Insufficiency",
    1018: "Cushing Syndrome",
    1026: "Gastroenteritis",
}



# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Load model and vectorizer
model = joblib.load("nb_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit UI
st.title("🧠 Patient Symptom Classifier")
st.write("Enter patient symptoms below to predict the medical condition label.")

user_input = st.text_area("✏️ Enter symptom narrative here:", height=150)

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    
    label_name = label_map.get(prediction, "Unknown condition")  # <-- move it here
    
    st.success(f"✅ Predicted Condition: **{label_name}** (Label {prediction})")


