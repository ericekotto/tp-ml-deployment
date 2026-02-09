# Version forcing 1.0 - Refreshing environment
import streamlit as st
import pandas as pd
import joblib
import os
import sys
import numpy as np

st.sidebar.write(f"🐍 Python : {sys.version.split()[0]}")
st.sidebar.write(f"📊 Scikit-Learn : {sklearn.__version__}")

# Configuration de la page
st.set_page_config(page_title="Dashboard Multi-Projets ML", layout="wide", page_icon="📊")

# --- CONFIGURATION DES CHEMINS ---
# Dossier contenant les modèles
MODEL_DIR = "models"
# Dossier contenant les données (pour rappel ou affichage optionnel)
DATA_DIR = "data"

# Fonction pour charger les modèles avec gestion d'erreur
def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.error(f"⚠️ Erreur : Le fichier {path} est introuvable.")
        return None

# --- NAVIGATION ---
st.sidebar.title("📌 Menu Principal")
projet = st.sidebar.radio("Sélectionnez un projet :", 
    ["Accueil", "1. Census (Revenus)", "2. Auto-MPG (Consommation)", "3. Bank Marketing (Souscription)"])

# --- PAGE D'ACCUEIL ---
if projet == "Accueil":
    st.title("🚀 Interface de Déploiement Machine Learning")
    st.write("Bienvenue dans votre application. Cette plateforme permet de tester vos 3 modèles entraînés.")
    st.info(f"📁 Modèles chargés depuis : `/{MODEL_DIR}`\n\n📁 Données sources situées dans : `/{DATA_DIR}`")

# --- PROJET 1 : CENSUS ---
elif projet == "1. Census (Revenus)":
    st.header("📈 Prédiction des Tranches de Revenus (Census)")
    model = load_model("census.pkl")
    
    if model:
        st.subheader("Paramètres d'entrée")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Âge", 17, 90, 30)
            hours = st.slider("Heures travaillées par semaine", 1, 99, 40)
        with col2:
            edu_num = st.number_input("Années d'éducation", 1, 16, 10)
            capital_gain = st.number_input("Gain en capital", 0, 100000, 0)

        if st.button("Prédire le Revenu"):
            # Simulation du vecteur d'entrée selon votre entraînement
            input_data = np.array([[age, edu_num, capital_gain, hours]])
            prediction = model.predict(input_data)
            label = ">50K$" if prediction[0] == 1 else "<=50K$"
            st.success(f"Résultat de la prédiction : **{label}**")

# --- PROJET 2 : AUTO-MPG ---
elif projet == "2. Auto-MPG (Consommation)":
    st.header("🚗 Estimation de la Consommation (Auto-MPG)")
    model = load_model("auto-mpg.pkl")
    
    if model:
        st.subheader("Caractéristiques du véhicule")
        c1, c2, c3 = st.columns(3)
        with c1:
            cylinders = st.selectbox("Cylindres", [4, 6, 8])
            displacement = st.number_input("Cylindrée (Displacement)", 50.0, 500.0, 150.0)
        with c2:
            hp = st.number_input("Chevaux (Horsepower)", 40, 250, 100)
            weight = st.number_input("Poids (lbs)", 1500, 5000, 3000)
        with c3:
            accel = st.number_input("Accélération", 8.0, 25.0, 15.0)
            year = st.slider("Année du modèle (70-82)", 70, 82, 76)

        if st.button("Calculer MPG"):
            input_data = np.array([[cylinders, displacement, hp, weight, accel, year]])
            prediction = model.predict(input_data)
            st.warning(f"Consommation estimée : **{prediction[0]:.2f} MPG**")

# --- PROJET 3 : BANK MARKETING ---
elif projet == "3. Bank Marketing (Souscription)":
    st.header("🏦 Marketing Bancaire (Bank-Full)")
    model = load_model("bank_marketing.pkl")
    
    if model:
        st.subheader("Profil du Client")
        colA, colB = st.columns(2)
        with colA:
            age = st.number_input("Âge du client", 18, 100, 35)
            balance = st.number_input("Solde du compte (Balance)", -3000, 100000, 1000)
            duration = st.number_input("Durée du dernier contact (sec)", 0, 5000, 180)
        with colB:
            housing = st.selectbox("Prêt immobilier ?", ["Oui", "Non"])
            loan = st.selectbox("Prêt personnel ?", ["Oui", "Non"])
            h_val = 1 if housing == "Oui" else 0
            l_val = 1 if loan == "Oui" else 0

        if st.button("Prédire la Souscription"):
            # Votre modèle RandomForest attend 16 colonnes
            # On remplit les colonnes connues et on met 0 pour les autres (débrouillardise)
            full_input = np.zeros((1, 16))
            full_input[0, 0] = age
            full_input[0, 5] = balance
            full_input[0, 6] = h_val
            full_input[0, 7] = l_val
            full_input[0, 11] = duration
            
            prediction = model.predict(full_input)
            if prediction[0] == 1:
                st.success("✅ Résultat : Le client va SOUSCRIRE au dépôt à terme.")
            else:
                st.error("❌ Résultat : Le client ne va PAS souscrire.")
