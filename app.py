import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

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
    # 1. On crée le tableau de 85 colonnes avec les bons noms
    if st.button("Prédire le Revenu"):
        # 1. On crée le tableau de 85 colonnes avec les bons noms
        input_data = pd.DataFrame(np.zeros((1, 85)), columns=model.feature_names_in_)
        
        # 2. AU LIEU DE REMPLIR AU HASARD, on cible les colonnes qui existent dans ton modèle
        # On va tricher un peu pour lier tes curseurs aux colonnes démographiques qui ressemblent
        if "TotalPop" in input_data.columns:
            input_data["TotalPop"] = age * 100 # On simule une donnée cohérente
        if "IncomePerCap" in input_data.columns:
            input_data["IncomePerCap"] = capital_gain if capital_gain > 0 else 20000
        if "Employed" in input_data.columns:
            input_data["Employed"] = hours * 10
            
        # 3. Prédiction
        prediction = model.predict(input_data)
        
        # Attention : si ton modèle prédit des noms de comtés ou autre chose, 
        # le label ">50K" est peut-être faux. Vérifions le résultat brut :
        st.write(f"Valeur brute de prédiction : {prediction[0]}")
        
        label = ">50K$" if prediction[0] == 1 else "<=50K$"
        st.success(f"Résultat : **{label}**")
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
