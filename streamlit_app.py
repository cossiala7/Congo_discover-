import streamlit as st
import time
import os
from langchain_community.document_loaders import PyMuPDFLoader
from functions_rag import preprocessing_docs, chat

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Congo Discover AI",
    page_icon="🇨🇬",
    layout="centered"
)

# --- STYLE CSS PERSONNALISÉ ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7d32, #fbc02d, #d32f2f);
    }
    h1 {
        color: #1e3d59;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALISATION DE L'ÉTAT DU CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- BARRE LATÉRALE (SIDEBAR) ---
with st.sidebar:
    st.title("🇨🇬 Congo Discover")
    st.markdown("---")
    st.info("Ce chatbot est expert sur la République du Congo.")
    
    st.subheader("📁 Documents")
    uploaded_file = st.file_uploader("Ajouter un document PDF relatif à la république du Congo", type="pdf")
    
    if uploaded_file:
        with st.status("Traitement du document...", expanded=True) as status:
            # Sauvegarde temporaire
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Chargement et Preprocessing
            loader = PyMuPDFLoader(temp_path)
            docs = loader.load()
            preprocessing_docs(docs)
            
            os.remove(temp_path)
            status.update(label="Document indexé avec succès !", state="complete", expanded=False)
            st.success(f"{uploaded_file.name} est prêt.")

    if st.button("Effacer l'historique"):
        st.session_state.messages = []
        st.rerun()

# --- INTERFACE PRINCIPALE ---
st.markdown("# 🌴 Congo Discover AI")
st.markdown("##### Posez vos questions sur l'histoire, la culture ou la géographie du Congo.")

# Affichage des messages de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie du chat
if prompt := st.chat_input("Que voulez-vous savoir sur le Congo ?"):
    # Afficher le message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Générer la réponse
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Réflexion en cours..."):
            start_time = time.time()
            full_response = chat(prompt)
            end_time = time.time()
            
        # Petit effet de texte qui s'affiche progressivement (optionnel)
        message_placeholder.markdown(full_response)
        
        # Afficher le temps d'exécution en petit
        st.caption(f"Réponse générée en {end_time - start_time:.2f}s")

    # Ajouter à l'historique
    st.session_state.messages.append({"role": "assistant", "content": full_response})