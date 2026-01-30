import streamlit as st
import time
import os
from langchain_community.document_loaders import PyMuPDFLoader
# On importe get_vector_store pour l'initialisation initiale
from functions_rag_streamlit import preprocessing_docs, chat, get_vector_store

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Congo Discover AI",
    page_icon="🇨🇬",
    layout="centered"
)

# --- INITIALISATION DU VECTOR STORE ---
# On s'assure que le store est chargé une seule fois et partagé
if "vector_store" not in st.session_state:
    with st.spinner("Chargement de la base de connaissances..."):
        st.session_state.vector_store = get_vector_store()

# --- STYLE CSS PERSONNALISÉ ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stChatMessage { border-radius: 15px; padding: 10px; margin-bottom: 10px; }
    h1 { color: #1e3d59; text-align: center; }
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
    uploaded_file = st.file_uploader("Ajouter un PDF sur le Congo", type="pdf")
    
    if uploaded_file:
        with st.status("Traitement du document...", expanded=True) as status:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyMuPDFLoader(temp_path)
            docs = loader.load()
            
            # MISE À JOUR : On met à jour le store dans le session_state
            st.session_state.vector_store = preprocessing_docs(docs)
            
            os.remove(temp_path)
            status.update(label="Document indexé avec succès !", state="complete", expanded=False)
            st.success(f"{uploaded_file.name} est prêt.")

    if st.button("Effacer l'historique"):
        st.session_state.messages = []
        st.rerun()

# --- INTERFACE PRINCIPALE ---
st.markdown("# 🌴 Congo Discover AI")
st.markdown("##### Posez vos questions sur l'histoire, la culture ou la géographie du Congo.")

# Affichage de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie
if prompt := st.chat_input("Que voulez-vous savoir sur le Congo ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Réflexion en cours..."):
            start_time = time.time()
            # On appelle la fonction chat normalement
            full_response = chat(prompt)
            end_time = time.time()
            
        message_placeholder.markdown(full_response)
        st.caption(f"Réponse générée en {end_time - start_time:.2f}s via Groq ⚡")

    st.session_state.messages.append({"role": "assistant", "content": full_response})