import os
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# --- CONFIGURATION DES CLÉS (Streamlit Cloud) ---
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Dossiers pour les données
DOCS_FOLDER = "data_congo"
VECTOR_DB_PATH = "faiss_index_congo"

# --- INITIALISATION DES MODÈLES ---
# Gemini pour les vecteurs (Embeddings)
embedding_model = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004", 
    version="v1",
    google_api_key=GOOGLE_API_KEY
)

# Groq pour le cerveau (LLM) - Llama 3.3 70B est excellent pour le RAG
chat_model = ChatGroq(
    temperature=0.1, 
    groq_api_key=GROQ_API_KEY, 
    model_name="llama-3.3-70b-versatile"
)
def preprocessing_docs(new_docs):
    """Permet d'ajouter des documents uploadés manuellement au store existant."""
    global vector_store
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_documents(new_docs)
    
    if vector_store is None:
        vector_store = FAISS.from_documents(chunks, embedding_model)
    else:
        vector_store.add_documents(chunks)
    
    # On sauvegarde la mise à jour localement
    vector_store.save_local(VECTOR_DB_PATH)
    return vector_store

def load_and_preprocess():
    """Charge tous les PDF, les découpe et crée/met à jour le vector store."""
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
    
    all_documents = []
    
    # 1. Parcourir le dossier pour trouver les PDF
    pdf_files = [f for f in os.listdir(DOCS_FOLDER) if f.endswith('.pdf')]
    
    if not pdf_files:
        return None

    for file in pdf_files:
        loader = PyMuPDFLoader(os.path.join(DOCS_FOLDER, file))
        all_documents.extend(loader.load())

    # 2. Découpage intelligent
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_documents(all_documents)

    # 3. Création ou Chargement du Vector Store
    if os.path.exists(VECTOR_DB_PATH):
        # On charge l'existant et on ajoute les nouveaux (si nécessaire)
        # Note: Pour simplifier ici, on recrée s'il y a du changement
        vector_store = FAISS.from_documents(chunks, embedding_model)
    else:
        vector_store = FAISS.from_documents(chunks, embedding_model)
    
    # 4. Sauvegarde locale pour éviter de recalculer au prochain lancement
    vector_store.save_local(VECTOR_DB_PATH)
    return vector_store

def get_vector_store():
    """Récupère le store existant ou le crée s'il n'existe pas."""
    if os.path.exists(VECTOR_DB_PATH):
        return FAISS.load_local(VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True)
    else:
        return load_and_preprocess()

# Initialisation globale au chargement du module
vector_store = get_vector_store()

def chat(query):
    """Fonction principale de réponse."""
    global vector_store
    
    # Si le store est vide
    if vector_store is None:
        return "Désolé, je n'ai aucun document en mémoire pour répondre."

    # Recherche par similarité (Top 5 chunks)
    retrieved_docs = vector_store.similarity_search_with_relevance_scores(query, k=5)
    
    # Filtrage par score (optionnel mais recommandé pour éviter les hors-sujets)
    context_text = "\n\n".join([doc[0].page_content for doc in retrieved_docs if doc[1] > 0.3])

    if not context_text:
        return "Je ne sais pas, désolé (aucune information pertinente trouvée dans les documents)."

    prompt = f"""### RÔLE ###
Tu es un analyste expert du Congo-Brazzaville. Réponds uniquement en utilisant le contexte fourni.

### CONTEXTE ###
{context_text}

### QUESTION ###
{query}


### DIRECTIVES DE RÉPONSE ###
1. Utilise les extraits ci-dessus pour répondre de façon claire et concise.
2. Si la réponse n'est pas explicitement écrite mais peut être déduite logiquement des extraits, réponds avec nuance.
3. Si le sujet n'a absolument aucun rapport avec le Congo-Brazzaville ou les extraits, dis simplement : "Désolé, je n'ai pas d'informations précises sur ce sujet."
4. Structure ta réponse si nécessaire pour être bref.
5. Ne commence jamais par "D'après les documents..." ou "Le contexte dit...". Entre directement dans le vif du sujet.
6. Si on te pose la question "que sais tu faire ?" répond que tu sais répondre à différentes questions en rapport avec la République du Congo
"""

    messages = [
        SystemMessage(content="""Tu es 'Congo Discover AI', un expert chaleureux et précis sur le Congo-Brazzaville.
                                    Ton but : extraire la réponse la plus pertinente du contexte fourni.
                                    Style : Direct, sans fioritures, bref.
                                    Interdiction : Ne pas inventer de faits historiques ou chiffres s'ils ne sont pas suggérés par les documents."""),
        HumanMessage(content=prompt)
    ]

    response = chat_model.invoke(messages)
    return response.content





