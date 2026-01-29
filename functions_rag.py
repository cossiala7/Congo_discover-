from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI





# Au démarrage de l'application
pdf_file_path = "REPUBLIQUE_DU_CONGO.pdf"
loader = PyMuPDFLoader(pdf_file_path)
documents = loader.load()

vector_store = None
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=API_KEY)
llm_model = "gemini-2.5-flash"
chat_model = ChatGoogleGenerativeAI(model=llm_model, api_key=API_KEY)



def preprocessing_docs(doc):
    global vector_store
    # Découper le texte en petits morceaux
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50,separators=["\n\n", "\n", ".", " "])
    chunks = text_splitter.split_documents(doc)
    # 2. Ajout au vector_store existant
    if vector_store is None:
        # PREMIER CHARGEMENT : On crée le store
        vector_store = FAISS.from_documents(chunks, embedding_model)
    else:
        # AJOUTS SUIVANTS : On ajoute simplement les nouveaux vecteurs
        vector_store.add_documents(chunks)
        
    
    return vector_store

vector_store  = preprocessing_docs(documents)

def chat(query):
    query = query
    retrieved_docs = vector_store.similarity_search_with_relevance_scores(query, k=5)
   
    # Créer le prompt avec le contexte récupéré
    context = "\n\n".join([doc[0].page_content for doc in retrieved_docs])
    prompt = f""" ### INSTRUCTIONS ###
                    1. Analyse le DOCUMENT ci-dessous pour répondre à la QUESTION.
                    2. Si la question est "Qui es tu (Majuscule ou minuscule)" répond : "Je suis un chatbot qui t'aide à en savoir plus sur le Congo-Brazzavile"
                    3. Si la question est "Que sais tu faire (Majuscule ou minuscule)" répond : "Je suis conçu pour répondre aux questions uniquement liée au Congo-Brazzaville"
                    4. Si la réponse n'est pas présente dans le texte ou dans le document, réponds exactement : "Je ne sais pas, désolé", rien d'autre, j'insite dessus.
                    5. Ne cite jamais des informations issues de tes connaissances générales et de ce qui n'est pas dans le document.
                    6. Réponds de manière concise, structurée et brève.

                    ### DOCUMENT ###
                    {context}

                    ### QUESTION ###
                    {query}

                    ### RÉPONSE ###
                    """
    messages = [
        SystemMessage(content="""Tu es un analyste expert en extraction d'informations. 
                                Ton unique source de vérité est le document fourni. 
                                Tu dois être précis et ne jamais inventer d'informations en dehors du contexte.
                                """),
        HumanMessage(content=prompt)
    ]

    # Générer la réponse
    response = chat_model.invoke(messages)
    
    return response.content

