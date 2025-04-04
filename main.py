import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Query, Cookie, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from uvicorn import run
import requests
import tempfile
from io import BytesIO
import uuid
from typing import Dict, List, Optional
import time

# Importa componenti LangChain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Configurazione Iniziale ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# per eseguire app uvicorn main:app --reload

# Carica variabili d'ambiente (API Key)
load_dotenv()

# Prendi l'API key da Railway o dal file .env
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logging.error("GOOGLE_API_KEY non trovata nelle variabili d'ambiente.")
    raise ValueError("GOOGLE_API_KEY non configurata. Configurala su Railway o nel file .env")

# Configura l'API di Google Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# --- Costanti Configurabili ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
FAISS_INDEX_PATH = "faiss_index_gemini"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
GENERATION_MODEL_NAME = "gemini-1.5-flash-latest"

# --- Variabili Globali (saranno inizializzate all'avvio) ---
vector_store = None
qa_chain = None

# Gestione delle sessioni di conversazione
conversation_history: Dict[str, List[Dict[str, str]]] = {}
# Tempo di scadenza delle sessioni in secondi (3 ore)
SESSION_EXPIRY = 10800 

# Percorso della cartella PDF (configurabile tramite variabile d'ambiente)
PDF_FOLDER_PATH = os.getenv("PDF_FOLDER_PATH", "./documenti_pdf")

def download_pdf(url):
    """Scarica il PDF dall'URL e lo salva in un file temporaneo"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Crea un file temporaneo
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.write(response.content)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        logging.error(f"Errore durante il download del PDF: {e}")
        raise

# --- Funzioni di Setup ---

def setup_rag_pipeline():
    """
    Carica i documenti, crea gli embeddings, costruisce il vector store
    e inizializza la catena QA (Question Answering).
    Questa funzione viene eseguita all'avvio dell'applicazione.
    """
    global vector_store, qa_chain
    logging.info("Inizio setup pipeline RAG...")

    # 1. Caricamento e Divisione dei Documenti PDF
    if not os.path.exists(PDF_FOLDER_PATH):
         logging.error(f"La cartella specificata per i PDF non esiste: {PDF_FOLDER_PATH}")
         logging.error("Assicurati di creare la cartella e metterci dentro i file PDF.")
         raise FileNotFoundError(f"Cartella PDF non trovata: {PDF_FOLDER_PATH}")

    logging.info(f"Caricamento PDF dalla cartella: {PDF_FOLDER_PATH}")
    loader = PyPDFDirectoryLoader(PDF_FOLDER_PATH)
    try:
        documents = loader.load()
        if not documents:
            logging.warning(f"Nessun documento PDF trovato o caricato da {PDF_FOLDER_PATH}. La RAG non avrà contesto.")
        else:
             logging.info(f"Caricati {len(documents)} documenti PDF.")

    except Exception as e:
        logging.error(f"Errore durante il caricamento dei PDF: {e}")
        raise

    # Se non ci sono documenti, non possiamo creare embeddings o vector store
    if not documents:
         logging.warning("Nessun documento da processare. La pipeline RAG sarà limitata.")
         # In questo caso potremmo impostare vector_store e qa_chain a None o gestire diversamente
         # Per ora, procediamo assumendo che potrebbero esserci documenti, ma se non ci sono, le fasi successive falliranno o saranno vuote.
         # Questo blocco previene errori se la cartella è vuota.
         return # Interrompiamo il setup se non ci sono documenti

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(documents)
    logging.info(f"Documenti divisi in {len(texts)} blocchi di testo.")

    if not texts:
        logging.warning("Nessun blocco di testo generato dopo lo splitting. Controlla il PDF.")
        return # Interrompiamo se lo splitting non produce nulla

    # 2. Creazione Embeddings e Vector Store (FAISS)
    logging.info(f"Creazione embeddings con il modello: {EMBEDDING_MODEL_NAME}")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, task_type="retrieval_query") # specifica il task type per FAISS
    except Exception as e:
         logging.error(f"Errore durante l'inizializzazione del modello di embedding '{EMBEDDING_MODEL_NAME}': {e}")
         logging.error("Controlla che il nome del modello sia corretto e che l'API key sia valida.")
         raise

    logging.info("Costruzione dell'indice FAISS...")
    # Qui creiamo l'indice FAISS dai testi e dagli embeddings.
    # Per applicazioni reali, potresti voler salvare/caricare l'indice per evitare di ricalcolarlo ogni volta.
    # Esempio di salvataggio/caricamento:
    # if os.path.exists(FAISS_INDEX_PATH):
    #     logging.info(f"Caricamento indice FAISS da {FAISS_INDEX_PATH}...")
    #     vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True) # Attenzione a allow_dangerous_deserialization
    # else:
    #     logging.info("Creazione nuovo indice FAISS...")
    #     vector_store = FAISS.from_documents(texts, embeddings)
    #     logging.info(f"Salvataggio indice FAISS in {FAISS_INDEX_PATH}...")
    #     vector_store.save_local(FAISS_INDEX_PATH)

    # Per questo esempio, lo ricreiamo ogni volta all'avvio:
    try:
        vector_store = FAISS.from_documents(texts, embeddings)
        logging.info("Indice FAISS creato con successo in memoria.")
    except Exception as e:
        logging.error(f"Errore durante la creazione dell'indice FAISS: {e}")
        raise

    # 3. Inizializzazione del Modello Generativo Gemini
    logging.info(f"Inizializzazione modello generativo: {GENERATION_MODEL_NAME}")
    try:
        llm = ChatGoogleGenerativeAI(model=GENERATION_MODEL_NAME,
                                 temperature=0.4, # Rendi la risposta più deterministica/basata sui fatti
                                 convert_system_message_to_human=True) # Alcuni modelli lo richiedono
    except Exception as e:
         logging.error(f"Errore durante l'inizializzazione del modello generativo '{GENERATION_MODEL_NAME}': {e}")
         logging.error("Controlla che il nome del modello sia corretto e che l'API key sia valida.")
         raise

    # 4. Creazione della Catena RetrievalQA
    #    Questa catena combina il recupero dal vector store e la generazione con l'LLM.
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Recupera i top 3 chunks più rilevanti

    # Definiamo un prompt template per guidare l'LLM
    prompt_template = """Usa le seguenti informazioni di contesto per rispondere alla domanda alla fine.
      Ricorda che sei un cane e quindi rispondi in maniera divertente ma coincisa. Nel prompt ti sarà inviato anche la chat fino ad adesso usala per rispondere. 

Contesto:
{context}

Domanda: {question}

Risposta utile:"""
    QA_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    logging.info("Creazione della catena RetrievalQA...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Mette tutti i chunk recuperati nel contesto
        retriever=retriever,
        return_source_documents=True, # Opzionale: restituisce anche i documenti sorgente
        chain_type_kwargs={"prompt": QA_PROMPT}
    )

    logging.info("Pipeline RAG configurata con successo.")

# --- Applicazione FastAPI ---
app = FastAPI(
    title="API RAG con Gemini e PDF",
    description="Interroga i tuoi documenti PDF tramite un'API GET usando Gemini.",
    version="1.0.0"
)

# Configurazione CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permette tutte le origini
    allow_credentials=True,
    allow_methods=["*"],  # Permette tutti i metodi
    allow_headers=["*"],  # Permette tutti gli headers
)

# Monta la cartella static per servire i file statici
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    """
    Esegue il setup della pipeline RAG all'avvio del server FastAPI.
    """
    try:
        setup_rag_pipeline()
    except FileNotFoundError as e:
         logging.error(f"Errore critico durante l'avvio: {e}. Il server non può funzionare senza il PDF.")
         # In un'applicazione reale, potresti voler uscire o impedire l'avvio completo.
         # FastAPI potrebbe avviarsi ma l'endpoint /chiedi fallirà.
    except Exception as e:
         logging.error(f"Errore imprevisto durante il setup della pipeline RAG: {e}")
         # Gestisci altri errori critici qui se necessario

# --- Funzioni di utilità per la gestione delle conversazioni ---
def get_or_create_session(session_id: Optional[str] = None) -> str:
    """
    Ottiene una sessione esistente o ne crea una nuova se necessario
    """
    if not session_id or session_id not in conversation_history:
        # Crea una nuova sessione
        new_session_id = str(uuid.uuid4())
        conversation_history[new_session_id] = [
            {"role": "assistant", "content": "Ciao! Sono Mia! Come posso aiutarti oggi? 🐾"}
        ]
        return new_session_id
    return session_id

def get_conversation_history(session_id: str) -> List[Dict[str, str]]:
    """
    Recupera la cronologia della conversazione per una sessione
    """
    # Pulisci le sessioni scadute
    cleanup_expired_sessions()
    
    # Restituisci la cronologia della sessione o una lista vuota se non esiste
    return conversation_history.get(session_id, [])

def add_to_conversation(session_id: str, role: str, content: str):
    """
    Aggiunge un messaggio alla cronologia della conversazione
    """
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    # Aggiungi timestamp per la gestione della scadenza
    conversation_history[session_id].append({
        "role": role, 
        "content": content,
        "timestamp": time.time()
    })

def cleanup_expired_sessions():
    """
    Rimuove le sessioni scadute dalla memoria
    """
    current_time = time.time()
    expired_sessions = []
    
    for session_id, messages in conversation_history.items():
        if messages and "timestamp" in messages[-1]:
            last_activity = messages[-1]["timestamp"]
            if current_time - last_activity > SESSION_EXPIRY:
                expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        del conversation_history[session_id]
        logging.info(f"Sessione scaduta rimossa: {session_id}")

def format_conversation_for_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Formatta la cronologia della conversazione in un formato adatto al prompt
    """
    formatted = ""
    for msg in messages:
        if "role" in msg and "content" in msg:
            role = "Mia" if msg["role"] == "assistant" else "Utente"
            formatted += f"{role}: {msg['content']}\n"
    return formatted

@app.get("/chiedi", tags=["RAG"])
async def ask_question(
    domanda: str = Query(..., # Rendi il parametro obbligatorio
                         min_length=3,
                         description="La domanda da porre ai documenti PDF."
                         ),
    session_id: Optional[str] = Cookie(None)
    ):
    """
    Endpoint per porre una domanda.
    Recupera contesto dai PDF indicizzati e genera una risposta usando Gemini.
    Mantiene la cronologia della conversazione per sessione.
    """
    if qa_chain is None:
        logging.error("La catena QA non è stata inizializzata correttamente (forse mancano i PDF o c'è stato un errore all'avvio).")
        raise HTTPException(status_code=503, detail="Servizio non pronto: pipeline RAG non inizializzata. Controlla i log del server.")

    if not domanda:
        raise HTTPException(status_code=400, detail="Il parametro 'domanda' non può essere vuoto.")

    # Gestione della sessione
    session_id = get_or_create_session(session_id)
    
    # Aggiungi la domanda dell'utente alla cronologia
    add_to_conversation(session_id, "user", domanda)
    
    # Ottieni la cronologia completa formattata
    conversation_context = format_conversation_for_prompt(get_conversation_history(session_id))
    
    logging.info(f"Ricevuta domanda: '{domanda}' (Sessione: {session_id})")
    logging.debug(f"Contesto conversazione: {conversation_context}")

    try:
        logging.info("Esecuzione della catena QA...")
        # Includiamo la conversazione nel contesto della domanda
        full_query = f"Conversazione precedente:\n{conversation_context}\n\nDomanda attuale: {domanda}"
        
        # Usiamo invoke invece di run per avere più controllo e accesso ai source_documents
        result = qa_chain.invoke({"query": full_query})

        answer = result.get('result', "Non è stato possibile generare una risposta.")
        source_documents = result.get('source_documents', [])

        logging.info(f"Risposta generata: '{answer}'")
        if source_documents:
             logging.info(f"Basata su {len(source_documents)} documenti sorgente recuperati.")

        # Aggiungi la risposta alla cronologia
        add_to_conversation(session_id, "assistant", answer)

        # Restituiamo la risposta, i riferimenti e l'ID di sessione
        response_data = {
            "domanda": domanda,
            "risposta": answer,
            "session_id": session_id,
            # Opzionale: includi riferimenti ai documenti sorgente
            "riferimenti": [
                {
                    "sorgente": doc.metadata.get('source', 'N/A'),
                    "pagina": doc.metadata.get('page', 'N/A'),
                } for doc in source_documents
            ] if source_documents else []
        }
        return response_data

    except Exception as e:
        logging.error(f"Errore durante l'elaborazione della domanda '{domanda}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Errore interno del server durante l'elaborazione della richiesta: {e}")

@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        return FileResponse("static/index.html")
    except Exception as e:
        logging.error(f"Errore nel caricamento di index.html: {e}")
        return HTMLResponse("""
            <html>
                <head>
                    <title>Errore</title>
                </head>
                <body>
                    <h1>Errore nel caricamento della pagina</h1>
                    <p>Si è verificato un errore nel caricamento della pagina. Riprova più tardi.</p>
                </body>
            </html>
        """, status_code=500)

@app.get("/admin/sessions", tags=["Admin"])
async def get_sessions_info():
    """
    Endpoint amministrativo per visualizzare le statistiche delle sessioni e pulire quelle scadute.
    Nota: In un ambiente di produzione, questo dovrebbe essere protetto da autenticazione.
    """
    # Pulizia delle sessioni scadute
    cleanup_expired_sessions()
    
    # Calcola statistiche
    total_sessions = len(conversation_history)
    sessions_info = []
    
    for session_id, messages in conversation_history.items():
        last_activity = time.time()
        message_count = len(messages)
        
        if messages and "timestamp" in messages[-1]:
            last_activity = messages[-1]["timestamp"]
            
        sessions_info.append({
            "session_id": session_id[:8] + "...",  # Solo una parte dell'ID per privacy
            "messages": message_count,
            "last_activity_seconds_ago": int(time.time() - last_activity),
            "expires_in_seconds": int(SESSION_EXPIRY - (time.time() - last_activity))
        })
    
    return {
        "total_active_sessions": total_sessions,
        "session_expiry_seconds": SESSION_EXPIRY,
        "sessions": sessions_info
    }

# --- Esecuzione dell'Applicazione ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Avvio del server FastAPI sulla porta {port}...")
    print("Vai su http://127.0.0.1:8000 per l'endpoint root.")
    print("Vai su http://127.0.0.1:8000/docs per la documentazione API interattiva (Swagger UI).")
    # Esegui il server Uvicorn
    run(app, host="0.0.0.0", port=port)
    # Usare host="0.0.0.0" rende l'API accessibile da altre macchine nella stessa rete.
    # Usa host="127.0.0.1" per renderla accessibile solo localmente.