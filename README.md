# ChatBot RAG con Gemini e PDF 🐕

Un chatbot interattivo che utilizza RAG (Retrieval Augmented Generation) con Google Gemini per rispondere a domande sui documenti PDF.

## Caratteristiche

- Interfaccia web interattiva con tema da cane
- Integrazione con Google Gemini per la generazione di risposte
- Sistema RAG per il recupero di informazioni dai PDF
- API REST con FastAPI
- Supporto CORS per l'integrazione frontend

## Requisiti

- Python 3.8+
- Google API Key per Gemini
- FastAPI
- LangChain
- FAISS
- Altri requisiti elencati in `requirements.txt`

## Installazione

1. Clona il repository:
```bash
git clone [URL_DEL_TUO_REPOSITORY]
cd [NOME_CARTELLA]
```

2. Crea un ambiente virtuale e attivalo:
```bash
python -m venv rag_env
source rag_env/bin/activate  # Per Linux/Mac
rag_env\Scripts\activate     # Per Windows
```

3. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

4. Crea un file `.env` nella root del progetto e aggiungi la tua Google API Key:
```
GOOGLE_API_KEY=la_tua_api_key
```

5. Crea una cartella `documenti_pdf` e inserisci i tuoi PDF

## Utilizzo

1. Avvia il server FastAPI:
```bash
python main.py
```

2. Apri `index.html` nel tuo browser

3. Inizia a chattare con il bot!

## Struttura del Progetto

- `main.py`: Server FastAPI e logica RAG
- `index.html`: Interfaccia utente del chatbot
- `requirements.txt`: Dipendenze del progetto
- `documenti_pdf/`: Cartella per i PDF da processare
- `faiss_index_gemini/`: Indice FAISS per il recupero dei documenti

## Licenza

Questo progetto è privato e non distribuito pubblicamente. 