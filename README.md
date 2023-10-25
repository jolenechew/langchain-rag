## Deployment

#### 1. Clone Repository

```bash
  git clone https://github.com/jolenechew/langchain-rag.git
```

```bash
  cd Migrant Workers Chatbot
```

#### 2. Create Virtual Environment

```bash
  python -m venv env
```

- For Windows:

```bash
  .\env\Scripts\activate
```

- For macOS/Linux:

```bash
  source env/bin/activate
```

#### 3. To install require packages

```bash
  pip install -r requirements.txt
```

#### 4. Replace your own documents in **data** folder

#### 5. Replace your own OpenAI and LangChain API keys in indexing.py, main.py & utils.py

#### 7. Run the web app

```bash
  streamlit run main.py
```

Reference:
https://github.com/farukalamai/ai-chatbot-using-Langchain-Pinecone
