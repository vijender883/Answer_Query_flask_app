# Answer_Query_Flask_App

This repository hosts a Flask-based API for answering user queries using Gemini and Pinecone integrations.

## 🚀 Getting Started

### 1. Clone the Repository

Open a terminal in your desired folder and run:

```bash
git clone https://github.com/vijender883/Answer_Query_flask_app
cd Answer_Query_flask_app
````

---

### 2. Set Up Virtual Environment

#### 🪟 Windows CMD:

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

#### 🪟 Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### 🍏 macOS Terminal:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 🐧 Linux Terminal:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### 3. Create a `.env` File

In the root of the project, create a file named `.env` and add the following:

```env
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_INDEX=your_index_name
MCP_PORT=8000
```

---

### 4. Run the App

Run the following command to start the API:

```bash
python answer_query.py
```

The server will run at:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

### 5. Test the API with Postman

**POST** request to:
`http://127.0.0.1:5000/api/answer`

**JSON Body:**

```json
{
  "query": "your query about the data",
  "previous_chats": "optional previous context"
}
```

---

## 📫 Contact

For questions or suggestions, feel free to open an issue or contact the repo maintainer.

