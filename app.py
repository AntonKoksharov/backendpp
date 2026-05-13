import os
import json
import httpx
import asyncio
import sys
import io
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from sentence_transformers import SentenceTransformer, util

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GITHUB_REPO = "https://kb-api-server.onrender.com" 
GITHUB_PATH = "docs" 
DB_FILE = "vector_db.json"

client = genai.Client(api_key=GEMINI_API_KEY)
embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
vector_db = {}

async def sync_github_docs():
    print("Начало синхронизации с GitHub...")
    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_PATH}"
    
    async with httpx.AsyncClient() as http_client:
        try:
            resp = await http_client.get(api_url)
            if resp.status_code != 200:
                print(f"Ошибка GitHub API: {resp.status_code}")
                return
            
            files = resp.json()
            updated = False
            for file in files:
                if file["name"].endswith(".md") and file["path"] not in vector_db:
                    print(f"Новый файл: {file['name']}")
                    f_resp = await http_client.get(file["download_url"])
                    if f_resp.status_code == 200:
                        text = f_resp.text
                        chunks = [text[i:i+600] for i in range(0, len(text), 500)]
                        vectors = embed_model.encode(chunks).tolist()
                        vector_db[file["path"]] = {"chunks": chunks, "vectors": vectors}
                        updated = True
            
            if updated:
                with open(DB_FILE, "w", encoding="utf-8") as f:
                    json.dump(vector_db, f, ensure_ascii=False)
                print("Синхронизация завершена.")
        except Exception as e:
            print(f"Ошибка при синхронизации: {e}")

@app.on_event("startup")
async def startup():
    await sync_github_docs()

@app.post("/ask")
async def ask_endpoint(request: Request):
    body = await request.json()
    question = body.get("question", "")
    
    if not question:
        return {"answer": "Вопрос не введен."}

    query_vec = embed_model.encode(question)
    matches = []
    for data in vector_db.values():
        scores = util.cos_sim(query_vec, data["vectors"])[0]
        for i, s in enumerate(scores):
            if s > 0.35:
                matches.append(data["chunks"][i])
    
    context = "\n".join(matches[:3])
    if not context:
        return {"answer": "В базе знаний нет информации по этому вопросу."}

    prompt = f"Используй текст ниже для ответа:\n{context}\n\nВопрос: {question}"
    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model='gemini-2.5-flash',
            contents=prompt
        )
        return {"answer": response.text}
    except Exception as e:
        return {"answer": f"Ошибка ИИ: {str(e)}"}