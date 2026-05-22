import os
import json
import httpx
import asyncio
import sys
import io
import math
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

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
GITHUB_REPO = "luquii2/Knowledge-Base"
GITHUB_PATH = "places" 
DB_FILE = "vector_db.json"

client = genai.Client(api_key=GEMINI_API_KEY)
vector_db = {}

#ЛЕГКАЯ МАТЕМАТИКА ВМЕСТО ТЯЖЕЛЫХ БИБЛИОТЕК

def get_embeddings_batch(texts):
    """Отправляет пачку текста в Google для векторизации"""
    response = client.models.embed_content(
        model='gemini-embedding-2',
        contents=texts
    )
    return [e.values for e in response.embeddings]

def get_embedding_single(text):
    """Векторизует один вопрос пользователя"""
    response = client.models.embed_content(
        model='gemini-embedding-2',
        contents=text
    )
    return response.embeddings[0].values

def cosine_similarity(v1, v2):
    """Вычисляет похожесть без использования тяжелых библиотек"""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude1 = math.sqrt(sum(a * a for a in v1))
    magnitude2 = math.sqrt(sum(b * b for b in v2))
    if magnitude1 == 0 or magnitude2 == 0: return 0.0
    return dot_product / (magnitude1 * magnitude2)

#ОСНОВНАЯ ЛОГИКА 

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
                        
                        if chunks:
                            # Просим Gemini векторизовать текст (работает за миллисекунды)
                            vectors = await asyncio.to_thread(get_embeddings_batch, chunks)
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
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding="utf-8") as f:
            global vector_db
            vector_db = json.load(f)
    await sync_github_docs()

@app.post("/ask")
async def ask_endpoint(request: Request):
    body = await request.json()
    question = body.get("question", "")
    
    if not question:
        return {"answer": "Вопрос не введен."}

    try:
        # 1. Векторизуем вопрос пользователя через Google
        query_vec = await asyncio.to_thread(get_embedding_single, question)
    except Exception as e:
         return {"answer": f"Ошибка векторизации вопроса: {str(e)}"}

    # 2. Ищем совпадения локально с помощью легкой математики
    matches = []
    for data in vector_db.values():
        for i, chunk_vec in enumerate(data["vectors"]):
            score = cosine_similarity(query_vec, chunk_vec)
            if score > 0.5: # Порог похожести
                matches.append({"text": data["chunks"][i], "score": score})
    
    matches = sorted(matches, key=lambda x: x["score"], reverse=True)
    context = "\n".join([m["text"] for m in matches[:3]])
    
    if not context:
        return {"answer": "В базе знаний нет информации по этому вопросу."}

    # 3. Отправляем контекст в ИИ
    # 3. Отправляем контекст в ИИ
    prompt = f"""Используй предоставленный текст для формирования ответа на вопрос пользователя. 
Если в данном тексте нет нужной информации, то просто напиши, что ты не знаешь ответа. 
Каждый свой ответ обязательно начинай с фразы: 'Вот что удалось найти по вашему вопросу:'

Текст для поиска:
{context}

Вопрос пользователя: {question}"""
    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=1256,
            )
        )

        return {"answer": response.text}
    except Exception as e:
        return {"answer": f"Ошибка ИИ: {str(e)}"}

#Подключен сервис, который не даст заснуть серверу.
@app.get("/")
async def root():
    return {"status": "ok", "message": "Сервер не спит!"}