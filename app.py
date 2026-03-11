import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from retriever import retrieve_chunks
from fastapi.middleware.cors import CORSMiddleware

# Render NO necesita .env si usas Environment Variables,
# pero esto ayuda localmente:
load_dotenv()

app = FastAPI(title="RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # temporal para pruebas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")


class ChatRequest(BaseModel):
    question: str
    top_k: int = 6
    threshold: float = 0.70


class ChatResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    try:
        # 1) Retrieval
        chunks = retrieve_chunks(question, top_k=req.top_k, threshold=req.threshold)

        if not chunks:
            return {"answer": "No encontré información relevante en la base de conocimiento."}

        # 2) Contexto
        context = "\n\n---\n\n".join([c["content"] for c in chunks])

        # 3) LLM
        messages = [
            {
                "role": "system",
                "content": (
                    "Responde únicamente usando el contexto proporcionado. "
                    "Si no encuentras la respuesta en el contexto, indícalo claramente."
                ),
            },
            {
                "role": "user",
                "content": f"CONTEXTO:\n{context}\n\nPREGUNTA:\n{question}",
            },
        ]

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2,
        )

        answer = response.choices[0].message.content
        return {"answer": answer}

    except Exception as e:
        # En prod podrías loggear mejor; por ahora devolvemos mensaje genérico

        raise HTTPException(status_code=500, detail=f"Error procesando la consulta: {str(e)}")
