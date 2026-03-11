import os
from dotenv import load_dotenv
from supabase import create_client
from embeddings import get_embedding

load_dotenv()

_supabase = None

def get_supabase():
    global _supabase
    if _supabase is not None:
        return _supabase

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url:
        raise RuntimeError("SUPABASE_URL no está configurada.")
    if not key:
        raise RuntimeError("SUPABASE_KEY no está configurada.")

    # Quita espacios/saltos por si copiaste con newline
    url = url.strip()
    key = key.strip()

    _supabase = create_client(url, key)
    return _supabase


def retrieve_chunks(query: str, top_k: int = 5, threshold: float = 0.70):
    supabase = get_supabase()

    query_vector = get_embedding(query)

    resp = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_vector,
            "similarity_threshold": threshold,
            "match_count": top_k
        }
    ).execute()

    return resp.data or []
