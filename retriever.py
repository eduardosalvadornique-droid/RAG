import os
from dotenv import load_dotenv
from supabase import create_client
from embeddings import get_embedding

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

def retrieve_chunks(query: str, top_k: int = 5, threshold: float = 0.70):
    """
    1) Genera embedding de la pregunta
    2) Consulta Supabase vía RPC
    3) Devuelve los chunks más similares
    """

    query_vector = get_embedding(query)

    response = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_vector,
            "similarity_threshold": threshold,
            "match_count": top_k
        }
    ).execute()

    return response.data or []

