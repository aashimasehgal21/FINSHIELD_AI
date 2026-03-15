import os
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


class VectorSearch:

    def get_embedding(self, text):

        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )

        return response.data[0].embedding


    def search(self, query, top_k=3):

        query_embedding = self.get_embedding(query)

        result = supabase.rpc(
            "match_fraud_vectors",
            {
                "query_embedding": query_embedding,
                "match_count": top_k
            }
        ).execute()

        return result.data


if __name__ == "__main__":

    vs = VectorSearch()

    results = vs.search("rapid transactions from same account")

    print("\nVector Search Results:\n")

    for r in results:
        print(r)