import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


class VectorDBBuilder:

    def get_embedding(self, text):

        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )

        return response.data[0].embedding


    def build_index(self):

        df = pd.read_csv("data/fraud_rules.csv")

        print("Total rules:", len(df))

        for _, row in df.iterrows():

            rule = row["rule"]

            embedding = self.get_embedding(rule)

            supabase.table("fraud_vectors").insert({
                "rule": rule,
                "embedding": embedding
            }).execute()

            print("Inserted:", rule)


if __name__ == "__main__":

    builder = VectorDBBuilder()

    builder.build_index()