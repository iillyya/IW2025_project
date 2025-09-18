import argparse
import chromadb
import os
from google import genai
from google.genai.types import EmbedContentConfig

# def generate_embeddings(text, api_key=None):
#   api_key = api_key or os.environ.get('GEMINI_API_KEY')
#   if not api_key:
#     raise ValueError("Gemini API key not provided")

#   try:
#     client = genai.Client(api_key=api_key)

#     response = client.models.embed_content(
#       model="gemini-embedding-001",
#       contents=text,
#       config=EmbedContentConfig(
#         task_type="RETRIEVAL_DOCUMENT",  # Optional
#         output_dimensionality=3072,  # Optional
#       ),
#     )

#     return response.embeddings
#   except Exception as e:
#     raise Exception(f"Error generating embedding: {e}")


def main():
  parser = argparse.ArgumentParser(description="Search for material")

  parser.add_argument("--query", help="Search query / describe a material")
  # parser.add_argument("--api-key", help="Gemini API key (can also use GEMINI_API_KEY env variable)")
  args = parser.parse_args()

  # if args.api_key:
  #   os.environ['GEMINI_API_KEY'] = args.api_key

  client = chromadb.PersistentClient(path="./chroma_db")
  collection = client.get_or_create_collection("materials")

  query = args.query

  print(f"Searching for: {query}")

  results = collection.query(
    query_texts=[query],
    n_results=3
  )

  print("Results:")
  print(results)
  print("foo")
  for doc_id in results['ids'][0]:
    print(f"- {doc_id}")

if __name__ == "__main__":
  main()
