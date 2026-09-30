!pip install -q chromadb pypdf transformers accelerate
import re, chromadb
from pypdf import PdfReader
from google.colab import files
from transformers import pipeline

# Upload the PDF and save it in the database
name = list(files.upload().keys())[0]
text = "\n".join(p.extract_text() for p in PdfReader(name).pages)
chunks = re.split(r"\n(?=\d+\.\s)", text)
db = chromadb.Client().create_collection("docs")
db.add(documents=chunks, ids=[str(i) for i in range(len(chunks))])

# Load the free AI model
bot = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct", device_map="auto")

