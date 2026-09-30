!pip install -q langchain langchain-community langchain-huggingface langchain-text-splitters pypdf faiss-cpu sentence-transformers trl datasets

import torch
from google.colab import files
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import pipeline

# 1. Upload PDF and build retriever
pdf = list(files.upload().keys())[0]
docs = PyPDFLoader(pdf).load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
vectorstore = FAISS.from_documents(chunks, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("Retriever ready! Chunks:", len(chunks))

# 2. Training data
qa = [("What is programming?", "Programming is writing instructions that a computer can execute."),
      ("What is Python?", "Python is a high-level, general-purpose programming language."),
      ("What is debugging?", "Debugging is finding and fixing errors in a program.")]

prompt = lambda q: "Context:\n" + "\n\n".join(d.page_content for d in retriever.invoke(q)) + f"\n\nQuestion:\n{q}\n\nAnswer:"
data = Dataset.from_dict({"text": [prompt(q) + " " + a for q, a in qa]})

# 3. Fine-tune (auto-detects GPU or CPU)
gpu = torch.cuda.is_available()
trainer = SFTTrainer(model="Qwen/Qwen2.5-0.5B-Instruct", train_dataset=data,
                     args=SFTConfig(output_dir="./ft", num_train_epochs=3, per_device_train_batch_size=1,
                                    bf16=False, fp16=gpu, use_cpu=not gpu, report_to="none"))
trainer.train()
trainer.save_model("./ft")

# 4. Use the fine-tuned model
generator = pipeline("text-generation", model="./ft")
def fine_tuned_rag(q): return generator(prompt(q), max_new_tokens=80, return_full_text=False)[0]["generated_text"].strip()
print(fine_tuned_rag("What is Python?"))
import re, gradio as gr

# Smaller chunks, so retrieval is more specific
chunks2 = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20).split_documents(docs)
vs = FAISS.from_documents(chunks2, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

def fine_tuned_rag(q, max_dist=1.1):
    res = vs.similarity_search_with_score(q, k=2)
    print("distance:", round(float(res[0][1]), 2))          # for tuning; lower = more related
    if float(res[0][1]) > max_dist:
        return "Sorry, I can only answer questions from the programming PDF."
    context = "\n\n".join(d.page_content for d, _ in res)
    p = f"Context:\n{context}\n\nQuestion:\n{q}\n\nAnswer:"
    out = generator(p, max_new_tokens=60, return_full_text=False,
                    repetition_penalty=1.3, do_sample=False)[0]["generated_text"]
    return re.split(r"Question:|Context:|\n\n", out.strip())[0].strip()

gr.ChatInterface(fn=lambda q, h: fine_tuned_rag(q),
                 title="Programming RAG Chatbot").launch(share=True)