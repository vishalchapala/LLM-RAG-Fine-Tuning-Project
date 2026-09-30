pip install -q transformers accelerate
from transformers import pipeline

   chat = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
   history = [{"role": "system", "content": "You are a helpful assistant."}]

   while True:
       q = input("You: ")
       if q.lower() in ("exit", "quit"):
           break
       history.append({"role": "user", "content": q})
       out = chat(history, max_new_tokens=200)[0]["generated_text"]
       reply = out[-1]["content"]
       history.append({"role": "assistant", "content": reply})
       print("Bot:", reply)
