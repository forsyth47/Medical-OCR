import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

token = os.getenv("GROQ_API_KEY")
print("GROQ_API_KEY loaded:", bool(token))
if token:
    print("Preview:", token[:6] + "…" + token[-4:])

if token:
    try:
        llm = ChatOpenAI(
            model="qwen/qwen3.6-27b",
            api_key=token,
            base_url="https://api.groq.com/openai/v1"
        )
        out = llm.invoke("Reply with the single word: connected")
        print("✅ Connection OK →", out.content)
    except Exception as e:
        print("❌ Connection failed:", type(e).__name__, str(e)[:300])