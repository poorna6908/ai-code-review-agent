from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from groq import Groq

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Groq API key
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Input format
class CodeInput(BaseModel):
    code: str

@app.get("/")
def home():
    return {"message": "API working"}

# AI Code Review Endpoint
@app.post("/review")
def review_code(data: CodeInput):
    try:
        prompt = f"Review this code:\n{data.code}"

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return {
            "review": response.choices[0].message.content
        }

    except Exception as e:
        return {"error": str(e)}