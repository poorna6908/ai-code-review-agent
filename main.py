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

# Model used for reviews. Keep this in sync with what you write in your portfolio.
MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = """You are a senior code reviewer. Respond in markdown with exactly these sections:

## Summary
2-3 sentences on the overall quality of the code.

## Issues Found
A markdown table with columns: Severity | Issue | Fix
Cover bugs, style problems and performance problems. Keep each cell short.

## Improved Code
One code block with the corrected version of the code.

Be concise. Do not add any other sections, test checklists or long explanations."""


# Input format
class CodeInput(BaseModel):
    code: str
    language: str = "unknown"


@app.get("/")
def home():
    return {"message": "API working"}


# AI Code Review Endpoint
@app.post("/review")
def review_code(data: CodeInput):
    try:
        user_prompt = (
            f"Language: {data.language}\n\n"
            f"Review this code:\n```{data.language}\n{data.code}\n```"
        )

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            reasoning_effort="low",
            max_completion_tokens=4096,
        )

        return {"review": response.choices[0].message.content}

    except Exception as e:
        return {"error": str(e)}
