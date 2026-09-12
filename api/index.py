import os
import re
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import google.generativeai as genai
from supabase import create_client, Client
import edge_tts
from mangum import Mangum

app = FastAPI(title="English Learning Assistant API")

# Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


class TranslateRequest(BaseModel):
    text: str
    custom_prompt: str = None

class WordItem(BaseModel):
    word: str
    meaning_sinhala: str
    meaning_english: str

class StoryRequest(BaseModel):
    new_words: list[str]


@app.get("/")
def home():
    return {"status": "Backend Server is Running Successfully!"}


@app.post("/translate")
def translate_word(req: TranslateRequest):
    default_prompt = f"""
    Translate and break down: "{req.text}".
    Output JSON: {{"word": "{req.text}", "ipa_pronunciation": "/.../", "part_of_speech": "...", "sinhala_meaning": "...", "simple_english_definition": "...", "synonyms": [], "antonyms": [], "usage_examples": []}}
    Return clean JSON string without markdown formatting.
    """
    prompt = req.custom_prompt if req.custom_prompt else default_prompt
    try:
        response = model.generate_content(prompt)
        return {"result": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add-known-word")
def add_known_word(item: WordItem):
    cat_prompt = f"Categorize '{item.word}' into one: Noun, Verb, Adjective, Adverb, Idiom, Phrasal Verb. Output ONLY category."
    cat_response = model.generate_content(cat_prompt).text.strip()
    
    data = {
        "word": item.word.lower(),
        "meaning_sinhala": item.meaning_sinhala,
        "meaning_english": item.meaning_english,
        "category": cat_response
    }
    res = supabase.table("known_words").insert(data).execute()
    return {"status": "success", "category": cat_response, "data": res.data}


@app.get("/get-known-words")
def get_known_words(category: str = None, search: str = None):
    query = supabase.table("known_words").select("*")
    if category:
        query = query.eq("category", category)
    if search:
        query = query.ilike("word", f"%{search}%")
    res = query.execute()
    return {"words": res.data, "total_count": len(res.data)}

handler = Mangum(app)
