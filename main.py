import os
import re
import uuid
from typing import Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from google import genai
from supabase import create_client, Client
import edge_tts

app = FastAPI(title="English Learning Assistant API")

# --- 1. CONFIGURATION & CLIENTS ---
# Read credentials from environment variables (set these before starting the server).
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

missing = [
    name
    for name, value in [
        ("GEMINI_API_KEY", GEMINI_API_KEY),
        ("SUPABASE_URL", SUPABASE_URL),
        ("SUPABASE_KEY", SUPABASE_KEY),
    ]
    if not value
]
if missing:
    raise RuntimeError(
        "Missing required environment variable(s): " + ", ".join(missing) +
        ". Set them (e.g. in a .env file loaded before startup, or in your "
        "hosting platform's environment settings) before running the server."
    )

# The old "google-generativeai" package (google.generativeai) is officially
# deprecated/archived and no longer maintained. This uses Google's current
# "google-genai" SDK instead. gemini-1.5-flash and gemini-2.5-flash have both
# been (or are about to be) retired, so we use the current stable Flash model.
MODEL_NAME = "gemini-3.5-flash"

genai_client = genai.Client(api_key=GEMINI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

VALID_CATEGORIES = {"Noun", "Verb", "Adjective", "Adverb", "Idiom", "Phrasal Verb"}

# Folder where generated story audio is saved. Each file gets a unique name
# so simultaneous requests don't overwrite each other.
AUDIO_DIR = "generated_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)


# --- 2. DATA MODELS ---
class TranslateRequest(BaseModel):
    text: str
    custom_prompt: Optional[str] = None  # UI එකෙන් prompt එක customize කරන්න ඕන නම්

class WordItem(BaseModel):
    word: str
    meaning_sinhala: str
    meaning_english: str

class StoryRequest(BaseModel):
    new_words: List[str]


# --- 3. API ENDPOINTS ---

@app.get("/")
def home():
    return {"status": "Backend Server is Running Successfully!"}


# Feature 01: Smart AI Translation (Gemini Dynamic Breakdown)
@app.post("/translate")
def translate_word(req: TranslateRequest):
    default_prompt = f"""
    Translate and break down the word/phrase: "{req.text}".
    If it is English, translate to Sinhala & Simple English.
    If it is Sinhala, translate to English.
    
    Provide output strictly in JSON format with these exact keys:
    {{
        "word": "{req.text}",
        "ipa_pronunciation": "/.../",
        "part_of_speech": "Noun/Verb/Adjective/etc.",
        "sinhala_meaning": "සිංහල තේරුම",
        "simple_english_definition": "Simple English explanation",
        "synonyms": ["word1", "word2"],
        "antonyms": ["word1", "word2"],
        "usage_examples": ["Example sentence 1.", "Example sentence 2."]
    }}
    Do not add markdown backticks like ```json in raw text, return clean JSON string.
    """

    prompt = req.custom_prompt if req.custom_prompt else default_prompt
    try:
        response = genai_client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        return {"result": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {e}")


# Feature 02: Categorized Known Words Database Management
@app.post("/add-known-word")
def add_known_word(item: WordItem):
    word_lower = item.word.lower()
    try:
        # Avoid inserting a duplicate word (and a confusing crash if the
        # "word" column has a unique constraint in Supabase).
        existing = (
            supabase.table("known_words")
            .select("id")
            .eq("word", word_lower)
            .execute()
        )
        if existing.data:
            raise HTTPException(
                status_code=409,
                detail=f"'{item.word}' is already in your known words list.",
            )

        # Gemini මගින් Word එකේ Part of Speech (Noun, Verb, etc.) Automatically හොයාගැනීම
        cat_prompt = (
            f"Categorize the English word '{item.word}' into exactly one of "
            "these: Noun, Verb, Adjective, Adverb, Idiom, Phrasal Verb. "
            "Output ONLY the category name."
        )
        cat_response = genai_client.models.generate_content(
            model=MODEL_NAME,
            contents=cat_prompt,
        ).text.strip()

        # Fall back to "Uncategorized" if Gemini returns something unexpected,
        # instead of silently saving a bad category value.
        category = cat_response if cat_response in VALID_CATEGORIES else "Uncategorized"

        data = {
            "word": word_lower,
            "meaning_sinhala": item.meaning_sinhala,
            "meaning_english": item.meaning_english,
            "category": category,
        }

        res = supabase.table("known_words").insert(data).execute()
        return {"status": "success", "category_assigned": category, "data": res.data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not add word: {e}")


@app.get("/get-known-words")
def get_known_words(category: Optional[str] = None, search: Optional[str] = None):
    try:
        query = supabase.table("known_words").select("*")
        if category:
            query = query.eq("category", category)
        if search:
            query = query.ilike("word", f"%{search}%")

        res = query.execute()
        return {"words": res.data, "total_count": len(res.data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not fetch words: {e}")


# Feature 03: Audio Story & Practice Material Generator
@app.post("/generate-audio-story")
async def generate_story(req: StoryRequest):
    try:
        # Fetch some random known words from Database to mix in
        known_res = supabase.table("known_words").select("word").limit(10).execute()
        known_words = [item['word'] for item in known_res.data]

        story_prompt = f"""
        Create a highly engaging, memorable, short English story using these NEW words: {req.new_words}.
        Also naturally blend in some of these known words: {known_words}.
        The story should be clear, easy to understand, and help memorize the new words from context.
        Return ONLY the English story text.
        """

        # Using the async client here so this request doesn't block the
        # server from handling other requests while waiting on Gemini.
        story_response = await genai_client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=story_prompt,
        )
        story_text = story_response.text

        # Text-To-Speech conversion using Edge-TTS (Free Natural Voice).
        # A unique filename per request so concurrent story requests don't
        # overwrite each other's audio.
        output_filename = f"{AUDIO_DIR}/story_{uuid.uuid4().hex}.mp3"
        communicate = edge_tts.Communicate(story_text, "en-US-ChristopherNeural")
        await communicate.save(output_filename)

        return {
            "story_text": story_text,
            "audio_file": output_filename,
            "audio_status": "Audio generated successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Story generation failed: {e}")


# Feature 04: SRT Subtitle Vocabulary Extractor
@app.post("/filter-srt")
async def filter_srt(file: UploadFile = File(...)):
    try:
        content = await file.read()
        srt_text = content.decode("utf-8", errors="ignore")

        # Remove sequence-number + timestamp lines. `\r?\n` handles both
        # Windows-style (\r\n) and Unix-style (\n) line endings, which the
        # original pattern (\n only) missed for many real .srt files.
        clean_text = re.sub(
            r'\d+\r?\n\d\d:\d\d:\d\d,\d\d\d --> \d\d:\d\d:\d\d,\d\d\d\r?\n',
            '',
            srt_text,
        )
        # Actually strip HTML-style formatting tags (<i>, <b>, <font ...>, etc.)
        clean_text = re.sub(r'<[^>]+>', '', clean_text)

        # Keep contractions like "don't" / "it's" as single words instead of
        # splitting them into "don" + "t" / "it" + "s".
        words = {
            w.lower()
            for w in re.findall(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b", clean_text)
        }

        # Fetch all known words from Supabase Database
        db_res = supabase.table("known_words").select("word").execute()
        known_words_set = {item['word'].lower() for item in db_res.data}

        # Filter out known words. Sorted for a stable, predictable order
        # instead of Python's arbitrary set ordering.
        unknown_words = sorted(words - known_words_set)[:30]

        # Generate Quick Breakdown for unknown words using Gemini
        breakdown_prompt = (
            f"Provide brief Sinhala & Simple English meanings for these "
            f"movie vocabulary words: {unknown_words}. Format as a clean "
            "dictionary list."
        )
        breakdown_response = await genai_client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=breakdown_prompt,
        )

        return {
            "unknown_words_count": len(unknown_words),
            "unknown_words": unknown_words,
            "vocabulary_breakdown": breakdown_response.text,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SRT processing failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
