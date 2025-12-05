import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, FileResponse, HTMLResponse, Response
from pydantic import BaseModel
from typing import Optional
import google.generativeai as genai
from google.cloud import texttospeech
import base64
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("diff_tool")

# Initialize FastAPI
app = FastAPI(
    title="Bill Diff Tool API",
    description="API for comparing two versions of legislative bills using Google Gemini",
    version="1.0.0"
)

# Load environment variables
load_dotenv()
logger.info("Loaded environment variables.")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("✓ Gemini API configured")
else:
    logger.warning("⚠ GEMINI_API_KEY not found - API calls will fail")

# Request/Response Models
class CompareRequest(BaseModel):
    bill1_text: str
    bill2_text: str

class CompareResponse(BaseModel):
    summary: str
    audio_base64: Optional[str] = None
    success: bool
    error: Optional[str] = None

def preprocess_text_for_speech(text: str) -> str:
    """Preprocesses text to expand legislative abbreviations for better TTS pronunciation."""
    # List of replacements (Handles both standalone and number-preceding cases)
    replacements = [
        (r'\bHB\s*(?=\d)', 'House Bill '),
        (r'\bHB\b', 'House Bill'),
        (r'\bSB\s*(?=\d)', 'Senate Bill '),
        (r'\bSB\b', 'Senate Bill'),
        (r'\bHD\s*(?=\d)', 'House Draft '),
        (r'\bHD\b', 'House Draft'),
        (r'\bSD\s*(?=\d)', 'Senate Draft '),
        (r'\bSD\b', 'Senate Draft'),
        (r'\bCD\s*(?=\d)', 'Conference Draft '),
        (r'\bCD\b', 'Conference Draft'),
        (r'\bFD\s*(?=\d)', 'Final Draft '),
        (r'\bFD\b', 'Final Draft'),
        (r'\bGM\s*(?=\d)', 'Governor\'s Message '),
        (r'\bGM\b', 'Governor\'s Message'),

        (r'["\'()*\[\]{}]', ''),
        (r'§', 'Section '),
        (r'&', ' and '),
        (r'_', ' '),
        (r'https?://\S+', ''),
        (r'%', ' percent '),
        (r'\+', ' plus '),
        (r'=', ' equals '),
        (r'\s+', ' '),
    ]
    
    processed_text = text
    for pattern, replacement in replacements:
        processed_text = re.sub(pattern, replacement, processed_text)
        
    return processed_text

# FRONT END ROUTES
@app.get("/")
def root_redirect():
    """Redirect root to the UI page"""
    return RedirectResponse(url="/ui")

@app.get("/favicon.ico")
def favicon() -> Response:
    """Return 204 for favicon requests (no favicon)."""
    return Response(status_code=204)

@app.get("/ui")
def ui():
    """Serve the static/index.html file if present, otherwise return a simple 404 HTML response."""
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)

@app.get("/bills.json")
def bills_json():
    """Serve bills.json"""
    file_path = os.path.join(os.path.dirname(__file__), "bills.json")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/json")
    return Response(status_code=404)

@app.get("/style.css")
def style_css():
    """Serve style.css"""
    file_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="text/css")
    return Response(status_code=404)

# BACKEND ROUTES
@app.post("/compare-and-speak", response_model=CompareResponse)
def compare_and_speak(request: CompareRequest):
    """Compares two versions of a bill and generates audio from the comparison summary (google gemini 2.5 flash)."""
    try:
        # 1. Generate Summary
        logger.info("Processing bill comparison request")
        
        # Validate inputs
        if not request.bill1_text.strip() or not request.bill2_text.strip():
            raise HTTPException(status_code=400, detail="Both bill texts must be provided")

        prompt = f"""Role: Senior Legislative Analyst for a University Government Relations Office.
            Task: Write a concise (about 100 - 150 words or 1500 characters) comparative summary of the changes between the two provided bill versions.

            Guidelines:
            1.  Identify Bill Versions: Look at the top of the provided text to identify the specific bill numbers (e.g., "SB 119", "SB 119 CD1"). Use these specific names in your summary. If names are not found, use "the original bill" and "the amended version".
            2.  Single Paragraph: Output the entire summary as a single cohesive paragraph. Do not split into multiple paragraphs.
            3.  Focus on "Primary Differences": Start immediately by stating the main differences (e.g., "The primary differences between [Bill A] and [Bill B] lie in...").
            4.  Be Specific with Numbers: Explicitly compare funding amounts, fiscal years, and dates.
            5.  Structure vs. Content: Note changes in organization (e.g., "consolidates funding") as well as content.
            6.  Plain Language: Use simple, clear language. Avoid formal or complex words like "predominantly", "itemized", "pursuant to". Use everyday words instead (e.g., "mainly", "listed", "under").
            7.  Concise & Direct: Professional, objective tone. No filler.
            8.  No Hallucinations: Do not introduce any new facts, numbers, dates, or claims not present in the two provided bill texts. Rely only on the two bill texts as sources of truth.
            9.  Striclty output plain text only: Do not use any markdown formatting. No bolding (**text**), no italics (*text*), no headers (#), no bullet points. Write in standard paragraph form only.
            10. Standard Grammar: Use standard English grammar and capitalization. Do not uppercase words like "OR", "AND", or "NOT" for emphasis.

            Example Style:
            The primary differences between the original SB 119 and the final version, SB 119 CD1 (passed as Act 265), lie in the appropriation structure and the total funding amount for the second fiscal year. While both versions allocate $250,000 for fiscal year 2025-2026, the final CD1 version reduces the appropriation for fiscal year 2026-2027 from the originally proposed $430,000 to $350,000. Additionally, the original bill separated funding into specific line items for prerequisites, personnel, and supplies across multiple sections, whereas the final enacted version consolidates all funding into a single section with lump sums authorized for all program purposes.

            First Bill Text:
            {request.bill1_text}

            Second Bill Text:
            {request.bill2_text}
            """

        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        
        if not response or not response.text:
            raise Exception("Gemini API returned empty response")
            
        summary_text = response.text
        logger.info("✓ Summary generated")

        # 2. Generate Speech
        audio_base64 = None
        try:
            client = texttospeech.TextToSpeechClient()
            
            spoken_text = preprocess_text_for_speech(summary_text)
            
            synthesis_input = texttospeech.SynthesisInput(
                text=spoken_text
            )
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name="en-US-Standard-H"
            )
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=1
            )
            
            speech_response = client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            
            audio_base64 = base64.b64encode(speech_response.audio_content).decode('utf-8')
            logger.info("✓ Speech generated")
        
        # Return the summary even if speech fails
        except Exception as e:
            logger.error(f"Failed to generate speech in combined endpoint: {e}")
            audio_base64 = None

        return CompareResponse(
            summary=summary_text,
            audio_base64=audio_base64,
            success=True,
            error=None
        )

    except Exception as e:
        logger.error(f"Error comparing and speaking: {str(e)}")
        return CompareResponse(
            summary="",
            audio_base64=None,
            success=False,
            error=f"Failed to process request: {str(e)}"
        )
