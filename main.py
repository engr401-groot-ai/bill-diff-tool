import os
import logging
from dotenv import load_dotenv
from google.cloud import secretmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, FileResponse, HTMLResponse, Response
from pydantic import BaseModel
import google.generativeai as genai

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
def load_environment_variables() -> None:
    """Load environment variables from Secret Manager (Cloud Run) or .env (local)."""
    project_id = os.getenv("GCP_PROJECT_ID", "")
    secret_name = f"projects/{project_id}/secrets/bill-diff-tool-env/versions/latest"

    try:
        client = secretmanager.SecretManagerServiceClient()
        response = client.access_secret_version(request={"name": secret_name})
        payload = response.payload.data.decode("utf-8")

        for line in payload.strip().splitlines():
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

        logger.info("✓ Loaded environment variables from Secret Manager")
        return

    except Exception as e:
        logger.info(f"Secret Manager load failed ({e}); falling back to .env")
        load_dotenv()
        logger.info("✓ Loaded environment variables from .env (if present)")

load_environment_variables()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("✓ Gemini API configured")
else:
    logger.warning("⚠ GEMINI_API_KEY not found - API calls will fail")

class BillRequest(BaseModel):
    bill1_text: str
    bill2_text: str

class BillResponse(BaseModel):
    summary: str
    success: bool
    error: str | None = None

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
@app.post("/compare-bills", response_model=BillResponse)
def compare_bills(request: BillRequest):
    """
    Compare two versions of a bill using Google Gemini.
    """
    try:
        logger.info("Processing bill comparison request")
        
        # Validate inputs
        if not request.bill1_text.strip() or not request.bill2_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Both bill1_text and bill2_text must be non-empty"
            )
        
        # Create prompt for Gemini
        prompt = f"""Role: Senior Legislative Analyst for a University Government Relations Office.
Task: Write a concise (about 100-250 words or 1500 characters) comparative summary of the changes between the two provided bill versions.

Guidelines:
1.  **Identify Bill Versions:** Look at the top of the provided text to identify the specific bill numbers (e.g., "SB 119", "SB 119 CD1"). Use these specific names in your summary. If names are not found, use "the original bill" and "the amended version".
2.  **Single Paragraph:** Output the entire summary as a **single cohesive paragraph**. Do not split into multiple paragraphs.
3.  **Focus on "Primary Differences":** Start immediately by stating the main differences (e.g., "The primary differences between [Bill A] and [Bill B] lie in...").
4.  **Be Specific with Numbers:** Explicitly compare funding amounts, fiscal years, and dates.
5.  **Structure vs. Content:** Note changes in organization (e.g., "consolidates funding") as well as content.
6.  **Plain Language:** Use simple, clear language. Avoid formal or complex words like "predominantly", "itemized", "pursuant to". Use everyday words instead (e.g., "mainly", "listed", "under").
7.  **Concise & Direct:** Professional, objective tone. No filler.

Example Style:
"The primary differences between the original SB 119 and the final version, SB 119 CD1 (passed as Act 265), lie in the appropriation structure and the total funding amount for the second fiscal year. While both versions allocate $250,000 for fiscal year 2025-2026, the final CD1 version reduces the appropriation for fiscal year 2026-2027 from the originally proposed $430,000 to $350,000. Additionally, the original bill separated funding into specific line items for prerequisites, personnel, and supplies across multiple sections, whereas the final enacted version consolidates all funding into a single section with lump sums authorized for all program purposes."

**First Bill Text:**
{request.bill1_text}

**Second Bill Text:**
{request.bill2_text}
"""

        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        
        if not response or not response.text:
            raise Exception("Gemini API returned empty response")
        
        logger.info("✓ Bill comparison completed successfully")
        
        return BillResponse(
            summary=response.text,
            success=True,
            error=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing bills: {str(e)}")
        return BillResponse(
            summary="",
            success=False,
            error=f"Failed to compare bills: {str(e)}"
        )
    