import os
import json
import uuid
import shutil
import tempfile
from pathlib import Path
from moviepy.editor import VideoFileClip, AudioFileClip

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import google.generativeai as genai

load_dotenv()

app = FastAPI(title="MeetingIQ – AI Meeting Intelligence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("WARNING: GROQ_API_KEY not found in environment or .env file.")

client = Groq(api_key=api_key)

gemini_key = os.environ.get("GEMINI_API_KEY")
if gemini_key:
    genai.configure(api_key=gemini_key)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    print("WARNING: GEMINI_API_KEY not found. Fallback disabled.")
    gemini_model = None

TRANSCRIPTION_MODEL = "whisper-large-v3"
NOTES_MODEL = "llama-3.3-70b-versatile"

SUPPORTED_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".wmv", ".flv", ".webm",
    ".m4v", ".3gp", ".ts", ".mts", ".m2ts",
    ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".wma",
    ".opus", ".webm", ".mpga", ".mpeg",
}

TEMP_DIR = Path(tempfile.gettempdir()) / "meetingiq"
TEMP_DIR.mkdir(exist_ok=True)


def extract_audio(input_path: Path, output_path: Path) -> None:
    """Extract audio from video or convert audio to MP3 using moviepy."""
    print(f"DEBUG: Extracting audio from {input_path.name} to {output_path.name}...")
    try:
        if input_path.suffix.lower() in {'.mp4', '.mkv', '.mov', '.avi', '.wmv', '.flv', '.webm', '.m4v'}:
            clip = VideoFileClip(str(input_path))
            audio_target = clip.audio
        else:
            clip = AudioFileClip(str(input_path))
            audio_target = clip
        
        audio_target.write_audiofile(
            str(output_path), 
            bitrate="64k",
            verbose=False,
            logger=None
        )
        clip.close()
        print(f"DEBUG: Extraction complete. Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"ERROR during extraction: {e}")
        raise RuntimeError(f"Audio extraction error: {str(e)}")


def transcribe_audio(audio_path: Path) -> str:
    """Send audio to Groq Whisper and return transcript text."""
    file_size = audio_path.stat().st_size
    print(f"DEBUG: Transcribing {audio_path.name} (Size: {file_size / 1024 / 1024:.2f} MB)")
    
    MAX_BYTES = 22 * 1024 * 1024 

    if file_size <= MAX_BYTES:
        try:
            with open(audio_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    file=(audio_path.name, f.read()),
                    model=TRANSCRIPTION_MODEL,
                    response_format="text",
                    language="en",
                    temperature=0.0,
                )
            return result if isinstance(result, str) else result.text
        except Exception as e:
            print(f"DEBUG: Transcription failed for single file: {e}")
            if "413" in str(e) or "too_large" in str(e):
                print("DEBUG: Force chunking due to size error.")
                pass 
            else:
                raise e

    print("DEBUG: File too large or failed, starting chunked transcription...")
    audio_clip = AudioFileClip(str(audio_path))
    duration = audio_clip.duration
    chunk_duration = 300 
    
    transcripts = []
    start = 0
    chunk_idx = 0
    
    while start < duration:
        end = min(start + chunk_duration, duration)
        chunk_path = audio_path.parent / f"chunk_{chunk_idx}.mp3"
        print(f"DEBUG: Processing chunk {chunk_idx} ({start}s to {end}s)")
        
        try:
            subclip = audio_clip.subclip(start, end)
            subclip.write_audiofile(
                str(chunk_path), 
                bitrate="64k",
                verbose=False,
                logger=None
            )
            
            with open(chunk_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    file=(chunk_path.name, f.read()),
                    model=TRANSCRIPTION_MODEL,
                    response_format="text",
                    language="en",
                    temperature=0.0,
                )
            transcripts.append(result if isinstance(result, str) else result.text)
        except Exception as e:
            print(f"ERROR on chunk {chunk_idx}: {e}")
            # Continue to next chunk or raise? Let's try to get what we can.
            transcripts.append(f"[Error transcribing segment {start}-{end}]")
        finally:
            if chunk_path.exists():
                chunk_path.unlink()
        
        start += chunk_duration
        chunk_idx += 1
    
    audio_clip.close()
    return " ".join(transcripts)


NOTES_SYSTEM_PROMPT = """You are an expert meeting analyst and executive assistant AI.
Your job is to transform raw meeting transcripts into structured, actionable intelligence.

Always respond with a valid JSON object following this exact schema:
{
  "title": "Auto-detected or inferred meeting title",
  "summary": "2-3 sentence executive summary",
  "key_topics": ["topic1", "topic2", "topic3"],
  "action_items": [
    {
      "id": 1,
      "task": "Clear, specific task description",
      "assignee": "Person responsible or 'TBD'",
      "priority": "High|Medium|Low",
      "deadline": "Suggested deadline (e.g. 'By Friday', 'Next sprint', 'Within 48h')",
      "category": "Development|Design|Testing|Communication|Planning|Research|Finance|HR|Other",
      "status": "Pending"
    }
  ],
  "decisions": [
    {
      "id": 1,
      "decision": "What was decided",
      "impact": "Brief impact statement",
      "made_by": "Who decided or 'Group'"
    }
  ],
  "key_insights": ["insight1", "insight2"],
  "next_steps": "Paragraph describing overall next steps",
  "meeting_effectiveness": "High|Medium|Low",
  "sentiment": "Positive|Neutral|Negative|Mixed",
  "follow_up_required": true,
  "participants": ["name1", "name2"]
}

Rules:
- Extract ALL action items including implied ones
- High priority = urgent/blocking; Medium = important; Low = nice-to-have
- Suggest realistic deadlines based on context
- Identify all speakers mentioned by name as participants
- Return ONLY valid JSON, no markdown fences, no extra text"""


def generate_notes_gemini(transcript: str, title: str = "") -> dict:
    """Fallback: Use Gemini 1.5 Flash for analysis."""
    print("DEBUG: Using Gemini for analysis...")
    if not gemini_model:
        raise RuntimeError("Gemini model not initialized.")
    
    prompt = f"{NOTES_SYSTEM_PROMPT}\n\nMeeting Title: {title}\n\nTranscript:\n{transcript}"
    
    response = gemini_model.generate_content(prompt)
    text = response.text.strip()
    
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def generate_notes(transcript: str, title: str = "") -> dict:
    if len(transcript) > 40000 and gemini_model:
        return generate_notes_gemini(transcript, title)

    try:
        print("DEBUG: Using Groq for analysis...")
        user_msg = f"""Meeting Title: {title or 'Auto-detect from transcript'}
Raw Transcript:
{transcript}
Analyze this transcript and extract all structured insights."""

        completion = client.chat.completions.create(
            model=NOTES_MODEL,
            messages=[
                {"role": "system", "content": NOTES_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=4000,
        )
        text = completion.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception as e:
        print(f"DEBUG: Groq analysis failed: {e}")
        if gemini_model:
            return generate_notes_gemini(transcript, title)
        raise e



@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models": {"transcription": TRANSCRIPTION_MODEL, "notes": NOTES_MODEL},
    }


@app.post("/api/process")
async def process_recording(
    file: UploadFile = File(...),
    title: str = Form(default=""),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    session_id = uuid.uuid4().hex
    session_dir = TEMP_DIR / session_id
    session_dir.mkdir()

    try:
        input_path = session_dir / f"input{ext}"
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        audio_path = session_dir / "audio.mp3"
        extract_audio(input_path, audio_path)

        transcript = transcribe_audio(audio_path)
        if not transcript.strip():
            raise HTTPException(status_code=422, detail="No speech detected in recording.")

        notes = generate_notes(transcript, title)
        notes["transcript"] = transcript
        notes["file_name"] = file.filename
        notes["session_id"] = session_id

        return JSONResponse(content=notes)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(session_dir, ignore_errors=True)


@app.post("/api/analyze-text")
async def analyze_text(request: dict):
    """Fallback: analyze raw transcript text directly."""
    transcript = request.get("transcript", "").strip()
    title = request.get("title", "")
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript text is required.")
    notes = generate_notes(transcript, title)
    notes["transcript"] = transcript
    return JSONResponse(content=notes)


@app.get("/")
async def serve_frontend():
    return FileResponse(str(Path(__file__).parent / "index.html"))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
