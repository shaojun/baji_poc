
from io import BytesIO
import os
import re
from time import time
import uuid
import json
import base64
import asyncio
import tempfile
import threading
from pathlib import Path
from typing import Optional

import requests as _requests_lib
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
from openai import OpenAI
import dashscope
from dashscope import VideoSynthesis
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
dashscope.api_key = DASHSCOPE_API_KEY

VIDEOS_DIR = Path("character_videos")
STATIC_DIR = Path("static")
BACKGROUNDS_DIR = Path("backgrounds")
VIDEOS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
BACKGROUNDS_DIR.mkdir(exist_ok=True)

EXPRESSIONS = ["开心", "伤心"]
POSES = ["站立"]

I2V_TASKS_FILE = BACKGROUNDS_DIR / "i2v_tasks.json"       # persisted task registry

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
PERSONALITY_DEFAULT = "你是一个亲切、聪明的AI伴侣，名叫巴吉（BAJI）。请用自然、温暖的中文与用户交流。"

# Mandatory rules appended to every system prompt — never exposed to end users.
_SYSTEM_PROMPT_MANDATORY = """

【重要规则】每次回复时，必须在回复的第一行加上且仅加上以下格式的元数据标签：
[META:expr=XXX,pose=YYY]

- expr 从以下几项中选一个最符合当前情绪的：开心 | 生气 | 伤心 | 害怕 | 平静
- pose 从以下几项中选一个最符合当前动作的：站立 |  跳舞 | seduce  

示例：
[META:expr=开心,pose=站立]
你好！很高兴认识你～有什么我可以帮你的吗？

注意：
1. 第一行必须是且仅是 [META:...] 标签，换行后才是正文
2. 不要向用户提及这个标签"""

PERSONALITY_FILE = Path("personality.json")
VOICE_DEFAULT = "longyingxiao"

def _load_config() -> dict:
    if PERSONALITY_FILE.exists():
        try:
            return json.loads(PERSONALITY_FILE.read_text())
        except Exception:
            pass
    return {}

def _save_config(**updates) -> None:
    config = _load_config()
    config.update(updates)
    PERSONALITY_FILE.write_text(json.dumps(config, ensure_ascii=False, indent=2))

def _load_personality() -> str:
    return _load_config().get("personality", PERSONALITY_DEFAULT)

def _save_personality(text: str) -> None:
    _save_config(personality=text)

def _load_voice() -> str:
    return _load_config().get("voice", VOICE_DEFAULT)

def _save_voice(voice: str) -> None:
    _save_config(voice=voice)

def _build_system_prompt() -> str:
    return _load_personality() + _SYSTEM_PROMPT_MANDATORY

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="BAJI AI Companion")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/backgrounds", StaticFiles(directory=str(BACKGROUNDS_DIR)), name="backgrounds")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# I2V task persistence helpers
# ---------------------------------------------------------------------------
def _load_i2v_tasks() -> dict:
    """Load persisted i2v task registry (task_id -> metadata)."""
    if I2V_TASKS_FILE.exists():
        try:
            return json.loads(I2V_TASKS_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_i2v_tasks(tasks: dict) -> None:
    I2V_TASKS_FILE.write_text(json.dumps(tasks, ensure_ascii=False, indent=2))


def _upsert_i2v_task(task_id: str, data: dict) -> None:
    tasks = _load_i2v_tasks()
    existing = tasks.get(task_id, {})
    existing.update(data)
    existing["task_id"] = task_id
    tasks[task_id] = existing
    _save_i2v_tasks(tasks)


# ---------------------------------------------------------------------------
# Aliyun temporary-URL upload helper
# ---------------------------------------------------------------------------
def _normalize_image(file_bytes: bytes, filename: str) -> tuple[bytes, str]:
    """
    Ensure image is RGB JPEG (no alpha channel).
    The DashScope API rejects PNG with transparency and prefers JPEG.
    Returns (jpeg_bytes, new_filename).
    """
    import io
    from PIL import Image
    img = Image.open(io.BytesIO(file_bytes))
    if img.mode in ("RGBA", "LA", "P"):
        # Composite alpha onto white background
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        bg.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    new_filename = filename.rsplit(".", 1)[0] + ".jpg"
    print(f"[normalize_image] {filename} → {new_filename} ({img.size[0]}x{img.size[1]}, mode was {img.mode})")
    return buf.getvalue(), new_filename


def _upload_file_to_oss_temp(file_bytes: bytes, filename: str, model_name: str = "wan2.7-i2v", api_key: str = "") -> str:
    """Upload bytes to Aliyun temp OSS and return oss:// URL."""
    _key = api_key or DASHSCOPE_API_KEY
    print(f"[oss_upload] Getting upload policy for {filename} ({len(file_bytes)} bytes) model={model_name}")
    # Step 1: get upload policy
    policy_resp = _requests_lib.get(
        "https://dashscope.aliyuncs.com/api/v1/uploads",
        headers={
            "Authorization": f"Bearer {_key}",
            "Content-Type": "application/json",
        },
        params={"action": "getPolicy", "model": model_name},
        timeout=30,
    )
    if policy_resp.status_code != 200:
        print(f"[oss_upload] Failed to get upload policy: {policy_resp.text}")
        raise RuntimeError(f"Failed to get upload policy: {policy_resp.text}")
    policy_data = policy_resp.json()["data"]
    print(f"[oss_upload] Policy acquired, upload_host={policy_data.get('upload_host')}")

    # Step 2: upload to OSS
    key = f"{policy_data['upload_dir']}/{filename}"
    form_fields = {
        "OSSAccessKeyId": (None, policy_data["oss_access_key_id"]),
        "Signature":      (None, policy_data["signature"]),
        "policy":         (None, policy_data["policy"]),
        "x-oss-object-acl":      (None, policy_data["x_oss_object_acl"]),
        "x-oss-forbid-overwrite": (None, policy_data["x_oss_forbid_overwrite"]),
        "key":                   (None, key),
        "success_action_status": (None, "200"),
        "file":                  (filename, file_bytes),
    }
    upload_resp = _requests_lib.post(
        policy_data["upload_host"],
        files=form_fields,
        timeout=120,
    )
    if upload_resp.status_code != 200:
        print(f"[oss_upload] OSS upload failed: {upload_resp.text}")
        raise RuntimeError(f"Failed to upload file to OSS: {upload_resp.text}")

    upload_host = policy_data["upload_host"].rstrip("/")
    oss_url = f"oss://{key}"
    print(f"[oss_upload] Upload succeeded: {oss_url}")
    return oss_url


def _parse_secs(filename: str) -> float:
    """Extract seconds from filename like {expression}-{pose}-{N}秒.mp4"""
    m = re.search(r'(\d+)秒', filename)
    return float(m.group(1)) if m else 0.0


def parse_meta(text: str) -> tuple[str, str, str]:
    """Extract [META:expr=X,pose=Y] from first line and return (clean_text, expr, pose)."""
    match = re.search(r'\[META:expr=(\w+)[,\uff0c]pose=(\w+)\]', text)
    if match:
        expr = match.group(1) if match.group(
            1) in EXPRESSIONS else EXPRESSIONS[0]
        pose = match.group(2) if match.group(2) in POSES else POSES[0]
        clean = text[match.end():].strip()
        return clean, expr, pose
    return text.strip(), EXPRESSIONS[0], POSES[0]


def find_match_video(char_id: str, speech_secs: float, expression: str, pose: str) -> Optional[str]:
    """Find best matching video by closest duration, then expression/pose fallback."""
    fallback_expr = EXPRESSIONS[0]
    fallback_pose = POSES[0]
    for expr, ps in [(expression, pose), (expression, fallback_pose), (fallback_expr, fallback_pose)]:
        candidates = list(VIDEOS_DIR.glob(f"{expr}-{ps}-*秒.mp4"))
        if candidates:
            file_name = min(candidates, key=lambda f: abs(
                _parse_secs(f.name) - speech_secs)).name
            print(f"[find_match_video] found match: {file_name}")
            return file_name
    # fallback: any mp4
    files = list(VIDEOS_DIR.glob("*.mp4"))
    if files:
        file_name = min(files, key=lambda f: abs(
            _parse_secs(f.name) - speech_secs)).name
        print(f"[find_match_video] no exact match, falling back to {file_name}")
        return file_name
    return None


# ---------------------------------------------------------------------------
# Routes — pages
# ---------------------------------------------------------------------------
@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------
@app.get("/api/status")
async def status():
    # Ready as long as any mp4 exists in character_videos/
    if list(VIDEOS_DIR.glob("*.mp4")):
        return {"character_id": "default", "videos_ready": True}
    return {"character_id": None, "videos_ready": False}


@app.get("/api/personality")
async def get_personality():
    return {"personality": _load_personality()}

@app.post("/api/personality")
async def set_personality(payload: dict):
    text = (payload.get("personality") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Personality text cannot be empty")
    _save_personality(text)
    return {"ok": True}

@app.get("/api/voice")
async def get_voice():
    return {"voice": _load_voice()}

@app.post("/api/voice")
async def set_voice(payload: dict):
    voice = (payload.get("voice") or "").strip()
    if not voice:
        raise HTTPException(status_code=400, detail="Voice cannot be empty")
    _save_voice(voice)
    return {"ok": True}



async def chat(audio: UploadFile = File(...)):
    """
    Full pipeline: audio → ASR → LLM → TTS → JSON response.
    Returns transcript, AI text, expression/pose metadata, video filename, audio base64.
    """
    if not list(VIDEOS_DIR.glob("*.mp4")):
        raise HTTPException(
            status_code=400, detail="Please upload a character image first")

    char_id = "default"
    audio_bytes = await audio.read()

    # Determine suffix from content-type
    ct = audio.content_type or ""
    if "wav" in ct:
        suffix = ".wav"
    elif "ogg" in ct:
        suffix = ".ogg"
    elif "mp4" in ct or "m4a" in ct:
        suffix = ".m4a"
    else:
        suffix = ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        transcript = await _asr(tmp_path)
        llm_raw = _llm(transcript)
        clean_text, expression, pose = parse_meta(llm_raw)

        speech_secs_est = max(1.0, len(clean_text) / 3.0)

        video_file = find_match_video(char_id, speech_secs_est, expression, pose)
        print(
            f"[chat] transcript={transcript} | clean_text={clean_text} | expression={expression} | pose={pose} | video_file={video_file}")
        audio_b64, speech_secs = await _tts(clean_text)
        if speech_secs == 0.0:
            speech_secs = speech_secs_est
        print(f"[chat] actual speech_secs={speech_secs:.2f}")

        return {
            "transcript":     transcript,
            "response_text":  clean_text,
            "expression":     expression,
            "pose":           pose,
            "speech_seconds": round(speech_secs, 1),
            "video_filename": video_file,
            "audio_base64":   audio_b64,
        }
    finally:
        os.unlink(tmp_path)


@app.post("/api/chat-text")
async def chat_text(payload: dict):
    """Text-only chat endpoint (skips ASR). Used by the text input fallback."""
    if not list(VIDEOS_DIR.glob("*.mp4")):
        raise HTTPException(
            status_code=400, detail="Please upload a character image first")

    user_text = (payload.get("text") or "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Empty text")

    char_id = "default"

    perf_llm_starting_time = time()
    llm_raw_text = _llm(user_text)
    print(
        f"[chat-text] LLM response time: {time() - perf_llm_starting_time:.2f}s")
    clean_text, expression, pose = parse_meta(llm_raw_text)

    speech_secs_rough_est = max(1.0, len(clean_text) / 3.0)
    perf_tts_starting_time = time()
    audio_b64, speech_secs_accurate_est = await _tts(clean_text)
    print(
        f"[chat-text] TTS response time: {time() - perf_tts_starting_time:.2f}s | actual speech_secs={speech_secs_accurate_est:.2f}")
    if speech_secs_accurate_est == 0.0:
        print(f"why the hell is TTS duration zero?? falling back to rough estimate")
        speech_secs_accurate_est = speech_secs_rough_est
    # 从实际测试中的经验推导, 实际人耳听到的声音总是要比估算出来的短0.3s
    speech_secs_accurate_est = speech_secs_accurate_est - 0.3
    video_file = find_match_video(char_id, speech_secs_accurate_est, expression, pose)
    print(
        f"[chat-text] user_text={user_text} | clean_text={clean_text} | expression={expression} | pose={pose} | video_file={video_file}")
    

    return {
        "transcript":     user_text,
        "response_text":  clean_text,
        "expression":     expression,
        "pose":           pose,
        "speech_seconds": round(speech_secs_accurate_est, 1),
        "video_filename": video_file,
        "audio_base64":   audio_b64,
    }


@app.get("/api/videos/{filename}")
async def get_video(filename: str):
    # Prevent path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    filepath = VIDEOS_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(filepath, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/api/character")
async def get_character():
    orig = VIDEOS_DIR / "default_original.jpg"
    if orig.exists():
        return FileResponse(orig)
    raise HTTPException(status_code=404, detail="No character uploaded")


# ---------------------------------------------------------------------------
# Routes — Image-to-Video (i2v)
# ---------------------------------------------------------------------------

@app.post("/api/i2v/submit")
async def i2v_submit(
    first_frame: UploadFile = File(...),
    last_frame:  UploadFile = File(None),
    prompt:      str        = Form(""),
    duration:    int        = Form(10),
    resolution:  str        = Form("720P"),
    api_key:     str        = Form(""),
):
    """
    Upload 1 or 2 frame images, kick off an async wan2.7-i2v task and return task_id immediately.
    The task is persisted to disk so it survives a server restart.
    """
    effective_api_key = api_key.strip()
    if not effective_api_key:
        raise HTTPException(status_code=500, detail="DASHSCOPE_API_KEY must be explicitly provided")

    duration = max(2, min(15, duration))

    # Resolution → size string
    size_map = {
        "480P":  "832*480",
        "720P":  "1280*720",
        "1080P": "1920*1080",
    }
    size = size_map.get(resolution.upper(), "1280*720")

    loop = asyncio.get_event_loop()

    print(f"[i2v/submit] Starting — duration={duration}s resolution={resolution} size={size} prompt={prompt!r}")

    # --- Upload first frame ---
    first_bytes = await first_frame.read()
    first_name  = first_frame.filename or "first_frame.jpg"
    first_bytes, first_name = _normalize_image(first_bytes, first_name)
    print(f"[i2v/submit] Uploading first frame: {first_name} ({len(first_bytes)} bytes)")
    _ff_name = f"ff_{uuid.uuid4().hex[:8]}_{first_name}"
    first_oss   = await loop.run_in_executor(
        None, lambda: _upload_file_to_oss_temp(first_bytes, _ff_name, api_key=effective_api_key)
    )
    print(f"[i2v/submit] First frame uploaded: {first_oss}")

    # --- Upload last frame (optional) ---
    last_oss = None
    if last_frame and last_frame.filename:
        last_bytes = await last_frame.read()
        last_name  = last_frame.filename or "last_frame.jpg"
        last_bytes, last_name = _normalize_image(last_bytes, last_name)
        print(f"[i2v/submit] Uploading last frame: {last_name} ({len(last_bytes)} bytes)")
        _lf_name = f"lf_{uuid.uuid4().hex[:8]}_{last_name}"
        last_oss   = await loop.run_in_executor(
            None, lambda: _upload_file_to_oss_temp(last_bytes, _lf_name, api_key=effective_api_key)
        )
        print(f"[i2v/submit] Last frame uploaded: {last_oss}")

    # --- Submit async task via direct HTTP (SDK uses wrong field names) ---
    media = [{"type": "first_frame", "url": first_oss}]
    if last_oss:
        media.append({"type": "last_frame", "url": last_oss})

    body = {
        "model": "wan2.7-i2v",
        "input": {
            "prompt": prompt or "",
            "media": media,
        },
        "parameters": {
            "resolution": resolution.upper(),
            "duration": duration,
        },
    }
    print(f"[i2v/submit] POST to video-synthesis API with body={body}")

    def _submit():
        return _requests_lib.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis",
            headers={
                "Authorization": f"Bearer {effective_api_key}",
                "Content-Type": "application/json",
                "X-DashScope-Async": "enable",
                "X-DashScope-OssResourceResolve": "enable",
            },
            json=body,
            timeout=60,
        )

    resp = await loop.run_in_executor(None, _submit)
    resp_json = resp.json()
    print(f"[i2v/submit] API response status={resp.status_code} body={resp_json}")

    if resp.status_code not in (200, 201):
        code = resp_json.get("code", resp.status_code)
        message = resp_json.get("message", resp.text)
        print(f"[i2v/submit] Submit failed: code={code} message={message}")
        raise HTTPException(status_code=502, detail=f"i2v submit failed: {code} – {message}")

    output = resp_json.get("output", {})
    task_id = output.get("task_id")
    task_status = output.get("task_status", "PENDING")
    print(f"[i2v/submit] Task created: task_id={task_id} status={task_status}")

    # Persist
    _upsert_i2v_task(task_id, {
        "status":     task_status,
        "prompt":     prompt,
        "duration":   duration,
        "resolution": resolution,
        "created_at": time(),
        "video_url":  None,
        "local_file": None,
    })

    return {"task_id": task_id, "status": task_status}


@app.get("/api/i2v/tasks")
async def i2v_list_tasks():
    """Return all known i2v tasks (persisted)."""
    tasks = _load_i2v_tasks()
    return {"tasks": list(tasks.values())}


@app.get("/api/i2v/task/{task_id}")
async def i2v_get_task(task_id: str):
    """Poll the upstream API for task status; update persistence; return current state."""
    tasks = _load_i2v_tasks()
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task_meta = tasks[task_id]

    # If already finished, return cached result
    if task_meta.get("status") in ("SUCCEEDED", "FAILED", "CANCELED"):
        print(f"[i2v/poll] task_id={task_id} already in terminal state={task_meta['status']}, returning cached result")
        return task_meta

    loop = asyncio.get_event_loop()

    print(f"[i2v/poll] Polling upstream for task_id={task_id}")

    def _fetch():
        return _requests_lib.get(
            f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}",
            headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}"},
            timeout=30,
        )

    resp = await loop.run_in_executor(None, _fetch)
    resp_json = resp.json()
    print(f"[i2v/poll] task_id={task_id} raw response: {resp_json}")

    if resp.status_code != 200:
        message = resp_json.get("message", resp.text)
        print(f"[i2v/poll] Upstream fetch error for task_id={task_id}: status={resp.status_code} message={message}")
        raise HTTPException(status_code=502, detail=f"Fetch failed: {message}")

    output = resp_json.get("output", {})
    new_status = output.get("task_status", "UNKNOWN")
    print(f"[i2v/poll] task_id={task_id} status={new_status}")
    updates: dict = {"status": new_status}

    if new_status == "SUCCEEDED":
        video_url = output.get("video_url")
        updates["video_url"] = video_url
        print(f"[i2v/poll] task_id={task_id} SUCCEEDED, video_url={video_url}")

        # Download and persist locally
        if video_url:
            local_filename = f"bg_{task_id[:8]}_{int(time())}.mp4"
            local_path = BACKGROUNDS_DIR / local_filename
            print(f"[i2v/poll] Downloading video to {local_path}")
            try:
                def _download():
                    r = _requests_lib.get(video_url, timeout=300, stream=True)
                    r.raise_for_status()
                    with open(local_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=65536):
                            f.write(chunk)
                await loop.run_in_executor(None, _download)
                updates["local_file"] = local_filename
                print(f"[i2v/poll] Download complete: {local_filename}")
            except Exception as e:
                updates["download_error"] = str(e)
                print(f"[i2v/poll] Download failed for task_id={task_id}: {e}")

    elif new_status == "FAILED":
        error_code = output.get("code")
        error_message = output.get("message")
        updates["error_code"] = error_code
        updates["error_message"] = error_message
        print(f"[i2v/poll] task_id={task_id} FAILED — code={error_code} message={error_message}")

    _upsert_i2v_task(task_id, updates)
    return _load_i2v_tasks()[task_id]


@app.post("/api/i2v/task/{task_id}/cancel")
async def i2v_cancel_task(task_id: str):
    """Cancel a PENDING async i2v task."""
    tasks = _load_i2v_tasks()
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    loop = asyncio.get_event_loop()
    print(f"[i2v/cancel] Cancelling task_id={task_id}")

    def _cancel():
        return _requests_lib.post(
            f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}/cancel",
            headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}"},
            timeout=30,
        )

    resp = await loop.run_in_executor(None, _cancel)
    resp_json = resp.json()
    print(f"[i2v/cancel] task_id={task_id} response status={resp.status_code} body={resp_json}")

    if resp.status_code != 200:
        code = resp_json.get("code", resp.status_code)
        message = resp_json.get("message", resp.text)
        print(f"[i2v/cancel] Cancel failed: code={code} message={message}")
        raise HTTPException(status_code=502, detail=f"Cancel failed: {code} – {message}")

    _upsert_i2v_task(task_id, {"status": "CANCELED"})
    print(f"[i2v/cancel] task_id={task_id} marked as CANCELED")
    return {"task_id": task_id, "status": "CANCELED"}


@app.get("/api/backgrounds")
async def list_backgrounds():
    """List all locally saved background mp4 files."""
    files = sorted(BACKGROUNDS_DIR.glob("*.mp4"), key=lambda f: f.stat().st_mtime, reverse=True)
    return {"backgrounds": [f.name for f in files]}


@app.post("/api/backgrounds/set-idle")
async def set_idle_background(payload: dict):
    """Copy (symlink) the chosen background mp4 to character_videos/idle.mp4."""
    filename = (payload.get("filename") or "").strip()
    if not filename or ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    src = BACKGROUNDS_DIR / filename
    if not src.exists():
        raise HTTPException(status_code=404, detail="Background file not found")

    dst = VIDEOS_DIR / "idle.mp4"
    # backup old idle.mp4 if exists
    if dst.exists():
        backup = VIDEOS_DIR / f"idle_backup_{int(time())}.mp4"
        dst.rename(backup)
        print(f"[set_idle_background] Backed up existing idle.mp4 to {backup}")
    import shutil
    shutil.copy2(src, dst)
    return {"success": True, "idle": "idle.mp4"}


# ---------------------------------------------------------------------------
# Routes — File browser
# ---------------------------------------------------------------------------
_ALLOWED_FILE_FOLDERS: dict[str, Path] = {"character_videos": VIDEOS_DIR}


@app.get("/api/files/{folder}")
async def list_folder_files(folder: str):
    """List files in an allowed folder."""
    folder_path = _ALLOWED_FILE_FOLDERS.get(folder)
    if folder_path is None:
        raise HTTPException(status_code=400, detail="Invalid folder")
    if not folder_path.exists():
        return {"files": []}
    files = []
    for f in sorted(folder_path.iterdir(), key=lambda x: x.name):
        if f.is_file() and not f.name.startswith('.'):
            stat = f.stat()
            files.append({"name": f.name, "size": stat.st_size, "mtime": stat.st_mtime})
    return {"files": files}


@app.post("/api/files/character_videos/upload")
async def upload_to_character_videos(file: UploadFile = File(...)):
    """Upload a file directly into the character_videos folder."""
    filename = Path(file.filename or "upload").name  # strip any directory components
    if not filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    dest = VIDEOS_DIR / filename
    content = await file.read()
    dest.write_bytes(content)
    return {"success": True, "filename": filename, "size": len(content)}


@app.delete("/api/files/{folder}/{filename}")
async def delete_folder_file(folder: str, filename: str):
    """Delete a file from an allowed folder."""
    folder_path = _ALLOWED_FILE_FOLDERS.get(folder)
    if folder_path is None:
        raise HTTPException(status_code=400, detail="Invalid folder")
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    file_path = folder_path / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    file_path.unlink()
    return {"success": True}


@app.post("/api/files/character_videos/{filename}/set-idle")
async def set_cv_file_as_idle(filename: str):
    """Copy a video file from character_videos to idle.mp4."""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    src = VIDEOS_DIR / filename
    if not src.exists():
        raise HTTPException(status_code=404, detail="File not found")
    dst = VIDEOS_DIR / "idle.mp4"
    if dst.resolve() == src.resolve():
        return {"success": True, "idle": "idle.mp4"}
    if dst.exists():
        backup = VIDEOS_DIR / f"idle_backup_{int(time())}.mp4"
        dst.rename(backup)
        print(f"[set_cv_idle] Backed up existing idle.mp4 to {backup}")
    import shutil
    shutil.copy2(src, dst)
    return {"success": True, "idle": "idle.mp4"}


# ---------------------------------------------------------------------------
# AI pipeline helpers (run blocking SDK calls in thread executor)
# ---------------------------------------------------------------------------
def _to_wav_16k(src_path: str) -> str:
    """Convert any audio file to 16000Hz mono WAV using ffmpeg. Returns new tmp path."""
    out = src_path + "_converted.wav"
    import subprocess
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", src_path,
         "-ar", "16000", "-ac", "1", "-f", "wav", out],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg conversion failed: {result.stderr.decode()[-300:]}")
    return out


async def _asr(audio_path: str) -> str:
    loop = asyncio.get_event_loop()

    def _call():
        converted = _to_wav_16k(audio_path)
        fmt, sample_rate = "wav", 16000

        final_texts: list[str] = []
        done = threading.Event()
        errors: list[str] = []

        class _Cb(RecognitionCallback):
            def on_complete(self):
                done.set()

            def on_error(self, result):
                errors.append(str(result))
                done.set()

            def on_event(self, result: RecognitionResult):
                if not (result.output and result.output.get("sentence")):
                    return
                s = result.output["sentence"]
                # Only take the finalised sentence
                if s.get("sentence_end"):
                    final_texts.append(s.get("text", ""))

        rec = Recognition(
            model="paraformer-realtime-v2",
            format=fmt,
            sample_rate=sample_rate,
            callback=_Cb(),
        )
        rec.start()
        try:
            with open(converted, "rb") as f:
                while chunk := f.read(8192):
                    rec.send_audio_frame(chunk)
        finally:
            rec.stop()
            try:
                os.unlink(converted)
            except OSError:
                pass

        done.wait(timeout=30)

        if errors:
            raise RuntimeError(f"ASR service error: {errors[0]}")

        return "".join(final_texts).strip()

    try:
        text = await loop.run_in_executor(None, _call)
        return text or "（未能识别语音内容）"
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ASR error: {e}")


def _llm(user_text: str) -> str:
    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    resp = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user",   "content": user_text},
        ], extra_body={"enable_thinking": False},
    )
    return resp.choices[0].message.content


async def _tts(text: str) -> tuple[Optional[str], float]:
    """Return (base64-encoded MP3 audio or None, actual_duration_secs).

    Duration is computed from the raw byte count of the CBR 256 kbps stream:
    bytes / (256_000 bits/s ÷ 8 bits/byte) = bytes / 32_000.
    """
    loop = asyncio.get_event_loop()
    from mutagen.mp3 import MP3  # 替换pydub，无依赖不报错

    def get_audio_duration(audio_bytes: bytes) -> float:
        """
        从MP3二进制数据计算精准播放时长（秒）
        兼容Python3.13，无audioop依赖，零报错
        """
        if not audio_bytes:
            return 0.0
        # 直接解析二进制音频流，无需文件
        audio = MP3(BytesIO(audio_bytes))
        return audio.info.length  # 精准时长（秒）
    def _call():
        synth = SpeechSynthesizer(
            model="cosyvoice-v2",
            voice=_load_voice(),
            format=AudioFormat.MP3_22050HZ_MONO_256KBPS,
        )
        audio_bytes = synth.call(text)
        if audio_bytes:
            duration = get_audio_duration(audio_bytes)
            return base64.b64encode(audio_bytes).decode(), duration
        return None, 0.0

    try:
        return await loop.run_in_executor(None, _call)
    except Exception as e:
        print(f"[TTS] error: {e}")
        return None, 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8100, reload=True)
