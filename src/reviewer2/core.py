"""Gemini API client with upload caching, retry logic, and PDF tools."""

from __future__ import annotations

import atexit
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime

from pathlib import Path

from google import genai
from google.genai import types
from google.genai.types import HarmBlockThreshold, HarmCategory, SafetySetting
from pypdf import PdfReader, PdfWriter

from reviewer2.paths import prompts_dir

# Per-call usage records. Appended by call_gemini, consumed by helpers.calculate_cost.
USAGE_LOG: list[dict] = []

# { local_file_path: uploaded_gemini_file_object }. Reset with cleanup_resources.
_FILE_CACHE: dict[str, object] = {}

MODELS = {
    "flash_lite": "gemini-2.5-flash-lite",
    "flash_lite_3": "gemini-3.1-flash-lite-preview",
    "flash_2_5": "gemini-2.5-flash",
    "flash_3": "gemini-3-flash-preview",
    "pro_2_5": "gemini-2.5-pro",
    "pro_3": "gemini-3-pro-preview",
    "pro_3_1": "gemini-3.1-pro-preview",
}


def get_or_upload_file(client, file_path, force_upload=False):
    """Upload a PDF to Gemini once per session; reuse the handle after that."""
    global _FILE_CACHE

    if not force_upload and file_path in _FILE_CACHE:
        return _FILE_CACHE[file_path]

    print(f"    ↳ 📤 Uploading NEW file to Gemini: {os.path.basename(file_path)}...")
    try:
        uploaded_file = client.files.upload(file=file_path)
        _FILE_CACHE[file_path] = uploaded_file
        print(f"    ✓ Upload complete: {uploaded_file.name} (Cached)")
        return uploaded_file
    except Exception as e:
        if file_path in _FILE_CACHE:
            del _FILE_CACHE[file_path]
        raise IOError(f"Error uploading PDF '{file_path}': {e}")


def cleanup_resources():
    """Delete every remote file uploaded during this session."""
    global _FILE_CACHE
    if not _FILE_CACHE:
        return

    print("\n🧹 Cleaning up Gemini resources...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return

    try:
        client = genai.Client(api_key=api_key)
        for _path, file_obj in list(_FILE_CACHE.items()):
            try:
                client.files.delete(name=file_obj.name)
                print(f"  ✓ Deleted remote file: {file_obj.name}")
            except Exception as e:
                if "404" not in str(e):
                    print(f"  ⚠ Failed to delete {file_obj.name}: {e}")
        _FILE_CACHE.clear()
    except Exception as e:
        print(f"  ⚠ Cleanup initialization failed: {e}")


atexit.register(cleanup_resources)


def call_gemini(
    prompt=None,
    pdf_file_path=None,
    model_type="flash_lite",
    temperature=0.1,
    thinking_level=None,
    thinking_budget=None,
    media_resolution="MEDIA_RESOLUTION_MEDIUM",
    system_instruction=None,
    max_retries=10,
    retry_forever_on_rate_limit=True,
    step=None,
    use_search=False,
    max_output_tokens=None,
):
    """Call Gemini with retries, PDF caching, and structured thinking support.

    ``GEMINI_MODEL_OVERRIDE`` in the environment forces every call to a single
    model, useful for cheap smoke runs against Flash Lite.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment")

    override = os.environ.get("GEMINI_MODEL_OVERRIDE")
    if override:
        model_type = override

    model_name = MODELS.get(model_type, model_type)
    if not model_name:
        print(f"  ⚠  Warning: Unknown model_type '{model_type}', falling back to flash_lite")
        model_name = MODELS["flash_lite"]

    http_options = types.HttpOptions(client_args={"timeout": None})
    client = genai.Client(api_key=api_key, http_options=http_options)

    safety_settings = [
        SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.BLOCK_NONE),
        SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_NONE),
        SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_NONE),
        SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.BLOCK_NONE),
    ]

    thinking_cfg = None
    model_supports_thinking = (
        "pro" in model_type or "2.5" in model_type or "flash" in model_type or "3" in model_name
    )
    if model_supports_thinking:
        if thinking_budget is not None:
            thinking_cfg = types.ThinkingConfig(thinking_budget=int(thinking_budget))
        elif thinking_level:
            thinking_cfg = types.ThinkingConfig(thinking_level=thinking_level, include_thoughts=False)
        else:
            thinking_cfg = types.ThinkingConfig(thinking_budget=-1)

    tools = None
    if use_search:
        tools = [types.Tool(google_search=types.GoogleSearch())]

    attempt = 0
    base_delay = 5
    max_delay = 300
    has_forced_reupload = False

    while True:
        attempt += 1

        parts = []
        if pdf_file_path:
            force = attempt > 1 and has_forced_reupload
            try:
                current_uploaded_file = get_or_upload_file(client, pdf_file_path, force_upload=force)
                has_forced_reupload = False
                parts.append(types.Part.from_uri(
                    file_uri=current_uploaded_file.uri,
                    mime_type="application/pdf",
                ))
            except Exception as e:
                print(f"    ⚠  Upload failed inside retry loop: {e}")
                time.sleep(5)
                continue

        if prompt:
            parts.append(types.Part.from_text(text=prompt))

        contents = [types.Content(role="user", parts=parts)]

        formatted_system_instruction = None
        if system_instruction:
            formatted_system_instruction = [types.Part.from_text(text=system_instruction)]

        gen_config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=formatted_system_instruction,
            thinking_config=thinking_cfg,
            media_resolution=media_resolution,
            safety_settings=safety_settings,
            tools=tools,
            max_output_tokens=max_output_tokens,
        )

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=gen_config,
            )

            if response is None:
                raise ValueError("API returned None response object")

            usage_meta = getattr(response, "usage_metadata", None)
            if usage_meta:
                p_tok = getattr(usage_meta, "prompt_token_count", 0) or 0
                c_tok = getattr(usage_meta, "candidates_token_count", 0) or 0
                t_tok = getattr(usage_meta, "thoughts_token_count", 0) or 0
                USAGE_LOG.append({
                    "model_name": model_name,
                    "input_tokens": p_tok,
                    "output_tokens": c_tok + t_tok,
                    "thoughts_tokens": t_tok,
                    "timestamp": datetime.now().isoformat(),
                    "step": step or "unknown",
                })

            try:
                feedback = getattr(response, "prompt_feedback", None)
                if feedback and getattr(feedback, "block_reason", None):
                    error_msg = f"FATAL: Prompt blocked by filters: {feedback.block_reason}"
                    print(f"    ❌ {error_msg}")
                    raise RuntimeError(error_msg)

                candidates = getattr(response, "candidates", []) or []
                if not candidates:
                    raise ValueError("No candidates returned (Possible Safety Block or Empty Response)")

                first_cand = candidates[0]
                finish_reason = str(getattr(first_cand, "finish_reason", "UNKNOWN"))
                valid_reasons = ["STOP", "1", "FinishReason.STOP", "MAX_TOKENS", "2", "FinishReason.MAX_TOKENS"]

                if finish_reason not in valid_reasons:
                    safety_ratings = getattr(first_cand, "safety_ratings", "Unknown")
                    error_msg = f"FATAL: Generation stopped. Reason: {finish_reason}. Ratings: {safety_ratings}"
                    print(f"    ❌ {error_msg}")
                    raise RuntimeError(error_msg)

                if thinking_cfg:
                    final_text = []
                    content = getattr(first_cand, "content", None)
                    parts = getattr(content, "parts", None)
                    if parts is None:
                        if "MAX_TOKENS" in str(finish_reason):
                            raise ValueError(f"FATAL: Model hit MAX_TOKENS and returned no content. (Finish Reason: {finish_reason})")
                        raise ValueError(f"API returned malformed content: 'parts' is None (Finish Reason: {finish_reason})")
                    for part in parts:
                        if part.text and not getattr(part, "thought", False):
                            final_text.append(part.text)
                    result = "".join(final_text).strip()
                else:
                    if hasattr(response, "text") and response.text:
                        result = response.text
                    else:
                        content = getattr(first_cand, "content", None)
                        parts = getattr(content, "parts", None)
                        if parts and len(parts) > 0:
                            result = parts[0].text
                        else:
                            raise ValueError("Standard model returned no text parts.")

                if not result or not result.strip():
                    raise ValueError("API returned empty string")
                return result

            except Exception as e:
                if "FATAL" in str(e):
                    raise
                if "API returned malformed content" in str(e):
                    raise
                raise ValueError(f"Failed to extract text from response: {e}")

        except Exception as e:
            err_str = str(e)

            if "FATAL" in err_str:
                raise
            if "'NoneType' object is not iterable" in err_str:
                print("    ✗ FATAL ERROR: Malformed API response (NoneType iterable). Stopping.")
                raise

            if "403" in err_str and ("access the File" in err_str or "PERMISSION_DENIED" in err_str):
                print("    ⚠  File Permission/Access Error (403). Forcing re-upload on next attempt.")
                has_forced_reupload = True
                if pdf_file_path and pdf_file_path in _FILE_CACHE:
                    del _FILE_CACHE[pdf_file_path]
                time.sleep(2)
                if attempt >= max_retries:
                    raise RuntimeError(f"File access failed after {attempt} attempts: {err_str}")
                continue

            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)

            if "429" in err_str:
                print(f"    ↳ Rate Limited (429) on attempt {attempt}. Waiting {delay}s...")
                time.sleep(delay)
                if not retry_forever_on_rate_limit and attempt >= max_retries:
                    raise RuntimeError(f"Rate limit exceeded after {attempt} attempts: {err_str}")
                continue
            elif "Server disconnected" in err_str or "RemoteProtocolError" in err_str:
                print(f"    ⚠  Network Timeout (Attempt {attempt}/{max_retries}). Retrying in {delay}s...")
                time.sleep(delay)
            elif "500" in err_str or "503" in err_str or "502" in err_str:
                print(f"    ⚠  Server Error (Attempt {attempt}). Waiting {delay}s...")
                time.sleep(delay)
                if not retry_forever_on_rate_limit and attempt >= max_retries:
                    raise RuntimeError(f"Server error after {attempt} attempts: {err_str}")
                continue
            else:
                print(f"    ⚠  API Error (Attempt {attempt}/{max_retries}): {err_str[:200]}")
                time.sleep(min(delay, 30))

            if attempt >= max_retries:
                raise RuntimeError(f"Gemini API call failed after {attempt} attempts. Last error: {err_str}")


def save_output(content, filename, output_dir):
    if content is None:
        print(f"  ⚠  WARNING: content is None for {filename}. Creating empty file.")
        content = ""
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  ✓ Saved: {filename}")


def load_prompt(prompt_path):
    """Load a prompt file and inline the ``{{OUTPUT_FORMAT}}`` resource if referenced.

    A relative path of the form ``prompts/<name>`` resolves against the packaged
    prompts directory (overridable via ``REVIEWER2_PROMPTS_DIR``), so pipeline
    callers can keep using short literals regardless of cwd.
    """
    path = Path(prompt_path)
    if not path.is_absolute() and path.parts and path.parts[0] == "prompts":
        path = prompts_dir().joinpath(*path.parts[1:])

    if not path.exists():
        return f"[Error: Prompt file {path} missing]"

    text = path.read_text(encoding="utf-8")

    if "{{OUTPUT_FORMAT}}" in text:
        output_format_path = prompts_dir() / "resources" / "output_format.txt"
        if output_format_path.exists():
            text = text.replace("{{OUTPUT_FORMAT}}", output_format_path.read_text(encoding="utf-8"))
        else:
            text = text.replace("{{OUTPUT_FORMAT}}", "")
    return text


def sanitize_pdf_ghostscript(input_path):
    if os.path.getsize(input_path) == 0:
        return input_path

    fd, fixed_path = tempfile.mkstemp(suffix=".pdf", prefix="sanitized_")
    os.close(fd)

    gs_cmd = shutil.which("gs")
    if gs_cmd:
        cmd = [gs_cmd, "-o", fixed_path, "-sDEVICE=pdfwrite", "-dPDFSETTINGS=/prepress", "-dQUIET", input_path]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(fixed_path) and os.path.getsize(fixed_path) > 0:
                return fixed_path
        except subprocess.CalledProcessError:
            pass

    try:
        reader = PdfReader(input_path)
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        writer.write(fixed_path)
        return fixed_path
    except Exception:
        pass

    return input_path


def merge_pdfs_python(main_pdf, supplement_source, output_dir=None):
    if output_dir:
        output_path = os.path.join(output_dir, os.path.basename(main_pdf))
    else:
        output_path = main_pdf

    temp_path = output_path + ".tmp_merging"

    writer = PdfWriter()
    page_info = {"pages": []}

    supplement_pdfs = []
    if isinstance(supplement_source, list):
        supplement_pdfs = supplement_source
    elif isinstance(supplement_source, str) and os.path.isdir(supplement_source):
        files = sorted(os.listdir(supplement_source))
        supplement_pdfs = [os.path.join(supplement_source, f) for f in files if f.lower().endswith(".pdf")]

    try:
        reader = PdfReader(main_pdf)
        main_page_count = len(reader.pages)
        for page in reader.pages:
            writer.add_page(page)
        page_info["pages"].append({
            "name": "Main document",
            "pages": main_page_count,
            "numbering": "See main text page numbers",
            "pdf_start": 1,
            "pdf_end": main_page_count,
        })
    except Exception as e:
        print(f"  ✗ Error merging main PDF: {e}")
        return main_pdf, page_info

    current_pdf_page = main_page_count + 1

    if supplement_pdfs:
        sep_path = prompts_dir() / "resources" / "separator_supp.pdf"
        if sep_path.exists():
            try:
                sep_reader = PdfReader(str(sep_path))
                sep_count = len(sep_reader.pages)
                for page in sep_reader.pages:
                    writer.add_page(page)
                current_pdf_page += sep_count
            except Exception as e:
                print(f"  ⚠ Warning: Could not merge separator: {e}")
        else:
            print(f"  ⚠ Warning: Separator file not found at {sep_path}")

    for _i, supp_pdf in enumerate(supplement_pdfs, 1):
        try:
            reader = PdfReader(supp_pdf)
            supp_page_count = len(reader.pages)
            for page in reader.pages:
                writer.add_page(page)

            first_page_text = ""
            try:
                first_page_text = reader.pages[0].extract_text()
            except Exception:
                pass

            numbering = "Unknown"
            if re.search(r"\bS\d+\b", first_page_text):
                numbering = "S-numbered"
            elif re.search(r"\bAppendix\s+[A-Z]\b", first_page_text, re.IGNORECASE):
                numbering = "Appendix-lettered"

            filename = os.path.basename(supp_pdf)
            display_name = re.sub(r"^\d+_\d+_", "", filename)
            page_info["pages"].append({
                "name": display_name,
                "pages": supp_page_count,
                "numbering": numbering,
                "pdf_start": current_pdf_page,
                "pdf_end": current_pdf_page + supp_page_count - 1,
            })
            current_pdf_page += supp_page_count
        except Exception as e:
            print(f"  ⚠  Skipped corrupt supplement {supp_pdf}: {e}")

    with open(temp_path, "wb") as f:
        writer.write(f)
    if os.path.exists(output_path):
        os.remove(output_path)
    shutil.move(temp_path, output_path)
    return output_path, page_info


