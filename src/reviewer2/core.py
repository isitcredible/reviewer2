import os
import time
import shutil
import tempfile
import subprocess
import re
import atexit
import threading
from datetime import datetime
from pypdf import PdfWriter, PdfReader

from reviewer2.paths import prompt_path

# --- SDK IMPORT ---
from google import genai
from google.genai import types
from google.genai.types import SafetySetting, HarmCategory, HarmBlockThreshold

# Global Usage Log
USAGE_LOG = []

# Set per-run by the pipeline so each usage row can be attributed to a run.
CURRENT_JOB_UUID = None

# GLOBAL CACHE REGISTRY
# Format: { "path/to/local/file.pdf": uploaded_file_object }
_FILE_CACHE = {}
# Stages now run concurrently, so the first upload of a shared PDF must be
# serialized per path or every parallel stage re-uploads the same 50MB file.
_CACHE_LOCK = threading.Lock()
_UPLOAD_LOCKS = {}

MODELS = {
    "flash_lite": "gemini-2.5-flash-lite",
    "flash_lite_3": "gemini-3.1-flash-lite-preview",
    "flash_2_5": "gemini-2.5-flash",
    "flash_3": "gemini-3-flash-preview",
    "pro_2_5": "gemini-2.5-pro",
    "pro_3": "gemini-3-pro-preview",
    "pro_3_1": "gemini-3.1-pro-preview"
}

def get_or_upload_file(client, file_path, force_upload=False):
    """
    Checks if file is already uploaded in this session.
    If yes, returns existing handle. If no (or force=True), uploads and caches it.
    """
    global _FILE_CACHE

    with _CACHE_LOCK:
        if not force_upload and file_path in _FILE_CACHE:
            # Optional: could verify file state here, but that adds latency.
            # We rely on the 403 retry logic to handle stale files.
            return _FILE_CACHE[file_path]
        path_lock = _UPLOAD_LOCKS.setdefault(file_path, threading.Lock())

    with path_lock:
        # Another thread may have finished the upload while we waited.
        if not force_upload and file_path in _FILE_CACHE:
            return _FILE_CACHE[file_path]
        return _upload_file(client, file_path)


def _upload_file(client, file_path):
    # Shrink oversized PDFs under Gemini's 50MB document limit by stripping
    # photographic figures (which the finders ignore) while preserving text
    # and tables. No-op for PDFs already under the target. Lazy import so a
    # missing PyMuPDF degrades to uploading the original rather
    # than breaking core.py at import time.
    upload_path = file_path
    if isinstance(file_path, str) and file_path.lower().endswith(".pdf"):
        try:
            from reviewer2.pdf_compress import compress_pdf_for_gemini
            compressed, cinfo = compress_pdf_for_gemini(file_path)
            if cinfo.get("action") == "stripped":
                print(f"    ↳ 🗜  PDF {cinfo['orig_bytes']//1048576}MB → "
                      f"{cinfo['final_bytes']//1048576}MB "
                      f"(stripped {cinfo['stripped']} figure image(s); "
                      f"{cinfo['protected']} protected)")
                if cinfo.get("over_limit"):
                    print("    ⚠ still over Gemini's 50MB limit after stripping; "
                          "upload may be rejected")
            upload_path = compressed
        except Exception as e:
            print(f"    ⚠ PDF compression skipped ({e}); uploading original")
            upload_path = file_path

    print(f"    ↳ 📤 Uploading NEW file to Gemini: {os.path.basename(file_path)}...")
    try:
        uploaded_file = client.files.upload(file=upload_path)
        _FILE_CACHE[file_path] = uploaded_file
        print(f"    ✓ Upload complete: {uploaded_file.name} (Cached)")
        return uploaded_file
    except Exception as e:
        # If upload fails, ensure we don't leave bad cache
        if file_path in _FILE_CACHE:
            del _FILE_CACHE[file_path]
        raise IOError(f"Error uploading PDF '{file_path}': {e}")

def cleanup_resources():
    """
    Deletes all files uploaded during this session.
    """
    global _FILE_CACHE
    if not _FILE_CACHE:
        return

    print("\n🧹 Cleaning up Gemini resources...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: 
        return

    try:
        client = genai.Client(api_key=api_key)
        for path, file_obj in list(_FILE_CACHE.items()):
            try:
                client.files.delete(name=file_obj.name)
                print(f"  ✓ Deleted remote file: {file_obj.name}")
            except Exception as e:
                # 404 means it's already gone, which is fine.
                if "404" not in str(e):
                    print(f"  ⚠ Failed to delete {file_obj.name}: {e}")
        _FILE_CACHE.clear()
    except Exception as e:
        print(f"  ⚠ Cleanup initialization failed: {e}")

# Register cleanup to run automatically on exit
atexit.register(cleanup_resources)

def call_gemini(prompt=None, pdf_file_path=None, model_type="flash_lite",
                temperature=0.1, thinking_level=None, thinking_budget=None,
                media_resolution="MEDIA_RESOLUTION_MEDIUM", system_instruction=None,
                max_retries=10, retry_forever_on_rate_limit=True,
                step=None, use_search=False, max_output_tokens=None,
                use_code_exec=False, timeout=1800):
    """
    Call Gemini API with robust retry logic and proper Content structure.
    Uses caching for PDF uploads, with auto-invalidation on permission errors.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment")

    override = os.environ.get("GEMINI_MODEL_OVERRIDE")
    if override:
        model_type = override

    # Env-gated code execution: comma-separated step names in CODE_EXEC_STEPS
    if step and step in os.environ.get("CODE_EXEC_STEPS", "").split(","):
        use_code_exec = True

    model_name = MODELS.get(model_type, model_type)
    if not model_name:
        print(f"  ⚠  Warning: Unknown model_type '{model_type}', falling back to flash_lite")
        model_name = MODELS["flash_lite"]

    # Finite read timeout: a silently dead connection (e.g. local link drop)
    # must surface as an exception for the retry loop, not block forever. The
    # 1800s default suits the PDF-reading stages; a caller doing a quick lookup
    # (ref-check grounding) passes a much shorter value so a dead socket fails
    # fast instead of stalling the whole run.
    http_options = types.HttpOptions(client_args={'timeout': timeout})
    client = genai.Client(api_key=api_key, http_options=http_options)

    # =========================================================================
    # SAFETY SETTINGS
    # =========================================================================
    safety_settings = [
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
    ]

    # =========================================================================
    # THINKING CONFIG (Fixed for Gemini 3)
    # =========================================================================
    thinking_cfg = None
    model_supports_thinking = (
        "pro" in model_type or 
        "2.5" in model_type or 
        "flash" in model_type or 
        "3" in model_name
    )
    
    if model_supports_thinking:
        # If the user passes a specific integer budget (Gemini 2.5 style)
        if thinking_budget is not None:
             thinking_cfg = types.ThinkingConfig(thinking_budget=int(thinking_budget))
             
        # If the user passes a level string (Gemini 3 style: low, medium, high)
        elif thinking_level:
            # Gemini 3 requires valid strings. Default to "include_thoughts=False" (summary only)
            # unless you specifically want to see the raw thoughts for debugging.
            thinking_cfg = types.ThinkingConfig(
                thinking_level=thinking_level, 
                include_thoughts=False # Change to True if you want to debug the thoughts
            )
            
        # Default behavior (Dynamic)
        else:
             thinking_cfg = types.ThinkingConfig(thinking_budget=-1)

    # =========================================================================
    # TOOLS (SEARCH)
    # =========================================================================
    tools = None
    if use_search:
        # Enable Google Search Grounding
        tools = [types.Tool(google_search=types.GoogleSearch())]
    elif use_code_exec:
        # Enable Python code execution (numerical checks in the 01fe defence)
        tools = [types.Tool(code_execution=types.ToolCodeExecution())]

    # =========================================================================
    # RETRY LOOP
    # =========================================================================
    attempt = 0
    base_delay = 5
    max_delay = 300
    
    # Track if we have forced a re-upload for this specific call
    has_forced_reupload = False

    while True:
        attempt += 1
        
        # ---------------------------------------------------------------------
        # PREPARE CONTENTS (Done inside loop to handle re-uploads)
        # ---------------------------------------------------------------------
        parts = []
        current_uploaded_file = None

        if pdf_file_path:
            # If we previously hit a 403 on the file, we force upload
            force = (attempt > 1 and has_forced_reupload)
            try:
                current_uploaded_file = get_or_upload_file(client, pdf_file_path, force_upload=force)
                # Reset flag so we don't upload every single retry if it was a different error
                has_forced_reupload = False 
                
                parts.append(types.Part.from_uri(
                    file_uri=current_uploaded_file.uri,
                    mime_type="application/pdf"
                ))
            except Exception as e:
                print(f"    ⚠  Upload failed inside retry loop: {e}")
                # If upload fails, we likely can't proceed, but we'll let the loop handle delay
                time.sleep(5)
                continue

        if prompt:
            parts.append(types.Part.from_text(text=prompt))

        contents = [types.Content(role="user", parts=parts)]

        # ---------------------------------------------------------------------
        # BUILD CONFIG
        # ---------------------------------------------------------------------
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
            max_output_tokens=max_output_tokens
        )

        # ---------------------------------------------------------------------
        # EXECUTE REQUEST
        # ---------------------------------------------------------------------
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=gen_config
            )

            if response is None:
                raise ValueError("API returned None response object")

            # Safe Usage Logging
            usage_meta = getattr(response, "usage_metadata", None)
            if usage_meta:
                p_tok = getattr(usage_meta, 'prompt_token_count', 0) or 0
                c_tok = getattr(usage_meta, 'candidates_token_count', 0) or 0
                t_tok = getattr(usage_meta, 'thoughts_token_count', 0) or 0
                
                USAGE_LOG.append({
                    "model_name": model_name,
                    "input_tokens": p_tok,
                    "output_tokens": c_tok + t_tok,
                    "thoughts_tokens": t_tok,
                    "timestamp": datetime.now().isoformat(),
                    "step": step or "unknown",
                    "job_uuid": CURRENT_JOB_UUID
                })

            # Safe Text Extraction
            try:
                # 1. Global Feedback Check (Did the prompt trigger a block immediately?)
                feedback = getattr(response, "prompt_feedback", None)
                if feedback and getattr(feedback, "block_reason", None):
                    error_msg = f"FATAL: Prompt blocked by filters: {feedback.block_reason}"
                    print(f"    ❌ {error_msg}")
                    raise RuntimeError(error_msg)

                candidates = getattr(response, "candidates", []) or []
                if not candidates:
                    # If candidates list is empty, it's usually a hard block or server failure.
                    # We treat this as retryable unless we can prove otherwise, but often it's safety.
                    raise ValueError("No candidates returned (Possible Safety Block or Empty Response)")

                # 2. Check Candidate Finish Reasons (The Critical Fix)
                # We check the first candidate (usually the only one)
                first_cand = candidates[0]
                finish_reason = str(getattr(first_cand, "finish_reason", "UNKNOWN"))
                
                # Valid reasons are "STOP" (success) or "MAX_TOKENS" (truncated but valid).
                # Common Failure Reasons: "SAFETY", "RECITATION" (Copyright), "OTHER"
                valid_reasons = ["STOP", "1", "FinishReason.STOP", "MAX_TOKENS", "2", "FinishReason.MAX_TOKENS"]
                
                if finish_reason not in valid_reasons:
                    safety_ratings = getattr(first_cand, "safety_ratings", "Unknown")
                    error_msg = f"FATAL: Generation stopped. Reason: {finish_reason}. Ratings: {safety_ratings}"
                    print(f"    ❌ {error_msg}")
                    raise RuntimeError(error_msg)

                # 3. Extract Text (Thinking vs Standard)
                if thinking_cfg:
                    final_text = []
                    content = getattr(first_cand, "content", None)
                    parts = getattr(content, "parts", None)
                    
                    if parts is None:
                        # --- MODIFIED BLOCK START ---
                        # If the model hit the limit and returned nothing, we MUST stop.
                        # Adding "FATAL" to the string kills the retry loop.
                        if "MAX_TOKENS" in str(finish_reason):
                             raise ValueError(f"FATAL: Model hit MAX_TOKENS and returned no content. (Finish Reason: {finish_reason})")
                        else:
                             raise ValueError(f"API returned malformed content: 'parts' is None (Finish Reason: {finish_reason})")
                        # --- MODIFIED BLOCK END ---

                    for part in parts:
                        # Filter out thoughts, keep text
                        if part.text and not getattr(part, 'thought', False): 
                            final_text.append(part.text)
                    
                    result = "".join(final_text).strip()
                else:
                    # Standard Model Extraction
                    # We prefer using the direct .text accessor if available, but safeguard it
                    if hasattr(response, 'text') and response.text:
                        result = response.text
                    else:
                        # Manual extraction if helper fails
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
                # If we identified a FATAL error above, re-raise it to break the loop
                if "FATAL" in str(e):
                    raise e
                    
                # Standard retries for other extraction errors
                if "API returned malformed content" in str(e):
                    raise e
                    
                raise ValueError(f"Failed to extract text from response: {e}")

        except Exception as e:
            err_str = str(e)
            
            # STOP RETRYING ON FATAL ERRORS
            if "FATAL" in err_str:
                raise e
            
            if "'NoneType' object is not iterable" in err_str:
                print(f"    ✗ FATAL ERROR: Malformed API response (NoneType iterable). Stopping.")
                raise e

            # -----------------------------------------------------------------
            # SPECIAL HANDLING FOR FILE PERMISSIONS (403)
            # -----------------------------------------------------------------
            if "403" in err_str and ("access the File" in err_str or "PERMISSION_DENIED" in err_str):
                print(f"    ⚠  File Permission/Access Error (403). Forcing re-upload on next attempt.")
                # Mark to force re-upload next loop
                has_forced_reupload = True
                # Remove from cache immediately to be safe
                if pdf_file_path and pdf_file_path in _FILE_CACHE:
                    del _FILE_CACHE[pdf_file_path]
                
                # Short delay for 403, as we have a fix strategy
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


# =============================================================================
# OPTIONAL PROVIDER ROUTING
# =============================================================================
# When DEEPSEEK_API_KEY is set, these clerical stages go to DeepSeek V4 Flash,
# which is much cheaper than Gemini Flash for them. They work purely on prior
# text: DeepSeek has no PDF vision, so ONLY stages whose prompt is fully
# self-contained in text belong here. 00b_metadata_clean's prompt says "rely
# ONLY on the INPUT DATA", so it qualifies. Without the key, everything runs
# on Gemini.
DS_FLASH_STAGES = {
    "00b_metadata_clean",
    "00b_2_metadata_math",
    "02b_compiler_1",
    "02d_compiler_2",
    "02f_compiler_3",
    "06_legal",
}


def call_model(prompt=None, pdf_file_path=None, model_type="flash_lite",
               step=None, **kwargs):
    """Dispatcher used by the bureaucratic stages. Routes DS_FLASH_STAGES to
    DeepSeek V4 Flash, everything else to Gemini. A GEMINI_MODEL_OVERRIDE
    (cheap test runs) keeps everything on Gemini. The DeepSeek path drops the PDF —
    these stages are text-only by construction — and falls back to Gemini on
    any DeepSeek error so a bureaucratic stage can never sink a report."""
    override = os.environ.get("GEMINI_MODEL_OVERRIDE")
    route_ds = (step in DS_FLASH_STAGES and not override
                and os.environ.get("DEEPSEEK_API_KEY"))
    if route_ds:
        try:
            from reviewer2.deepseek_client import call_deepseek
            return call_deepseek(
                prompt,
                model="deepseek-v4-flash",
                temperature=kwargs.get("temperature", 0.0),
                thinking_budget=kwargs.get("thinking_budget"),
                thinking_level=kwargs.get("thinking_level"),
                system_instruction=kwargs.get("system_instruction"),
                step=step,
                max_output_tokens=kwargs.get("max_output_tokens"),
            )
        except Exception as e:
            print(f"    ⚠  DeepSeek failed for {step} ({e}); falling back to Gemini")
    return call_gemini(prompt, pdf_file_path, model_type=model_type, step=step, **kwargs)


def save_output(content, filename, output_dir):
    if content is None:
        print(f"  ⚠  WARNING: content is None for {filename}. Creating empty file.")
        content = ""
        
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  ✓ Saved: {filename}")


def load_prompt(relative_path):
    path = prompt_path(relative_path)
    if not os.path.exists(path):
        return f"[Error: Prompt file {relative_path} missing]"

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if "{{OUTPUT_FORMAT}}" in text:
        output_format_path = prompt_path("resources/output_format.txt")
        if os.path.exists(output_format_path):
            with open(output_format_path, "r", encoding="utf-8") as f:
                fmt = f.read()
            text = text.replace("{{OUTPUT_FORMAT}}", fmt)
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
        cmd = [
            gs_cmd, "-o", fixed_path, 
            "-sDEVICE=pdfwrite", 
            "-dPDFSETTINGS=/prepress", 
            "-dQUIET", 
            input_path
        ]
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
    # 1. Determine Output Path
    # We want to overwrite the original or keep the same filename in the target dir.
    # No more "_merged" suffix.
    if output_dir:
        output_path = os.path.join(output_dir, os.path.basename(main_pdf))
    else:
        output_path = main_pdf

    # Write to a temp file first to avoid read/write conflicts if overwriting
    temp_path = output_path + ".tmp_merging"

    writer = PdfWriter()
    page_info = {'pages': []}
    
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
        
        page_info['pages'].append({
            'name': 'Main document',
            'pages': main_page_count,
            'numbering': 'See main text page numbers',
            'pdf_start': 1,
            'pdf_end': main_page_count
        })
    except Exception as e:
        print(f"  ✗ Error merging main PDF: {e}")
        # If main PDF fails, we can't proceed safely
        return main_pdf, page_info

    current_pdf_page = main_page_count + 1
    
    # --- INSERT SEPARATOR IF SUPPLEMENTS EXIST ---
    if supplement_pdfs:
        sep_path = str(prompt_path("resources/separator_supp.pdf"))
        if os.path.exists(sep_path):
            try:
                sep_reader = PdfReader(sep_path)
                sep_count = len(sep_reader.pages)
                for page in sep_reader.pages:
                    writer.add_page(page)
                
                # IMPORTANT: Increment the PDF page counter so the next section
                # (the actual supplements) has the correct start page in the index.
                current_pdf_page += sep_count
            except Exception as e:
                print(f"  ⚠ Warning: Could not merge separator: {e}")
        else:
             print(f"  ⚠ Warning: Separator file not found at {sep_path}")

    # --- APPEND SUPPLEMENTS ---
    for i, supp_pdf in enumerate(supplement_pdfs, 1):
        try:
            reader = PdfReader(supp_pdf)
            supp_page_count = len(reader.pages)
            
            for page in reader.pages:
                writer.add_page(page)
            
            first_page_text = ""
            try: first_page_text = reader.pages[0].extract_text()
            except: pass
            
            numbering = "Unknown"
            if re.search(r'\bS\d+\b', first_page_text):
                numbering = "S-numbered"
            elif re.search(r'\bAppendix\s+[A-Z]\b', first_page_text, re.IGNORECASE):
                numbering = "Appendix-lettered"
            
            filename = os.path.basename(supp_pdf)
            # Strip the staging prefixes (10_10_, then an NNN__ ordering
            # prefix if present), leaving the original name.
            display_name = re.sub(r'^\d+_\d+_', '', filename)
            display_name = re.sub(r'^\d+__', '', display_name)
            
            page_info['pages'].append({
                'name': display_name,
                'pages': supp_page_count,
                'numbering': numbering,
                'pdf_start': current_pdf_page,
                'pdf_end': current_pdf_page + supp_page_count - 1
            })
            current_pdf_page += supp_page_count
            
        except Exception as e:
            print(f"  ⚠  Skipped corrupt supplement {supp_pdf}: {e}")

    # --- SAVE AND MOVE ---
    with open(temp_path, "wb") as f:
        writer.write(f)
    
    # Overwrite the target (or source) with the new merged version
    if os.path.exists(output_path):
        os.remove(output_path)
    shutil.move(temp_path, output_path)
    
    return output_path, page_info


def format_page_mapping_for_prompt(page_mapping):
    # New deterministic map (pagemap.build_page_map): render the real
    # physical->printed sections + appendix ceiling.
    if isinstance(page_mapping, dict) and ('sections' in page_mapping or 'confidence' in page_mapping):
        from reviewer2 import pagemap
        return pagemap.render_details(page_mapping)
    # Legacy shape (kept for safety; not produced anywhere now).
    if not page_mapping.get('ranges'):
        return "- Document structure: Check printed page numbers in the document"
    lines = []
    for range_info in page_mapping['ranges']:
        line = f"- {range_info['section']}: PDF viewer pages {range_info['pdf_pages']} → Printed pages: {range_info['printed_pages']}"
        lines.append(line)
    return "\n".join(lines)


def load_page_numbers_snippet(page_mapping):
    template_path = str(prompt_path("resources/page_numbers.txt"))
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()
    except FileNotFoundError:
        print(f"  ⚠  Warning: {template_path} not found. Using basic instructions.")
        return "\n\n## PAGE NUMBERS\n- Use printed page numbers from the document, not PDF viewer page numbers"
    
    structure_details = format_page_mapping_for_prompt(page_mapping)
    formatted = template.replace("{{PAGE_STRUCTURE_DETAILS}}", structure_details)
    
    return "\n\n" + formatted