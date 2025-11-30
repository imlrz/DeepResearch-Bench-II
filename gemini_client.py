# -*- coding: utf-8 -*-
"""
Gemini API client.

Provides a unified multimodal input interface that supports text, files (PDF, images, etc.)
and extra images, with configuration loaded from a .env file.
"""

import os
import base64
import mimetypes
import requests
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path


def _load_env_file(env_path: str = ".env") -> Dict[str, str]:
    """
    Load a .env file (simple implementation without python-dotenv).

    Args:
        env_path: Path to the .env file.

    Returns:
        A dict of configuration values.
    """
    env_vars = {}
    
    # Look for the .env file (prefer current directory, then script directory)
    possible_paths = [
        env_path,
        os.path.join(os.getcwd(), env_path),
        os.path.join(Path(__file__).parent, env_path)
    ]
    
    env_file = None
    for path in possible_paths:
        if os.path.exists(path):
            env_file = path
            break
    
    if not env_file:
        return env_vars
    
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Parse KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    except Exception as e:
        print(f"[warn] failed to read .env file: {e}")
    
    return env_vars


# Load .env configuration
_ENV_CONFIG = _load_env_file()


def get_config(key: str, default: str = None) -> str:
    """
    Get a configuration value (preferring environment variables, then .env file).

    Args:
        key: Configuration key.
        default: Default value if not found.

    Returns:
        The configuration value.
    """
    # Prefer environment variables
    value = os.environ.get(key)
    if value:
        return value
    
    # Then fall back to .env
    value = _ENV_CONFIG.get(key)
    if value:
        return value
    
    # Finally return default if still missing
    return default


# =========================
# Dataclass definitions
# =========================

@dataclass
class GeminiInput:
    """
    Input payload for the Gemini API.

    Fields:
        - text: str - Required. Prompt text.
        - file_path: Optional[str] - Optional. Main file path (PDF, image, etc.).
        - extra_images: List[Tuple[str, bytes]] - Optional. Extra images in the form
          [(mime_type, image_bytes), ...].
        - stream: bool - Optional. Whether to use streaming responses. Default False.

    Example:
        # Text-only input
        input1 = GeminiInput(text="Please analyze this paragraph...")

        # Text + PDF file
        input2 = GeminiInput(text="Please analyze this PDF document.", file_path="document.pdf")

        # Text + multiple images
        input3 = GeminiInput(
            text="Please describe these images.",
            extra_images=[("image/png", img1_bytes), ("image/jpeg", img2_bytes)]
        )
    """
    text: str
    file_path: Optional[str] = None
    extra_images: Optional[List[Tuple[str, bytes]]] = None
    stream: bool = False


@dataclass
class GeminiOutput:
    """
    Output payload from the Gemini API.

    Fields:
        - text: str - Text content returned by the model.
        - usage_metadata: Dict - Token usage statistics:
            - promptTokenCount: int - Number of input tokens.
            - candidatesTokenCount: int - Number of output tokens.
            - totalTokenCount: int - Total token count.
            - thoughtsTokenCount: int - Token count for hidden reasoning (if available).
        - upload_stats: Dict - Summary of what was uploaded:
            - text_segments: int - Number of text segments.
            - files: List[Dict] - Per-file metadata with type and source.
        - raw_response: Dict - Raw API JSON response.
    """
    text: str
    usage_metadata: Dict = field(default_factory=dict)
    upload_stats: Dict = field(default_factory=dict)
    raw_response: Dict = field(default_factory=dict)


# =========================
# Gemini client class
# =========================

class GeminiClient:
    """
    Gemini API client.
    
    Supports multimodal input (text, images, PDFs, etc.) and handles API calls.
    """
    
    # MIME types that can be uploaded as inline binary data
    ALLOWED_INLINE_MIMES = {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
        "image/bmp",
        "image/tiff",
        "application/pdf",
    }
    
    def __init__(
        self,
        api_url: Optional[str] = None,
        api_token: Optional[str] = None,
        model: Optional[str] = None,
        request_id: Optional[str] = None,
        verbose: bool = True
        ):
        """
        Initialize Gemini client.

        Args:
            api_url: API endpoint URL (optional, defaults to GEMINI_API_URL from env/.env).
            api_token: API access token (optional, defaults to GEMINI_API_TOKEN).
            model: Model name (optional, defaults to GEMINI_MODEL).
            request_id: Request identifier (optional, defaults to GEMINI_REQUEST_ID).
            verbose: Whether to print detailed logs.

        Configuration priority:
            1. Explicit constructor arguments
            2. Environment variables
            3. .env file
            4. Defaults
        """
        self.api_url = api_url or get_config("GEMINI_API_URL")
        self.api_token = api_token or get_config("GEMINI_API_TOKEN")
        self.model = model or get_config("GEMINI_MODEL")
        self.request_id = request_id or get_config("GEMINI_REQUEST_ID", "default-request")
        self.verbose = verbose
        
        # Validate required configuration
        if not self.api_url:
            raise ValueError("Missing configuration: GEMINI_API_URL (set it in .env or pass via parameter).")
        if not self.api_token:
            raise ValueError("Missing configuration: GEMINI_API_TOKEN (set it in .env or pass via parameter).")
        if not self.model:
            raise ValueError("Missing configuration: GEMINI_MODEL (set it in .env or pass via parameter).")
    
    def query(self, input_data: GeminiInput) -> GeminiOutput:
        """
        Send a query to the Gemini API.

        Args:
            input_data: GeminiInput object.

        Returns:
            GeminiOutput object containing response text and metadata.

        Raises:
            requests.HTTPError: If the API request fails.
            ValueError: If the response cannot be parsed.
        """
        # Build Gemini "parts"
        parts, upload_stats = self._build_parts(
            input_data.text,
            input_data.file_path,
            input_data.extra_images
        )
        
        # Print upload info
        if self.verbose:
            self._print_upload_info(upload_stats)
        
        # Build request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_token}",
            "rock-request-id": self.request_id
        }
        
        payload = {
            "model": self.model,
            "contents": [{"role": "user", "parts": parts}]
        }
        
        if input_data.stream:
            payload["stream"] = True
        
        # Send request
        resp = requests.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=600,
            stream=input_data.stream
        )
        
        # Basic error handling
        if resp.status_code >= 400:
            error_body = resp.text[:1000] if hasattr(resp, 'text') else "failed to read error body"
            print(f"[err] HTTP {resp.status_code} body: {error_body}")
            resp.raise_for_status()
        
        # Parse response
        resp_json = resp.json()
        text_content = self._extract_text_from_response(resp_json)
        usage_metadata = resp_json.get("usageMetadata", {})
        
        return GeminiOutput(
            text=text_content,
            usage_metadata=usage_metadata,
            upload_stats=upload_stats,
            raw_response=resp_json
        )
    
    def _build_parts(
        self,
        text: str,
        file_path: Optional[str] = None,
        extra_images: Optional[List[Tuple[str, bytes]]] = None
    ) -> Tuple[List[Dict], Dict]:
        """
        Build Gemini parts and upload statistics.

        Returns:
            (parts, upload_stats)
        """
        parts = [{"text": text}]
        upload_stats = {
            "text_segments": 1,
            "files": []
        }
        
        # Add main file (if any)
        if file_path:
            mime = self._get_mime_type(file_path)
            if mime in self.ALLOWED_INLINE_MIMES:
                parts.append({
                    "inlineData": {
                        "mimeType": mime,
                        "data": self._encode_file_base64(file_path)
                    }
                })
                upload_stats["files"].append({"type": mime, "source": "main_file"})
        
        # Add extra images (if any)
        if extra_images:
            for mime_type, img_bytes in extra_images:
                parts.append({
                    "inlineData": {
                        "mimeType": mime_type,
                        "data": base64.b64encode(img_bytes).decode("utf-8")
                    }
                })
                upload_stats["files"].append({"type": mime_type, "source": "extracted_image"})
        
        return parts, upload_stats
    
    def _extract_text_from_response(self, resp_json: Dict) -> str:
        """
        Extract text content from the Gemini response JSON.
        """
        text_content = ""
        if 'candidates' in resp_json and resp_json['candidates']:
            cand = resp_json['candidates'][0]
            if 'content' in cand and 'parts' in cand['content']:
                for part in cand['content']['parts']:
                    if isinstance(part, dict) and 'text' in part:
                        text_content += part['text']
        return text_content
    
    def _get_mime_type(self, file_path: str, default: str = "application/octet-stream") -> str:
        """
        Get MIME type for a given file path.
        """
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or default
    
    def _encode_file_base64(self, file_path: str) -> str:
        """
        Encode file contents to a base64 string.
        """
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _print_upload_info(self, upload_stats: Dict):
        """
        Print a short summary of what was uploaded.
        """
        text_count = upload_stats.get("text_segments", 0)
        files = upload_stats.get("files", [])
        
        # Aggregate file types
        file_type_count = {}
        for f in files:
            ftype = f["type"]
            file_type_count[ftype] = file_type_count.get(ftype, 0) + 1
        
        info_parts = []
        if text_count > 0:
            info_parts.append(f"{text_count} text segment(s)")
        
        for ftype, count in file_type_count.items():
            info_parts.append(f"{count} file(s) of type {ftype}")
        
        if info_parts:
            print(f"[upload] {', '.join(info_parts)}")

