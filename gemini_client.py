# -*- coding: utf-8 -*-
"""
Gemini API 客户端
提供统一的多模态输入接口，支持文本、文件（PDF、图片等）和额外图片
配置从 .env 文件读取
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
    加载 .env 文件（简单实现，不依赖 python-dotenv）
    
    参数：
    - env_path: .env 文件路径
    
    返回：
    - 配置字典
    """
    env_vars = {}
    
    # 查找 .env 文件（优先当前目录，然后是脚本所在目录）
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
                # 跳过注释和空行
                if not line or line.startswith('#'):
                    continue
                # 解析 KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    except Exception as e:
        print(f"[warn] 读取 .env 文件失败：{e}")
    
    return env_vars


# 加载 .env 配置
_ENV_CONFIG = _load_env_file()


def get_config(key: str, default: str = None) -> str:
    """
    获取配置值（优先从环境变量，然后从 .env 文件）
    
    参数：
    - key: 配置键
    - default: 默认值
    
    返回：
    - 配置值
    """
    # 优先从环境变量读取
    value = os.environ.get(key)
    if value:
        return value
    
    # 然后从 .env 文件读取
    value = _ENV_CONFIG.get(key)
    if value:
        return value
    
    # 返回默认值
    return default


# =========================
# 数据类定义
# =========================

@dataclass
class GeminiInput:
    """
    Gemini API 输入数据
    
    字段说明：
    - text: str - 必需，提示词文本内容
    - file_path: Optional[str] - 可选，主文件路径（PDF、图片等）
    - extra_images: List[Tuple[str, bytes]] - 可选，额外的图片列表，格式为 [(mime_type, image_bytes), ...]
    - stream: bool - 可选，是否使用流式响应，默认 False
    
    示例：
        # 纯文本输入
        input1 = GeminiInput(text="请分析这段文字...")
        
        # 文本 + PDF 文件
        input2 = GeminiInput(text="请分析这个PDF", file_path="document.pdf")
        
        # 文本 + 多个图片
        input3 = GeminiInput(
            text="描述这些图片",
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
    Gemini API 输出数据
    
    字段说明：
    - text: str - 模型返回的文本内容
    - usage_metadata: Dict - token 使用统计
        - promptTokenCount: int - 输入 token 数
        - candidatesTokenCount: int - 输出 token 数
        - totalTokenCount: int - 总 token 数
        - thoughtsTokenCount: int - 思考 token 数（如果有）
    - upload_stats: Dict - 上传内容统计
        - text_segments: int - 文本段落数
        - files: List[Dict] - 文件列表，每个文件包含 type 和 source
    - raw_response: Dict - 原始 API 响应（完整 JSON）
    
    示例：
        output = GeminiOutput(
            text="这是模型的回复...",
            usage_metadata={
                "promptTokenCount": 1000,
                "candidatesTokenCount": 500,
                "totalTokenCount": 1500
            },
            upload_stats={
                "text_segments": 1,
                "files": [{"type": "application/pdf", "source": "main_file"}]
            }
        )
    """
    text: str
    usage_metadata: Dict = field(default_factory=dict)
    upload_stats: Dict = field(default_factory=dict)
    raw_response: Dict = field(default_factory=dict)


# =========================
# Gemini Client 类
# =========================

class GeminiClient:
    """
    Gemini API 客户端
    
    支持多模态输入（文本、图片、PDF 等），统一处理 API 调用
    """
    
    # 支持以二进制方式上传的 MIME 类型
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
        初始化 Gemini 客户端
        
        参数：
        - api_url: API 端点 URL（可选，默认从 .env 读取 GEMINI_API_URL）
        - api_token: API 访问令牌（可选，默认从 .env 读取 GEMINI_API_TOKEN）
        - model: 模型名称（可选，默认从 .env 读取 GEMINI_MODEL）
        - request_id: 请求标识符（可选，默认从 .env 读取 GEMINI_REQUEST_ID）
        - verbose: 是否打印详细信息
        
        配置优先级：
        1. 构造函数参数
        2. 环境变量
        3. .env 文件
        4. 默认值
        """
        self.api_url = api_url or get_config("GEMINI_API_URL")
        self.api_token = api_token or get_config("GEMINI_API_TOKEN")
        self.model = model or get_config("GEMINI_MODEL")
        self.request_id = request_id or get_config("GEMINI_REQUEST_ID", "default-request")
        self.verbose = verbose
        
        # 验证必需配置
        if not self.api_url:
            raise ValueError("缺少配置：GEMINI_API_URL（请在 .env 文件中设置或通过参数传入）")
        if not self.api_token:
            raise ValueError("缺少配置：GEMINI_API_TOKEN（请在 .env 文件中设置或通过参数传入）")
        if not self.model:
            raise ValueError("缺少配置：GEMINI_MODEL（请在 .env 文件中设置或通过参数传入）")
    
    def query(self, input_data: GeminiInput) -> GeminiOutput:
        """
        发送查询到 Gemini API
        
        参数：
        - input_data: GeminiInput 对象
        
        返回：
        - GeminiOutput 对象，包含响应文本和元数据
        
        异常：
        - requests.HTTPError: API 请求失败
        - ValueError: 响应解析失败
        """
        # 构建 Gemini parts
        parts, upload_stats = self._build_parts(
            input_data.text,
            input_data.file_path,
            input_data.extra_images
        )
        
        # 打印上传信息
        if self.verbose:
            self._print_upload_info(upload_stats)
        
        # 构建请求
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
        
        # 发送请求
        resp = requests.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=600,
            stream=input_data.stream
        )
        
        # 错误处理
        if resp.status_code >= 400:
            error_body = resp.text[:1000] if hasattr(resp, 'text') else "无法读取错误信息"
            print(f"[err] HTTP {resp.status_code} body: {error_body}")
            resp.raise_for_status()
        
        # 解析响应
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
        构建 Gemini parts 和上传统计
        
        返回：(parts, upload_stats)
        """
        parts = [{"text": text}]
        upload_stats = {
            "text_segments": 1,
            "files": []
        }
        
        # 添加主文件
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
        
        # 添加额外图片
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
        从 Gemini 响应中提取文本内容
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
        获取文件的 MIME 类型
        """
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or default
    
    def _encode_file_base64(self, file_path: str) -> str:
        """
        将文件编码为 base64 字符串
        """
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _print_upload_info(self, upload_stats: Dict):
        """
        打印上传内容的统计信息
        """
        text_count = upload_stats.get("text_segments", 0)
        files = upload_stats.get("files", [])
        
        # 统计文件类型
        file_type_count = {}
        for f in files:
            ftype = f["type"]
            file_type_count[ftype] = file_type_count.get(ftype, 0) + 1
        
        info_parts = []
        if text_count > 0:
            info_parts.append(f"{text_count} 段文字")
        
        for ftype, count in file_type_count.items():
            info_parts.append(f"{count} 个 {ftype} 文件")
        
        if info_parts:
            print(f"[上传] {', '.join(info_parts)}")

