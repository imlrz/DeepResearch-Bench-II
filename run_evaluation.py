# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import builtins
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

from gemini_client import GeminiClient, GeminiInput, GeminiOutput, get_config

# =========================
# 全局变量
# =========================
# Client 实例（在 main 函数中初始化）
client: Optional[GeminiClient] = None


# 从 .env 读取配置（带默认值）
def get_default_config():
    """从 .env 文件获取默认配置"""
    return {
        'pdf_dir': get_config('PDF_DIR', 'grok'),
        'out_jsonl': get_config('OUT_JSONL', 'eval_result_grok.jsonl'),
        'tasks_jsonl': get_config('TASKS_JSONL', 'tasks_and_rubrics.jsonl'),
        'chunk_size': int(get_config('CHUNK_SIZE', '50')),
        'max_workers': int(get_config('MAX_WORKERS', '10')),
        'max_retries': int(get_config('MAX_RETRIES', '5')),
        'max_paper_chars': int(get_config('MAX_PAPER_CHARS', '150000')),
        'log_file': get_config('LOG_FILE', 'run_evaluation.log'),
    }


def setup_print_logger(log_file: str):
    """
    简单日志机制：将所有 print 输出同时写入日志文件。

    - 不改变现有的 print 调用习惯；
    - 控制台仍然正常输出；
    - 每一行 print 的文本会追加写入 log_file。
    """
    if not log_file:
        return

    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # 避免重复包一层
    if getattr(builtins, "_orig_print", None) is None:
        builtins._orig_print = builtins.print

    def logged_print(*args, **kwargs):
        # 先输出到控制台
        builtins._orig_print(*args, **kwargs)
        try:
            text = " ".join(str(a) for a in args)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except Exception:
            # 日志异常不影响主流程
            pass

    builtins.print = logged_print

# =========================
# 提示词模板（三分类）
# =========================
PROMPT_TEMPLATE = """
You will receive an article, a task, and a list of grading rubric items. Your job is to assess whether the article satisfies each rubric item, and provide a THREE-WAY score for EACH rubric item.

Scoring rule per rubric item (strict):
- Score = 1: The article clearly satisfies the rubric item AND the specific supporting sentence(s) do NOT cite any reference listed in "blocked" (match by title/urls). For numerical data, exact values must be explicitly listed and match the rubric.
- Score = 0: The article does NOT mention this rubric item at all.
- Score = -1: The article mentions this rubric item, BUT the supporting sentence(s) cite a blocked reference.

For EACH rubric item, you MUST provide:
1. "score": 1, 0, or -1
2. "reason": A brief explanation
3. "evidence": The specific supporting sentence(s) from the article (empty string if score is 0)

The input format is:
<input_format>
{{
    "task": "...",
    "rubric_items": ["rubric item 1", "rubric item 2", ...],
    "blocked": {{
        "title": "...",
        "authors": ["...", "..."],
        "urls": ["...", "..."]
    }}
}}
</input_format>

Your output MUST strictly follow this JSON format (no extra keys, and the rubric item text MUST match the input EXACTLY):
<output_format>
{{
    "results": [
        {{
            "rubric_item": "rubric item 1",
            "score": 1 or 0 or -1,
            "reason": "brief explanation",
            "evidence": "supporting sentence(s) from the article"
        }},
        {{
            "rubric_item": "rubric item 2",
            "score": 1 or 0 or -1,
            "reason": "brief explanation",
            "evidence": "supporting sentence(s) from the article"
        }},
        ...
    ]
}}
</output_format>

CRITICAL: You MUST return results for ALL rubric items in the input, and the "rubric_item" text MUST match the input text EXACTLY (character-level match).

<passage>
{paper}
</passage>
<task_and_rubric>
{rubric}
</task_and_rubric>
Now, please begin your generation
"""

PAPER_PLACEHOLDER = "（PDF 已附在本条消息；请将其全文视作 <passage> 的内容进行查找与修改。）"

# =========================
# 读取 tasks.jsonl 索引
# =========================
def load_tasks_data(path: str):
    tasks_data = {}
    original_data = {}
    if not os.path.exists(path):
        print(f"[warn] 未找到 {path}，将默认全部跳过处理")
        return tasks_data, original_data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                idx_val = obj.get("idx", None)
                content = obj.get("content", "")
                if idx_val is not None:
                    try:
                        key = int(idx_val)
                    except Exception:
                        continue
                    tasks_data[key] = content
                    original_data[key] = obj
            except Exception:
                continue
    return tasks_data, original_data

# 延迟加载（在 main 中加载）
TASKS_DATA = {}
ORIGINAL_DATA = {}

# =========================
# 文档内容提取
# =========================
def _extract_docx_content(path: str) -> Tuple[str, List[Tuple[str, bytes]]]:
    """
    提取 docx 内容，返回 (文本内容, [(图片mime, 图片bytes)])
    """
    try:
        from docx import Document  # python-docx
    except Exception as e:
        print(f"[warn] 未安装 python-docx，无法解析 .docx：{e}")
        return "", []
    try:
        doc = Document(path)
    except Exception as e:
        print(f"[warn] 读取 .docx 失败：{e}")
        return "", []
    
    # 提取所有内容（段落 + 表格，按文档顺序）
    all_content = []
    images = []
    
    # 提取所有图片
    for rel in doc.part.rels.values():
        if "image" in rel.target_ref:
            try:
                image_blob = rel.target_part.blob
                # 从关系中获取 content_type
                content_type = rel.target_part.content_type
                images.append((content_type, image_blob))
            except Exception as e:
                print(f"[warn] 提取图片失败：{e}")
    
    # 遍历文档中的所有元素（段落和表格）
    for element in doc.element.body:
        # 段落
        if element.tag.endswith('p'):
            # 找到对应的 Paragraph 对象
            for p in doc.paragraphs:
                if p._element == element:
                    text = (p.text or "").strip()
                    if text:
                        all_content.append(text)
                    break
        # 表格
        elif element.tag.endswith('tbl'):
            # 找到对应的 Table 对象
            for table in doc.tables:
                if table._element == element:
                    # 提取表格内容（转为 Markdown 格式）
                    table_text = _table_to_markdown(table)
                    if table_text:
                        all_content.append(table_text)
                    break
    
    content = "\n\n".join(all_content)
    # 注意：截断逻辑已移到调用处，这里返回完整内容
    
    return content, images

def _table_to_markdown(table) -> str:
    """
    将 docx 表格转换为 Markdown 格式的文本
    """
    if not table.rows:
        return ""
    
    lines = []
    for i, row in enumerate(table.rows):
        cells = [cell.text.strip().replace('\n', ' ') for cell in row.cells]
        # 用 | 分隔单元格
        lines.append("| " + " | ".join(cells) + " |")
        # 第一行后添加分隔线
        if i == 0:
            lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
    
    return "\n".join(lines)



# =========================
# JSON fenced block 抽取/解析
# =========================
FENCED_JSON_PATTERN = r'```json\s*(.*)```'

def _try_clean_and_load(s: str):
    json_clean = re.sub(
        r'"(?P<k>.*?)"(?=\s*:)',
        lambda m: '"' + re.sub(r'(?<!\\)"', r'\"', m.group('k')) + '"',
        s
    )
    return json.loads(json_clean.strip())

def parse_model_text(text: str):
    matches = re.findall(FENCED_JSON_PATTERN, text, re.DOTALL)
    if matches:
        try:
            return _try_clean_and_load(matches[0]), True
        except json.JSONDecodeError as e:
            print(f"[warn] fenced JSON 解析失败：{e}；尝试全文解析……")
    try:
        return _try_clean_and_load(text), True
    except json.JSONDecodeError as e:
        print(f"[warn] 全文 JSON 解析失败：{e}")
        return None, False

# =========================
# 分批评判与验证
# =========================
def validate_batch_result(rubric_items: List[str], parsed_result: Dict) -> bool:
    """
    验证模型返回是否包含所有 rubric_items，且文本严格匹配
    """
    if not isinstance(parsed_result, dict):
        return False
    results = parsed_result.get("results", [])
    if not isinstance(results, list):
        return False
    if len(results) != len(rubric_items):
        return False
    
    # 检查每个 rubric_item 是否严格匹配
    returned_items = [r.get("rubric_item", "") for r in results]
    for expected in rubric_items:
        if expected not in returned_items:
            return False
    
    return True

def query_rubric_batch(rubric_items: List[str], task: str, blocked: Dict, 
                       paper_content: str, pdf_path: str = None, 
                       extra_images: List[Tuple[str, bytes]] = None,
                       max_retries: int = 5) -> Tuple[Optional[List[Dict]], Dict]:
    """
    查询一批 rubric 条目，返回 (results_list, usage_metadata)
    最多重试 max_retries 次
    """
    global client
    if client is None:
        raise RuntimeError("GeminiClient 未初始化")
    
    rubric_input = {
        "task": task,
        "rubric_items": rubric_items,
        "blocked": blocked
    }
    rubric_json = json.dumps(rubric_input, ensure_ascii=False, indent=2)
    
    for attempt in range(max_retries):
        try:
            # 构造输入
            if pdf_path:
                # 有文件附件
                prompt = PROMPT_TEMPLATE.format(paper=PAPER_PLACEHOLDER, rubric=rubric_json)
                input_data = GeminiInput(
                    text=prompt,
                    file_path=pdf_path,
                    extra_images=extra_images
                )
            else:
                # 纯文本
                prompt = PROMPT_TEMPLATE.format(paper=paper_content, rubric=rubric_json)
                input_data = GeminiInput(
                    text=prompt,
                    extra_images=extra_images
                )
            
            # 调用 Client
            output = client.query(input_data)
            
            if not output.text:
                print(f"[warn] 批次尝试 {attempt+1}/{max_retries} 无文本内容")
                continue
            
            # 解析 JSON
            parsed, ok = parse_model_text(output.text)
            if not ok:
                print(f"[warn] 批次尝试 {attempt+1}/{max_retries} JSON 解析失败")
                continue
            
            # 验证结果
            if not validate_batch_result(rubric_items, parsed):
                print(f"[warn] 批次尝试 {attempt+1}/{max_retries} 验证失败：rubric_item 不匹配或数量不对")
                continue
            
            # 成功
            return parsed["results"], output.usage_metadata
            
        except Exception as e:
            print(f"[warn] 批次尝试 {attempt+1}/{max_retries} 请求异常：{e}")
            continue
    
    # 所有尝试失败
    return None, {}

# =========================
# 单文件处理（分批版本）
# =========================
def process_one_with_chunking(idx: int, pdf_path: str, rubric_content: Dict, chunk_size: int = 0, max_paper_chars: int = 150000, max_retries: int = 5):
    """
    处理单个文件，支持分批评判（Gemini 多模态）
    返回：(idx, result_dict, total_tokens)
    result_dict 包含 scores（按维度组织）和 total_tokens
    """
    print(f"[run] 处理 idx={idx}，文件={os.path.basename(pdf_path)}，chunk_size={chunk_size}")
    
    # 解析 rubric_content
    if not isinstance(rubric_content, dict):
        print(f"[err] idx={idx} rubric_content 格式错误")
        return idx, {"error": "invalid rubric_content"}, 0
    
    task = rubric_content.get("task", "")
    rubric = rubric_content.get("rubric", {})
    blocked = rubric_content.get("blocked", {})
    
    # 提取所有 rubric 条目（混合所有维度）
    all_items = []
    dimension_map = {}  # rubric_item -> dimension
    for dim in ["info_recall", "analysis", "presentation"]:
        items = rubric.get(dim, [])
        if isinstance(items, list):
            for item in items:
                all_items.append(item)
                dimension_map[item] = dim
    
    if not all_items:
        print(f"[warn] idx={idx} 无 rubric 条目")
        return idx, {"error": "no rubric items"}, 0
    
    # 准备文档内容（多模态策略）
    text_content = ""
    file_to_upload = None
    extra_images = []
    lower_name = pdf_path.lower()
    
    if lower_name.endswith('.pdf'):
        # PDF 默认作为附件上传
        file_to_upload = pdf_path
        print(f"[info] idx={idx} PDF 文件将作为附件上传")
    elif lower_name.endswith('.docx'):
        # DOCX 提取文本、表格和图片
        text_content, extra_images = _extract_docx_content(pdf_path)
        print(f"[info] idx={idx} DOCX 提取：文本长度={len(text_content)}，图片数量={len(extra_images)}")
        if not text_content and not extra_images:
            # 如果提取失败，尝试作为附件上传
            print(f"[warn] idx={idx} DOCX 提取失败，尝试作为纯文本")
    elif lower_name.endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff')):
        # 图片文件
        file_to_upload = pdf_path
        print(f"[info] idx={idx} 图片文件将作为附件上传")
    elif lower_name.endswith(('.txt', '.md', '.html')):
        # 纯文本文件
        try:
            with open(pdf_path, 'r', encoding='utf-8', errors='ignore') as rf:
                text_content = rf.read()
            print(f"[info] idx={idx} 文本文件读取：长度={len(text_content)}")
        except Exception as e:
            print(f"[warn] idx={idx} 读取文本文件失败：{e}")
            text_content = ""
    else:
        # 其他未知格式，尝试作为文本读取
        print(f"[warn] idx={idx} 未知格式，尝试读取为文本")
        try:
            with open(pdf_path, 'r', encoding='utf-8', errors='ignore') as rf:
                text_content = rf.read()
        except Exception:
            text_content = ""
    
    # 截断过长文本
    if text_content and len(text_content) > max_paper_chars:
        print(f"[info] idx={idx} 文本过长（{len(text_content)}），按 {max_paper_chars} 截断")
        text_content = text_content[:max_paper_chars]
    
    # 分批处理
    if chunk_size <= 0 or chunk_size >= len(all_items):
        # 不分批，一次性处理
        batches = [all_items]
    else:
        # 分批
        batches = [all_items[i:i+chunk_size] for i in range(0, len(all_items), chunk_size)]
    
    print(f"[info] idx={idx} 共 {len(all_items)} 个条目，分为 {len(batches)} 批")
    
    # 累积结果
    all_results = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_thoughts_tokens = 0
    total_tokens_sum = 0
    all_usage_metadata = []  # 保存每批次的完整 usageMetadata
    
    for batch_idx, batch_items in enumerate(batches):
        print(f"[info] idx={idx} 处理批次 {batch_idx+1}/{len(batches)}（{len(batch_items)} 个条目）")
        
        results, usage_metadata = query_rubric_batch(
            batch_items, task, blocked, text_content, 
            file_to_upload, 
            extra_images,
            max_retries
        )
        
        if results is None:
            print(f"[err] idx={idx} 批次 {batch_idx+1} 失败")
            return idx, {"error": f"batch {batch_idx+1} failed after {max_retries} retries"}, 0
        
        all_results.extend(results)
        all_usage_metadata.append(usage_metadata)
        
        # 累积各项 token 统计
        total_input_tokens += usage_metadata.get("promptTokenCount", 0)
        total_output_tokens += usage_metadata.get("candidatesTokenCount", 0)
        total_thoughts_tokens += usage_metadata.get("thoughtsTokenCount", 0)
        total_tokens_sum += usage_metadata.get("totalTokenCount", 0)
    
    # 按维度重新组织结果
    scores_by_dimension = {
        "info_recall": {},
        "analysis": {},
        "presentation": {}
    }
    
    for result in all_results:
        rubric_item = result.get("rubric_item", "")
        score = result.get("score", 0)
        reason = result.get("reason", "")
        evidence = result.get("evidence", "")
        
        dim = dimension_map.get(rubric_item)
        if dim:
            scores_by_dimension[dim][rubric_item] = {
                "score": score,
                "reason": reason,
                "evidence": evidence
            }
    
    result_dict = {
        "task": task,
        "scores": scores_by_dimension,
        "usage_summary": {
            "total_tokens": total_tokens_sum,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "thoughts_tokens": total_thoughts_tokens
        },
        "usage_metadata_per_batch": all_usage_metadata  # 保存每批次的完整 metadata
    }
    
    print(f"[ok] idx={idx} 完成，总 tokens={total_tokens_sum} (input={total_input_tokens}, output={total_output_tokens}, thoughts={total_thoughts_tokens})")
    return idx, result_dict, total_tokens_sum

# =========================
# 主程序
# =========================
def main():
    global client, TASKS_DATA, ORIGINAL_DATA
    
    # 获取默认配置
    config = get_default_config()
    
    parser = argparse.ArgumentParser(description="多模型学术内容评分（分批评判）")
    parser.add_argument("--pdf_dir", default=config['pdf_dir'], help="输入目录（包含模型子目录）")
    parser.add_argument("--out_jsonl", default=config['out_jsonl'], help="输出 jsonl 文件")
    parser.add_argument("--tasks_jsonl", default=config['tasks_jsonl'], help="任务和评分标准文件")
    parser.add_argument("--chunk_size", type=int, default=config['chunk_size'], help="分批大小（0=不分批）")
    parser.add_argument("--max_workers", type=int, default=config['max_workers'], help="并发数")
    parser.add_argument("--max_retries", type=int, default=config['max_retries'], help="最大重试次数")
    parser.add_argument("--max_paper_chars", type=int, default=config['max_paper_chars'], help="文本最大字符长度")
    parser.add_argument("--log_file", default=config['log_file'], help="日志文件路径（所有控制台输出将同步写入该文件）")
    parser.add_argument("--model", default=None, help="模型名称（可选，默认从 .env 读取）")
    parser.add_argument("--api_url", default=None, help="API URL（可选，默认从 .env 读取）")
    parser.add_argument("--token", default=None, help="API Token（可选，默认从 .env 读取）")
    parser.add_argument("--req_id", default=None, help="请求标识（可选，默认从 .env 读取）")
    args = parser.parse_args()
    
    # 初始化打印日志：所有 print 同步写入 log 文件
    setup_print_logger(args.log_file)
    print(f"[init] 日志文件：{args.log_file}")
    
    # 加载任务数据
    TASKS_DATA, ORIGINAL_DATA = load_tasks_data(args.tasks_jsonl)
    
    # 初始化 Gemini Client（从 .env 或命令行参数读取配置）
    try:
        client = GeminiClient(
            api_url=args.api_url,
            api_token=args.token,
            model=args.model,
            request_id=args.req_id,
            verbose=True
        )
        print(f"[init] Gemini Client 已初始化")
        print(f"  - 模型：{client.model}")
        print(f"  - API URL：{client.api_url}")
        print(f"  - 请求 ID：{client.request_id}")
    except ValueError as e:
        print(f"[err] 初始化失败：{e}")
        print(f"[提示] 请创建 .env 文件并配置 GEMINI_API_URL / GEMINI_API_TOKEN / GEMINI_MODEL，或通过命令行参数传入配置")
        return
    
    # 使用局部变量
    pdf_dir = args.pdf_dir
    out_jsonl = args.out_jsonl
    chunk_size = args.chunk_size
    max_retries = args.max_retries
    max_paper_chars = args.max_paper_chars
    
    if not TASKS_DATA:
        print(f"[info] 在 {args.tasks_jsonl} 未发现任何任务数据")
        return
    
    def _parse_idx_from_filename(name: str):
        m = re.match(r'^idx-(\d+)(?:\..+)?$', name, re.IGNORECASE)
        return int(m.group(1)) if m else None
    
    pairs = []  # (model, idx, fpath, content)
    if not os.path.isdir(pdf_dir):
        print(f"[info] 输入目录不存在：{pdf_dir}")
        return
    
    # 遍历一级目录的子目录，子目录名即 model
    for entry in os.listdir(pdf_dir):
        model_dir = os.path.join(pdf_dir, entry)
        if not os.path.isdir(model_dir):
            continue
        model_name = entry
        for fname in os.listdir(model_dir):
            idx_val = _parse_idx_from_filename(fname)
            if idx_val is None:
                continue
            fpath = os.path.join(model_dir, fname)
            content = TASKS_DATA.get(idx_val)
            if content is not None and os.path.exists(fpath):
                pairs.append((model_name, idx_val, fpath, content))
            else:
                if content is None:
                    print(f"[skip] 未在 {args.tasks_jsonl} 中找到 idx={idx_val} 的条目（model={model_name}）")
                if not os.path.exists(fpath):
                    print(f"[skip] 文件不存在：{fpath}")
    
    if not pairs:
        print("[info] 未找到任何可处理的配对")
        return
    
    print(f"[info] 共发现 {len(pairs)} 个配对，开始并行处理（chunk_size={chunk_size}）……")
    
    # 存储处理结果
    results = {}
    
    with ThreadPoolExecutor(max_workers=min(args.max_workers, len(pairs))) as pool:
        future_to_meta = {}
        for (model, idx, pdf_path, content) in pairs:
            fut = pool.submit(process_one_with_chunking, idx, pdf_path, content, chunk_size, max_paper_chars, max_retries)
            future_to_meta[fut] = (model, idx)
        for fut in as_completed(future_to_meta.keys()):
            model, _orig_idx = future_to_meta[fut]
            ridx, result_dict, total_tokens = fut.result()
            # 按 model 分桶，避免不同 model 同 idx 覆盖
            results.setdefault(model, {})[ridx] = result_dict
    
    # 以追加方式写入 jsonl
    print(f"[info] 以追加方式写入 {out_jsonl}...")
    out_dir = os.path.dirname(out_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_jsonl, "a", encoding="utf-8") as f:
        for model, idx_to_content in results.items():
            for idx, result_dict in idx_to_content.items():
                line_obj = {"model": model, "idx": idx, "result": result_dict}
                f.write(json.dumps(line_obj, ensure_ascii=False) + "\n")
    
    print(f"[done] 全部处理完成，结果已追加保存到 {out_jsonl}")

if __name__ == "__main__":
    main()
