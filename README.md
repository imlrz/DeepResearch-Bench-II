# DeepResearch-Bench-2

学术内容评分系统，支持多模态输入（PDF、DOCX、图片等），使用 Gemini API 进行智能评判。

## 快速开始

### 1. 配置环境（.env）

在项目根目录 `DeepResearch-Bench-2` 下创建 `.env` 文件，用于保存 API 配置和运行参数：

```bash
cd DeepResearch-Bench-2
touch .env
vim .env  # 或用你喜欢的编辑器
```

**必需配置**（请替换为你自己的值）：

```bash
GEMINI_API_URL=https://your-api-endpoint.com/v1/chat/completions
GEMINI_API_TOKEN=your-api-token
GEMINI_MODEL=gemini-2.5-pro
GEMINI_REQUEST_ID=eval-request-id

PDF_DIR=grok
OUT_JSONL=eval_result_grok.jsonl
TASKS_JSONL=tasks_and_rubrics.jsonl
CHUNK_SIZE=50
MAX_WORKERS=10
MAX_RETRIES=5
MAX_PAPER_CHARS=150000
```

### 2. 安装依赖（支持 uv）

#### 方式 A：使用 uv（推荐）

项目已提供 `pyproject.toml`，可以直接用 `uv` 管理虚拟环境和依赖：

```bash
# 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建/同步虚拟环境并安装依赖
cd DeepResearch-Bench-2
uv sync
```

##### 如何检查 uv 是否安装成功？

在终端执行以下命令（任意一个或全部）：

```bash
# 1. 查看版本（推荐）
uv --version

# 2. 查看可执行文件路径
which uv

# 3. 查看帮助信息
uv --help
```

- 如果 `uv --version` 输出类似 `uv 0.x.y`，说明安装成功；
- 如果提示 `command not found` 或找不到命令，说明未安装或未加入 `PATH`。

#### 方式 B：使用 pip

```bash
pip install requests python-docx
```

### 3. 运行评估

#### 用 uv 运行（推荐）

```bash
cd DeepResearch-Bench-2
uv run python run_evaluation.py
```

#### 直接用 python 运行

```bash
cd DeepResearch-Bench-2

# 使用 .env 配置运行
python run_evaluation.py

# 或通过命令行参数覆盖配置
python run_evaluation.py \
    --pdf_dir grok \
    --out_jsonl result.jsonl \
    --chunk_size 50
```

## 功能特性

### ✅ 多模态支持
- **PDF 文件**：直接作为附件上传，保留完整的图片和表格信息
- **DOCX 文件**：自动提取文本、表格（转 Markdown）和图片
- **图片文件**：支持 PNG、JPEG、WebP、GIF、BMP、TIFF
- **文本文件**：支持 TXT、MD、HTML

### ✅ 配置管理
- 使用 `.env` 文件管理敏感配置
- 支持环境变量和命令行参数覆盖
- 灵活的配置优先级机制

### ✅ 高性能
- 分批评判（避免超长上下文）
- 多线程并发处理
- 自动重试机制
- 详细的进度日志

### ✅ 模块化设计
- API 客户端独立封装（`gemini_client.py`）
- 清晰的接口定义（`GeminiInput` / `GeminiOutput`）
- 易于扩展到其他 AI 模型

## 文件结构

```
DeepResearch-Bench-2/
├── gemini_client.py           # Gemini API 客户端（仅一处负责调用 API）
├── run_evaluation.py          # 主评估脚本（评测逻辑，不含任何 API 细节）
├── aggregate_scores.py        # 分数聚合工具
├── tasks_and_rubrics.jsonl    # 任务和评分标准
├── human/                     # 输入数据（各模型生成的文本/PDF/DOCX 等）
├── pyproject.toml             # 使用 uv / pip 管理依赖的配置
├── README.md                  # 本文档
└── .env                       # 实际配置（需用户自行创建，已在 .gitignore 中）
```

## 配置说明

### .env 配置项

| 配置项 | 说明 | 是否必需 | 默认值 |
|--------|------|----------|--------|
| `GEMINI_API_URL` | API 端点 URL | ✅ 必需 | - |
| `GEMINI_API_TOKEN` | API 访问令牌 | ✅ 必需 | - |
| `GEMINI_MODEL` | 模型名称 | ✅ 必需 | - |
| `GEMINI_REQUEST_ID` | 请求标识符 | ❌ 可选 | `default-request` |
| `PDF_DIR` | 输入目录 | ❌ 可选 | `grok` |
| `OUT_JSONL` | 输出文件 | ❌ 可选 | `eval_result_grok.jsonl` |
| `TASKS_JSONL` | 任务文件 | ❌ 可选 | `tasks_and_rubrics.jsonl` |
| `CHUNK_SIZE` | 分批大小 | ❌ 可选 | `50` |
| `MAX_WORKERS` | 最大并发数 | ❌ 可选 | `10` |
| `MAX_RETRIES` | 最大重试次数 | ❌ 可选 | `5` |
| `MAX_PAPER_CHARS` | 文本最大长度 | ❌ 可选 | `150000` |
| `LOG_FILE` | 日志文件路径（所有控制台输出会同步写入） | ❌ 可选 | `run_evaluation.log` |

## 使用示例

### 基本使用

```bash
# 1. 配置 .env
cp .env.example .env
vim .env

# 2. 运行评估
python run_evaluation.py
```

### 自定义参数

```bash
# 指定输入输出路径
python run_evaluation.py \
    --pdf_dir my_data \
    --out_jsonl my_result.jsonl

# 调整性能参数
python run_evaluation.py \
    --chunk_size 30 \
    --max_workers 5 \
    --max_retries 3

# 覆盖模型配置
python run_evaluation.py \
    --model gemini-2.0-flash
```

### 编程式使用

```python
from gemini_client import GeminiClient, GeminiInput

# 初始化客户端（从 .env 读取配置）
client = GeminiClient()

# 发送查询
input_data = GeminiInput(
    text="请分析这个文档",
    file_path="document.pdf"
)
output = client.query(input_data)

print(f"回复：{output.text}")
print(f"Token 使用：{output.usage_metadata}")
```

## 输出格式

评估结果保存为 JSONL 格式：

```jsonl
{"model": "model_name", "idx": 1, "result": {...}}
{"model": "model_name", "idx": 2, "result": {...}}
```

每个 result 包含：
- `task`: 任务描述
- `scores`: 按维度组织的评分结果
  - `info_recall`: 信息回忆维度
  - `analysis`: 分析维度
  - `presentation`: 呈现维度
- `usage_summary`: Token 使用统计
- `usage_metadata_per_batch`: 每批次详细统计

## 常见问题

### Q: 如何配置 API 密钥？
A: 创建 `.env` 文件并设置 `GEMINI_API_TOKEN`。详见 [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)

### Q: 支持哪些文件格式？
A: 支持 PDF、DOCX、PNG、JPEG、WebP、GIF、BMP、TIFF、TXT、MD、HTML。

### Q: 如何查看上传了什么内容？
A: 运行时会打印 `[上传]` 日志，显示文本段数和文件类型。

### Q: PDF 中的图片会丢失吗？
A: 不会。PDF 以附件形式完整上传到 Gemini，图片和表格都会保留。

### Q: DOCX 中的图片如何处理？
A: 自动提取并作为额外的图片附件一起上传。

### Q: 如何提高处理速度？
A: 调整 `MAX_WORKERS` 增加并发数，或调整 `CHUNK_SIZE` 减少批次数。

### Q: 遇到 Rate Limit 错误怎么办？
A: 减少 `MAX_WORKERS` 并发数，增加重试次数 `MAX_RETRIES`。

## 文档索引

- **[README_CLIENT.md](README_CLIENT.md)** - Gemini Client 详细使用文档
- **[CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)** - 配置管理完整指南
- **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)** - 代码重构说明
- **[example_client_usage.py](example_client_usage.py)** - 客户端使用示例代码

## 安全提示

⚠️ **重要**：
1. **不要**将 `.env` 文件提交到 Git（已在 `.gitignore` 中）
2. **不要**在代码中硬编码 API 密钥
3. **不要**在公开场合分享 API 密钥
4. 定期更换 API 密钥
5. 限制 `.env` 文件权限：`chmod 600 .env`

## 开发指南

### 添加新的 AI 模型支持

1. 创建新的 Client 类（例如 `OpenAIClient`）
2. 实现相同的接口（`query` 方法）
3. 使用相同的 `Input/Output` 数据类
4. 在 `run_evaluation.py` 中根据配置选择 Client

示例：
```python
class OpenAIClient:
    def query(self, input_data: GeminiInput) -> GeminiOutput:
        # 实现 OpenAI API 调用
        pass
```

### 运行测试

```bash
# 运行示例代码
python example_client_usage.py

# 测试单个文件
python -c "
from gemini_client import GeminiClient, GeminiInput
client = GeminiClient()
output = client.query(GeminiInput(text='Hello'))
print(output.text)
"
```

## 许可证

（根据项目实际情况添加）

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

### v2.0.0 (2024)
- ✅ 重构代码，API 客户端独立
- ✅ 支持 .env 配置管理
- ✅ 增强 DOCX 图片提取
- ✅ PDF 默认附件上传
- ✅ 完善文档和示例

### v1.0.0
- 初始版本
