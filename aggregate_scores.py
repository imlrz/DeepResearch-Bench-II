#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import csv
import argparse
from collections import defaultdict


def try_parse_json_maybe(s):
    if isinstance(s, dict):
        return s
    if isinstance(s, str):
        s_strip = s.strip()
        if not s_strip:
            return None
        try:
            return json.loads(s_strip)
        except Exception:
            return None
    return None


def mean_or_none(values):
    nums = [v for v in values if isinstance(v, (int, float))]
    if not nums:
        return None
    return sum(nums) / len(nums)


def compute_dimension_averages(payload):
    """
    计算维度平均分（仅统计得分为 1 的比率）和 blocked_rate（得分为 -1 的比率）
    
    payload 期望结构：
    {
        "task": "...",
        "scores": {
            "info_recall": {
                "rubric_item_1": {"score": 1, "reason": "...", "evidence": "..."},
                "rubric_item_2": {"score": 0, "reason": "...", "evidence": "..."},
                ...
            },
            "analysis": {...},
            "presentation": {...}
        },
        "total_tokens": 1234
    }
    """
    if not isinstance(payload, dict):
        return {}

    out = {}
    scores = payload.get("scores", {}) or {}
    
    # 映射：json键 -> 输出维度名
    dim_map = [("info_recall", "inforecall"), ("analysis", "analysis"), ("presentation", "presentation")]

    total_ones = 0
    total_minus_ones = 0
    total_items = 0

    for json_key, out_key in dim_map:
        dim_obj = scores.get(json_key)
        if isinstance(dim_obj, dict) and dim_obj:
            # 提取所有条目的 score
            score_values = []
            for item_key, item_val in dim_obj.items():
                if isinstance(item_val, dict):
                    score = item_val.get("score")
                    if isinstance(score, (int, float)):
                        score_values.append(score)
            
            # 统计得分为 1 的比率
            ones_count = sum(1 for s in score_values if s == 1)
            minus_ones_count = sum(1 for s in score_values if s == -1)
            items_count = len(score_values)
            
            if items_count > 0:
                out[out_key] = ones_count / items_count
            else:
                out[out_key] = None
            
            total_ones += ones_count
            total_minus_ones += minus_ones_count
            total_items += items_count
        else:
            out[out_key] = None

    # 计算总分（所有维度中得分为 1 的比率）
    if total_items > 0:
        out["total"] = total_ones / total_items
        out["blocked_rate"] = total_minus_ones / total_items
    else:
        out["total"] = None
        out["blocked_rate"] = None

    return out


def load_task_metadata(tasks_file="tasks_and_rubrics.jsonl"):
    """
    从 tasks_and_rubrics.jsonl 加载 idx -> (language, theme) 映射
    """
    idx_to_meta = {}
    if not os.path.exists(tasks_file):
        print(f"[warning] 未找到任务元数据文件：{tasks_file}，将跳过主题和语言信息")
        return idx_to_meta
    
    try:
        with open(tasks_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    idx = obj.get("idx")
                    if idx is not None:
                        try:
                            idx_int = int(idx)
                            language = obj.get("language", "")
                            theme = obj.get("theme", "")
                            idx_to_meta[idx_int] = {"language": language, "theme": theme}
                        except Exception:
                            continue
                except Exception:
                    continue
    except Exception as e:
        print(f"[warning] 读取任务元数据文件时出错：{e}，将跳过主题和语言信息")
    
    return idx_to_meta


def main():
    parser = argparse.ArgumentParser(description="聚合三分类评分结果为 CSV（包含 blocked_rate）")
    parser.add_argument("--input", default="merged.jsonl", help="输入jsonl文件路径（由评分脚本输出）")
    parser.add_argument("--output-prefix", default="agg_scores", help="输出csv文件路径前缀")
    parser.add_argument("--tasks-file", default="tasks_and_rubrics.jsonl", help="任务元数据文件路径（用于获取主题和语言信息）")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"[error] 未找到输入文件：{args.input}")
    
    # 加载任务元数据（idx -> language, theme）
    idx_to_meta = load_task_metadata(args.tasks_file)

    # 聚合结构： (idx -> model -> list of per-line averages dict)
    idx_model_to_records = defaultdict(lambda: defaultdict(list))
    models_set = set()

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            model = obj.get("model")
            idx = obj.get("idx")
            
            # 新格式使用 "result" 字段
            result_content = obj.get("result")
            if result_content is None:
                # 兼容旧格式的 "new_content"
                result_content = obj.get("new_content")
            
            if model is None or idx is None:
                continue
            try:
                idx_int = int(idx)
            except Exception:
                continue

            payload = try_parse_json_maybe(result_content)
            if payload is None:
                continue

            dims = compute_dimension_averages(payload)
            idx_model_to_records[idx_int][model].append(dims)
            models_set.add(model)

    if not idx_model_to_records:
        raise SystemExit("[info] 未解析到任何有效记录，退出")

    # 计算聚合数据：idx_model_dim -> score
    # 结构：{idx: {model: {dim: score}}}
    aggregated_data = {}
    for idx in idx_model_to_records:
        aggregated_data[idx] = {}
        per_model = idx_model_to_records[idx]
        for model in per_model:
            recs = per_model[model]
            agg = {}
            for dim in ["inforecall", "analysis", "presentation", "total", "blocked_rate"]:
                vals = [r.get(dim) for r in recs if r.get(dim) is not None]
                agg[dim] = mean_or_none(vals)
            aggregated_data[idx][model] = agg

    models = sorted(models_set)
    # (数据键, 文件后缀)
    dim_configs = [("inforecall", "inforecall"), ("analysis", "analysis"), ("presentation", "presentation"), ("total", "total"), ("blocked_rate", "blocked")]
    
    # 为每个维度生成一个CSV
    out_dir = os.path.dirname(args.output_prefix) if os.path.dirname(args.output_prefix) else "."
    if out_dir and out_dir != ".":
        os.makedirs(out_dir, exist_ok=True)

    for dim_key, dim_suffix in dim_configs:
        # 构建数据矩阵
        # 计算每个模型在该维度的均分（用于排序列）
        model_avg = {}
        for model in models:
            scores = []
            for idx in aggregated_data:
                if model in aggregated_data[idx]:
                    score = aggregated_data[idx][model].get(dim_key)
                    if score is not None:
                        scores.append(score)
            model_avg[model] = mean_or_none(scores)
        
        # 按模型均分降序排序
        sorted_models = sorted(models, key=lambda m: model_avg.get(m) if model_avg.get(m) is not None else -1, reverse=True)
        
        # 计算每个idx在该维度的均分（用于排序行）
        idx_avg = {}
        for idx in aggregated_data:
            scores = []
            for model in models:
                if model in aggregated_data[idx]:
                    score = aggregated_data[idx][model].get(dim_key)
                    if score is not None:
                        scores.append(score)
            idx_avg[idx] = mean_or_none(scores)
        
        # 按idx均分降序排序
        sorted_idxs = sorted(aggregated_data.keys(), key=lambda i: idx_avg.get(i) if idx_avg.get(i) is not None else -1, reverse=True)
        
        # 写入CSV
        output_file = f"{args.output_prefix}_{dim_suffix}.csv"
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            
            # 表头：idx, language, theme, model1, model2, ..., avg
            writer.writerow(["idx", "language", "theme"] + sorted_models + ["avg"])
            
            # 数据行
            for idx in sorted_idxs:
                # 获取该idx的语言和主题
                meta = idx_to_meta.get(idx, {})
                language = meta.get("language", "")
                theme = meta.get("theme", "")
                
                row = [idx, language, theme]
                for model in sorted_models:
                    if model in aggregated_data[idx]:
                        score = aggregated_data[idx][model].get(dim_key)
                        row.append("" if score is None else f"{score:.6f}")
                    else:
                        row.append("")
                # 添加该idx的均分
                avg = idx_avg.get(idx)
                row.append("" if avg is None else f"{avg:.6f}")
                writer.writerow(row)
            
            # 最后一行：模型均分
            avg_row = ["avg", "", ""]  # idx, language, theme 位置留空
            for model in sorted_models:
                avg = model_avg.get(model)
                avg_row.append("" if avg is None else f"{avg:.6f}")
            # 右下角：总均分
            all_scores = []
            for idx in aggregated_data:
                for model in aggregated_data[idx]:
                    score = aggregated_data[idx][model].get(dim_key)
                    if score is not None:
                        all_scores.append(score)
            total_avg = mean_or_none(all_scores)
            avg_row.append("" if total_avg is None else f"{total_avg:.6f}")
            writer.writerow(avg_row)
        
        print(f"[done] 写入完成：{output_file}")


if __name__ == "__main__":
    main()
