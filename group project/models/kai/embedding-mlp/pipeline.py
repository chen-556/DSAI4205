"""Frozen Qwen features and an independently fitted sklearn MLP.

This module is embedded verbatim into the unified notebook; it has no API calls.
"""
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def embedding_digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def adapter_digest(folder):
    folder = Path(folder)
    files = sorted(folder.glob("adapter_model*.safetensors"))
    config = folder / "adapter_config.json"
    if not files or not config.is_file():
        raise FileNotFoundError(f"Adapter 不完整: {folder}")
    digest = hashlib.sha256()
    for path in [config, *files]:
        digest.update(path.name.encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def validate_cached_counterfactuals(rows, train, usable_ids, expected_count, text_hash, blocked):
    required = {"parent_row_id", "source_text", "source_label", "cf_text", "cf_label", "accepted"}
    if not required <= set(rows.columns):
        raise ValueError("反事实导出缺少父样本、文本、标签或核验字段。")
    parents = train.set_index("row_id", verify_integrity=True)
    if (len(rows) != expected_count or len(rows) == 0 or not rows.parent_row_id.is_unique
            or not rows.parent_row_id.isin(usable_ids).all() or not rows.accepted.eq(True).all()):
        raise ValueError("反事实接受数、核验标记或训练父样本与原协议不符。")
    if (not rows.source_label.isin([0, 1]).all()
            or not rows.cf_label.eq(1 - rows.source_label).all()
            or not rows.source_label.eq(rows.parent_row_id.map(parents.Sentiment)).all()
            or not rows.source_text.eq(rows.parent_row_id.map(parents.text)).all()
            or not rows.cf_text.map(lambda s: isinstance(s, str) and bool(s.strip())).all()):
        raise ValueError("反事实文本或正负翻转标签与原训练行不符。")
    hashes = rows.cf_text.map(text_hash)
    if not hashes.is_unique or set(hashes) & set(blocked):
        raise ValueError("反事实文本重复，或与原始训练/验证/测试文本重合。")
    return rows.sort_values("parent_row_id").reset_index(drop=True)


def guard_embedding_protocol(path, protocol):
    """Prevent reuse of completed adapters/heads after changing any experiment setting."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("续跑协议不同（数据/缓存/模型/超参数/版本），请创建新运行目录。")
    elif any(path.parent.glob("*_seed*/summary.json")) or (path.parent / "mlp").exists():
        raise ValueError("已有模型但缺少运行协议，无法安全续跑。")
    path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")


def last_valid_pool(hidden, mask):
    """Select the last non-padding token for either padding side."""
    if mask.ndim != 2 or not torch.all(mask.sum(dim=1) > 0):
        raise ValueError("每条输入必须至少包含一个有效 token。")
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    last = positions.masked_fill(mask.eq(0), -1).max(dim=1).values
    return hidden[torch.arange(len(hidden), device=hidden.device), last]


def freeze_decoder(decoder):
    decoder.requires_grad_(False)
    decoder.eval()
    if hasattr(decoder, "gradient_checkpointing_disable"):
        decoder.gradient_checkpointing_disable()
    decoder.config.use_cache = False
    if any(p.requires_grad for p in decoder.parameters()):
        raise RuntimeError("特征模型尚未完全冻结。")
    return decoder


def extract_frozen_vectors(decoder, tokenizer, input_ids, batch_size, device):
    """Input-only hidden states. No labels, generation, LM logits or training graph."""
    if not input_ids or batch_size <= 0:
        raise ValueError("需要非空输入和正数 batch size。")
    freeze_decoder(decoder)
    vectors = []
    with torch.inference_mode():
        for offset in range(0, len(input_ids), batch_size):
            rows = input_ids[offset:offset + batch_size]
            batch = tokenizer.pad([{"input_ids": ids, "attention_mask": [1] * len(ids)}
                                   for ids in rows], padding=True, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
            batch["position_ids"] = (batch["attention_mask"].cumsum(-1) - 1).clamp_min(0)
            outputs = decoder(**batch, use_cache=False, return_dict=True)
            pooled = last_valid_pool(outputs.last_hidden_state, batch["attention_mask"]).float()
            vectors.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy())
    result = np.concatenate(vectors).astype(np.float32)
    if not np.isfinite(result).all() or np.any(np.linalg.norm(result, axis=1) < 0.99):
        raise ValueError("提取到非有限值或零向量。")
    return result


def feature_cache(cache_root, spec, frame, input_ids, truncated, compute):
    """Content-addressed arrays with exact input/order/weight/config validation."""
    if set(frame.columns) != {"row_id", "text"} or not frame.row_id.is_unique:
        raise ValueError("特征提取只接受唯一 row_id 与 text，不接受标签或教师解释。")
    if len(frame) != len(input_ids) or len(frame) != len(truncated):
        raise ValueError("输入行数与 token 数组不匹配。")
    ids = frame.row_id.astype(str).to_numpy(dtype=str)
    signature = {"schema": 1, "representation": spec,
                 "rows": frame.astype(str).to_dict("records"), "input_ids": input_ids,
                 "truncated": [bool(x) for x in truncated]}
    key = embedding_digest(signature)
    path = Path(cache_root) / (key + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            features = data["features"]
            meta = json.loads(str(data["metadata"]))
            valid = (meta["key"] == key and np.array_equal(data["row_ids"], ids)
                     and np.array_equal(data["truncated"], np.asarray(truncated, dtype=bool))
                     and meta["features_sha256"] == hashlib.sha256(features.tobytes()).hexdigest())
        if not valid or features.shape != (len(frame), spec["dimension"]) or not np.isfinite(features).all():
            raise ValueError(f"向量缓存损坏或不匹配: {path}")
        return features, path, True
    features = np.asarray(compute(), dtype=np.float32)
    if features.shape != (len(frame), spec["dimension"]) or not np.isfinite(features).all():
        raise ValueError("向量形状或数值无效。")
    meta = {"key": key, "representation": spec,
            "features_sha256": hashlib.sha256(features.tobytes()).hexdigest()}
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, features=features, row_ids=ids,
                            truncated=np.asarray(truncated, dtype=bool), metadata=json.dumps(meta))
    os.replace(temporary, path)
    return features, path, False


def fit_independent_mlp(train_x, train_y, val_x, val_y, seed, settings):
    """Fit scaling on train only; select the MLP epoch by validation macro-F1."""
    if set(np.unique(train_y)) != {0, 1, 2, 3}:
        raise ValueError("MLP 训练集必须包含四个类别。")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x).astype(np.float32)
    val_scaled = scaler.transform(val_x).astype(np.float32)
    model = MLPClassifier(hidden_layer_sizes=(settings["hidden"],), activation="relu", solver="adam",
                          alpha=settings["alpha"], batch_size=min(settings["batch_size"], len(train_x)),
                          learning_rate_init=settings["learning_rate"], shuffle=True, random_state=seed,
                          early_stopping=False, max_iter=1)
    best, best_f1, best_epoch, stale, history = None, -1.0, 0, 0, []
    # A small head is faster with a bounded BLAS thread pool; GPU is reserved for Qwen.
    with threadpool_limits(limits=2):
        for epoch in range(1, settings["max_epochs"] + 1):
            model.partial_fit(train_scaled, train_y, classes=np.arange(4))
            pred = model.predict(val_scaled)
            score = f1_score(val_y, pred, labels=range(4), average="macro", zero_division=0)
            history.append({"epoch": epoch, "train_loss": float(model.loss_), "validation_macro_f1": score})
            if not np.isfinite(model.loss_):
                raise ValueError("MLP loss 非有限值。")
            if score > best_f1 + 1e-12:
                best, best_f1, best_epoch, stale = copy.deepcopy(model), score, epoch, 0
            else:
                stale += 1
            if stale >= settings["patience"]:
                break
    if best is None:
        raise ValueError("MLP 未完成训练，请检查最大 epoch 数。")
    return {"scaler": scaler, "classifier": best, "best_epoch": best_epoch,
            "validation_macro_f1": best_f1, "history": history, "seed": seed,
            "settings": settings, "train_rows": len(train_x), "feature_dimension": train_x.shape[1]}


def predict_independent_mlp(bundle, features):
    with threadpool_limits(limits=2):
        probabilities = bundle["classifier"].predict_proba(bundle["scaler"].transform(features))
    if not np.array_equal(bundle["classifier"].classes_, np.arange(4)):
        raise ValueError("分类器标签顺序错误。")
    return probabilities.argmax(axis=1), probabilities


def embedding_metrics(gold, predicted, label_names):
    return {"accuracy": accuracy_score(gold, predicted),
            "macro_f1": f1_score(gold, predicted, labels=range(4), average="macro", zero_division=0),
            "classification_report": classification_report(gold, predicted, labels=range(4),
                target_names=label_names, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(gold, predicted, labels=range(4)).tolist()}


def embedding_bootstrap(gold, predictions, seeds, count):
    comparisons = [("qwen_label", "qwen_base"), ("qwen_rationale", "qwen_label"),
                   ("qwen_rationale_cf", "qwen_rationale"), ("qwen_embedding", "qwen_base"),
                   ("qwen_rationale_cf", "qwen_embedding")]
    rows = []
    for seed in seeds:
        for treatment, control in comparisons:
            rng = np.random.default_rng(2026 + seed)
            def score(indices, arm):
                return f1_score(gold[indices], predictions[(arm, seed)][indices], labels=range(4),
                                average="macro", zero_division=0)
            differences = []
            for _ in range(count):
                idx = rng.integers(0, len(gold), len(gold))
                differences.append(score(idx, treatment) - score(idx, control))
            low, high = np.percentile(differences, [2.5, 97.5])
            idx = np.arange(len(gold))
            rows.append({"experiment": treatment, "control": control, "seed": seed,
                         "delta_macro_f1": score(idx, treatment) - score(idx, control),
                         "delta_ci_low": low, "delta_ci_high": high, "resamples": count})
    return rows
