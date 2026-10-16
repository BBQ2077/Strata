"""Hugging Face tokenizer files (an EXL3 model folder) -> Strata's tokenizer folder.

Strata's server tokenizes and renders the chat itself (tools/strata_tokenizer.py) and hands the engine token ids, so
it needs the model's vocabulary in its own format: vocab.json (token -> id), merges.txt, token_type.json (GGUF token
types: 1 normal, 3 control, 4 user defined, 5 unused), tokenizer.json (a small description) and chat_template.jinja.
Upstream setup extracts them from a GGUF; an EXL3 model only has the Hugging Face files, which carry the same facts:

  * added tokens with "special": true  -> 3 (CONTROL)      <|im_start|>, <|im_end|>, <|endoftext|>, ...
  * added tokens with "special": false -> 4 (USER_DEFINED) <think>, </think>, <tool_call>, ... (as llama.cpp's
    converter writes them, which Strata's frontend relies on to mark literal tags in message text)
  * ids past the tokenizer up to config.json's vocab_size (the padded embedding rows) -> "[PAD<id>]", 5 (UNUSED)

Strata's BPE uses the Qwen3.5 pre-tokenizer pattern; a model whose tokenizer.json splits differently would get other
ids than it was trained with, so the export refuses it instead of producing a tokenizer that is quietly wrong.

usage: python hf_tokenizer.py --model <EXL3 folder> --out <pack dir> [--check]
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))

CONTROL, USER_DEFINED, NORMAL, UNUSED = 3, 4, 1, 5
# the same split as strata_tokenizer.QWEN35_PATTERN, in the spelling Qwen's tokenizer.json uses ((?i:...) for the
# contractions instead of the bracketed letter pairs llama.cpp writes)
HF_QWEN35_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
                     r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def _merge_text(m) -> str:
    if isinstance(m, str):
        return m
    if isinstance(m, (list, tuple)) and len(m) == 2:
        return f"{m[0]} {m[1]}"
    raise ValueError(f"unexpected merge entry {m!r}")


def _chat_template(model: pathlib.Path) -> str | None:
    f = model / "chat_template.jinja"
    if f.is_file():
        return f.read_text(encoding="utf-8")
    try:
        cfg = json.loads((model / "tokenizer_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    tpl = cfg.get("chat_template")
    if isinstance(tpl, list):                    # named templates: the default one
        tpl = next((t.get("template") for t in tpl if isinstance(t, dict) and t.get("name") == "default"), None)
    return tpl if isinstance(tpl, str) else None


def _split_patterns(node) -> list[str]:
    """Every Split pre-tokenizer regex in tokenizer.json's pre_tokenizer tree, in order."""
    out = []
    if isinstance(node, dict):
        if node.get("type") == "Split" and isinstance(node.get("pattern"), dict) and "Regex" in node["pattern"]:
            out.append(node["pattern"]["Regex"])
        for v in node.values():
            out += _split_patterns(v)
    elif isinstance(node, list):
        for v in node:
            out += _split_patterns(v)
    return out


def read_hf(model) -> dict:
    model = pathlib.Path(model)
    tj = json.loads((model / "tokenizer.json").read_text(encoding="utf-8"))
    m = tj.get("model") or {}
    if m.get("type") != "BPE":
        raise ValueError(f"{model.name}: tokenizer model {m.get('type')!r}, Strata needs byte-level BPE")
    pre = _split_patterns(tj.get("pre_tokenizer") or {})
    vocab: dict[str, int] = dict(m["vocab"])
    types: dict[int, int] = {i: NORMAL for i in vocab.values()}
    for a in tj.get("added_tokens") or []:
        vocab[a["content"]] = int(a["id"])
        types[int(a["id"])] = CONTROL if a.get("special") else USER_DEFINED
    try:
        cfg = json.loads((model / "config.json").read_text(encoding="utf-8"))
        n_cfg = int((cfg.get("text_config") or cfg).get("vocab_size") or 0)
    except (OSError, ValueError, TypeError):
        n_cfg = 0
    size = max(n_cfg, max(vocab.values()) + 1)
    tokens: list[str | None] = [None] * size
    for t, i in vocab.items():
        if tokens[i] is not None and tokens[i] != t:
            raise ValueError(f"id {i} is both {tokens[i]!r} and {t!r}")
        tokens[i] = t
    for i, t in enumerate(tokens):
        if t is None:
            tokens[i] = f"[PAD{i}]"
            types[i] = UNUSED
    return {"tokens": tokens, "types": [types[i] for i in range(size)],
            "merges": [_merge_text(x) for x in m.get("merges") or []], "pre_tokenizer": pre,
            "chat_template": _chat_template(model)}


def export(model, out_dir) -> dict:
    import strata_tokenizer as ST
    h = read_hf(model)
    if h["pre_tokenizer"] not in ([ST.QWEN35_PATTERN], [HF_QWEN35_PATTERN]):
        raise ValueError("the model's pre-tokenizer is not the Qwen3.5 pattern Strata's tokenizer implements: "
                         f"{h['pre_tokenizer']}")
    tk = ST.Tokenizer(h["tokens"], h["merges"], h["types"])          # validates merges against the vocabulary
    special = {name: tk.ids[name] for name in ("<|endoftext|>", "<|im_start|>", "<|im_end|>") if name in tk.ids}
    out = pathlib.Path(out_dir) / "tokenizer"
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"model": "gpt2", "pre": "qwen35", "vocab_size": len(h["tokens"]), "n_merges": len(h["merges"]),
           "special_ids": special, "add_bos_token": False, "pre_pattern": ST.QWEN35_PATTERN,
           "pre_pattern_source": "Hugging Face tokenizer.json (EXL3 model folder)"}
    (out / "vocab.json").write_text(json.dumps(tk.ids, ensure_ascii=False), encoding="utf-8")
    (out / "merges.txt").write_text("\n".join(h["merges"]), encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps(h["types"]), encoding="utf-8")
    (out / "tokenizer.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    if h["chat_template"]:
        (out / "chat_template.jinja").write_text(h["chat_template"], encoding="utf-8", newline="\n")
    return cfg


SAMPLES = ["Hello, world!", "你好，世界。這是一段繁體中文測試。", "def f(x):\n    return x ** 2\n",
           "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nok",
           "naïve café — résumé 😀 \u0301combining", "  leading spaces\r\n\ttabs 12345 67.89"]


def check(model, out_dir) -> list[str]:
    """Strata's tokenizer (from the exported folder) against the model's own tokenizer.json (Hugging Face
    `tokenizers`, when it is installed): the ids must be identical."""
    import strata_tokenizer as ST
    try:
        from tokenizers import Tokenizer as HF
    except ImportError:
        return ["tokenizers is not installed: comparison skipped"]
    t = pathlib.Path(out_dir) / "tokenizer"
    vocab = json.loads((t / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for s, i in vocab.items():
        tokens[i] = s
    tk = ST.Tokenizer(tokens, (t / "merges.txt").read_text(encoding="utf-8").split("\n"),
                      json.loads((t / "token_type.json").read_text()))
    hf = HF.from_file(str(pathlib.Path(model) / "tokenizer.json"))
    bad = []
    for s in SAMPLES:
        a = tk.encode(s, parse_special=True)
        b = hf.encode(s, add_special_tokens=False).ids
        if a != b:
            bad.append(f"mismatch on {s[:30]!r}: strata {a[:12]} hf {b[:12]}")
    return bad


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="EXL3 model folder (tokenizer.json, config.json)")
    ap.add_argument("--out", required=True, help="pack folder; the tokenizer goes to <out>/tokenizer")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    cfg = export(a.model, a.out)
    print(f"tokenizer/: vocab {cfg['vocab_size']}, merges {cfg['n_merges']}, specials {cfg['special_ids']}")
    if a.check:
        bad = check(a.model, a.out)
        for b in bad:
            print(b)
        return 1 if any(b.startswith("mismatch") for b in bad) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
