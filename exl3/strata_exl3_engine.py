"""Strata engine protocol on ExLlamaV3 (EXL3), for native Windows (ROCm gfx1201 / CUDA).

Strata's server (serve/server.py, class StrataEngine) starts its engine as `<exe> --serve <args>` and talks to it over
stdin/stdout, one line per message.  This file speaks that protocol, so the whole Strata front end (OpenAI / Anthropic
API, tool calls, reasoning parser, web UI, Monitor) runs unchanged on an EXL3 model:

  server -> engine                           engine -> server
  GEN <max_new> [k=v ...] <id,id,...>        INFO k=v ...            (before READY)
  GENI ...                                   READY <ctx> stop
  STOP                                       RESUME <reused>         (prompt tokens taken from the cache)
  VRAM [mib]                                 PP <read> <total> <ms> <tok_s>
  SAVE <path> / RESTORE <path>               T <id>                  (one per generated token, stop token included)
  QUIT                                       DONE <15 fields>        (see _done)
                                             ERR <reason> / SERR <kind> <published> <reason>

It runs inside the portable EXL3 runtime (runtime.py `script` mode), which sets the DLL directories, sys.path and
the isolated environment.  The protocol goes to the real stdout; everything else that writes to stdout (ExLlamaV3's
load progress, the runtime's audit line, native prints) is sent to stderr - Strata's engine log - so a stray line can
never be read as a protocol message.

Multi-turn reuse: the generator's page cache matches the longest cached prefix of each prompt, and the recurrent
state checkpoints (-rcs, system RAM) let hybrid models (Qwen3.x GatedDeltaNet layers) resume from it instead of
reading the whole conversation again.  The reused count goes back to Strata in RESUME and DONE.
"""
from __future__ import annotations

import argparse
import math
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

ENGINE_VERSION = "exl3-0.1.0"
ROOT = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------------------- process plumbing
def _runtime_dll_dirs():
    """The portable runtime's DLL folders.  runtime.py adds them for the main process; a multiprocessing child
    (ExLlamaV3's CPU MoE worker, started with the spawn method) runs this file again as __mp_main__ without
    runtime.py, and os.add_dll_directory handles are not inherited - so the child adds them itself."""
    if os.name != "nt":
        return
    site = ROOT / "python" / "Lib" / "site-packages"
    for p in (ROOT / "python", ROOT / "native", site / "torch" / "lib", site / "_rocm_sdk_core" / "bin",
              site / "_rocm_sdk_libraries" / "bin", site / "_rocm_sdk_core" / "lib" / "rocm_sysdeps" / "bin"):
        if p.is_dir():
            try:
                _DLL_HANDLES.append(os.add_dll_directory(str(p)))
            except OSError:
                pass
    for p in (ROOT / "native", ROOT / "app"):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


_DLL_HANDLES: list = []
if __name__ == "__mp_main__":
    _runtime_dll_dirs()


class Protocol:
    """The protocol stream: a private duplicate of fd 1.  fd 1 itself (and sys.stdout) then point at stderr."""

    def __init__(self):
        sys.stdout.flush()
        fd = os.dup(1)
        os.dup2(2, 1)
        sys.stdout = sys.stderr
        self.out = os.fdopen(fd, "w", encoding="utf-8", newline="\n", buffering=1)
        self.lock = threading.Lock()

    def send(self, line: str):
        with self.lock:
            self.out.write(line + "\n")
            self.out.flush()


def log(msg: str):
    print(f"[strata-exl3] {msg}", file=sys.stderr, flush=True)


class Commands:
    """stdin on a thread, so a STOP (or QUIT) can arrive while a request runs."""

    def __init__(self):
        if hasattr(sys.stdin, "reconfigure"):
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for raw in sys.stdin:
            line = raw.strip("\r\n")
            if line:
                self.q.put(line)
        self.q.put(None)                      # EOF: the server is gone

    def get(self, timeout=None):
        return self.q.get(timeout=timeout)

    def poll(self):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return ""


# ----------------------------------------------------------------------------------------- arguments
STRATA_ONLY = {   # flags Strata's server may add for its own engine: (takes a value?)
    "--vision": False, "--vram-elastic": False, "--split-skip-if-fits": False, "--session-save-reclaim": False,
    "--batch": True, "--slots": True, "--batch-groups": True, "--batch-mtp": True, "--layer-split": True,
    "--vram-reserve-mib": True, "--vram-segment-mib": True, "--expert-profile": True, "--expert-profile-save": True,
    "--expert-profile-save-every": True, "--peer-device": True, "--native": True, "--pack": True,
}


def parse_args(argv):
    from exllamav3 import model_init
    ap = argparse.ArgumentParser(prog="strata_exl3_engine", allow_abbrev=False)
    model_init.add_args(ap, cache=True, add_draft_model_args=True, default_cache_size=65536,
                        default_recurrent_cache_size=4.0, default_chunk_size=2048)
    ap.add_argument("--serve", action="store_true", help="speak Strata's engine protocol on stdin/stdout")
    ap.add_argument("--max-context", type=int, default=None, help="Strata's name for the cache size (tokens)")
    ap.add_argument("--eos-ids", default=None, help="comma-separated stop token ids (default: the model's)")
    ap.add_argument("--tail-role-token", type=int, default=None,
                    help="accepted for Strata's effort_position=end; prefix caching keeps the shared part anyway")
    rest, out = [], []
    it = iter(argv)
    for a in it:
        if a in STRATA_ONLY:
            rest.append(a)
            if STRATA_ONLY[a]:
                rest.append(next(it, ""))
            continue
        out.append(a)
    args, unknown = ap.parse_known_args(out)
    if args.max_context:
        args.cache_size = args.max_context
    if args.cache_size % 256:
        ap.error("--max-context / -cs must be a multiple of 256")
    return args, rest + unknown


def parse_request(line: str):
    """`GEN <max_new> [k=v ...] <ids>` -> (max_new, keys, ids)."""
    f = line.split()
    if len(f) < 3:
        raise ValueError("GEN needs <max_new> and the prompt ids")
    max_new = int(f[1])
    keys = {}
    for kv in f[2:-1]:
        k, sep, v = kv.partition("=")
        if not sep:
            raise ValueError(f"malformed request key {kv!r}")
        keys[k] = v
    ids = [int(t) for t in f[-1].split(",") if t]
    if not ids:
        raise ValueError("empty prompt")
    if max_new < 1:
        raise ValueError("max_new must be at least 1")
    return max_new, keys, ids


def _num(keys, name, cast=float, default=None):
    v = keys.get(name)
    if v is None:
        return default
    x = cast(v)
    if isinstance(x, float) and not math.isfinite(x):
        raise ValueError(f"{name} must be finite")
    return x


def make_sampler(keys: dict, vocab_size: int):
    """Strata's request keys -> an ExLlamaV3 sampler.  No temperature (or 0) is greedy, as in Strata's engine."""
    from exllamav3 import ComboSampler
    bias = {}
    if keys.get("logit_bias"):
        for item in keys["logit_bias"].split(","):
            tid, _, b = item.partition(":")
            tid, b = int(tid), float(b)
            if not 0 <= tid < vocab_size:
                raise ValueError(f"logit_bias: token {tid} is outside the vocabulary")
            bias[tid] = b
    temp = _num(keys, "temperature", default=0.0)
    last_n = _num(keys, "penalty_last_n", int, 64)
    return ComboSampler(
        rep_p=_num(keys, "penalty_repeat", default=1.0),
        freq_p=_num(keys, "penalty_freq", default=0.0),
        pres_p=_num(keys, "penalty_present", default=0.0),
        rep_sustain_range=max(1, last_n), rep_decay_range=0,
        temperature=temp,
        min_p=_num(keys, "min_p", default=0.0),
        top_k=_num(keys, "top_k", int, 0),
        top_p=_num(keys, "top_p", default=1.0),
        temp_last=True,
        logit_bias=bias or None,
    )


# ----------------------------------------------------------------------------------------- the engine
class Engine:
    def __init__(self, args, proto: Protocol, cmds: Commands):
        self.args, self.proto, self.cmds = args, proto, cmds
        self.quit = False

    def load(self):
        import torch
        from exllamav3 import Generator, Job, model_init
        self.torch, self.Job = torch, Job
        with torch.inference_mode():
            model, self.config, self.cache, self.tokenizer, draft_model, _, draft_cache = model_init.init(self.args)
            self.generator = Generator(
                model=model, cache=self.cache, tokenizer=self.tokenizer,
                draft_model=draft_model, draft_cache=draft_cache, num_draft_tokens=self.args.num_draft_tokens,
                max_batch_size=1, max_chunk_size=self.args.chunk_size, cpu_cache_size=0,
                recurrent_cache_size=int(max(0.0, self.args.recurrent_cache_size) * 1024 ** 3))
        if self.args.eos_ids:
            stops = [int(x) for x in self.args.eos_ids.split(",") if x.strip()]
        else:
            stops = list(self.config.eos_token_id_list or [])
            for name in ("<|im_end|>", "<|endoftext|>"):
                tid = self.tokenizer.single_id(name) if hasattr(self.tokenizer, "single_id") else None
                if isinstance(tid, int) and tid >= 0:
                    stops.append(tid)
            if self.tokenizer.eos_token_id is not None:
                stops.append(self.tokenizer.eos_token_id)
        self.stop_ids = sorted(set(stops))
        self.vocab_size = int(getattr(self.config, "vocab_size", 0) or self.tokenizer.actual_vocab_size)
        self.max_context = int(self.generator.max_total_tokens)
        try:
            free = torch.cuda.mem_get_info()[0] // 2 ** 20
        except Exception:  # noqa: BLE001 - informational only
            free = -1
        k, v = self._kv_bits()
        self.proto.send(f"INFO engine={ENGINE_VERSION} backend=exllamav3 kv=k{k}v{v} logit_bias=1 "
                        f"vram_free_mib={free} recurrent_cache_mib={int(self.args.recurrent_cache_size * 1024)} "
                        f"chunk={self.args.chunk_size} draft={1 if draft_model else 0} "
                        f"stop_ids={','.join(map(str, self.stop_ids))}")
        self.proto.send(f"READY {self.max_context} stop")
        log(f"ready: context {self.max_context}, stop ids {self.stop_ids}, draft {'on' if draft_model else 'off'}")

    def _kv_bits(self):
        cq = getattr(self.args, "cache_quant", None)
        if not cq:
            return 16, 16
        p = [int(x) for x in str(cq).split(",")]
        return (p[0], p[0]) if len(p) == 1 else (p[0], p[1])

    # ------------------------------------------------------------------ one request
    def generate(self, line: str):
        torch = self.torch
        max_new, keys, ids = parse_request(line)
        if len(ids) + max_new > self.max_context:
            raise ValueError(f"prompt ({len(ids)}) + max_new ({max_new}) exceeds the context ({self.max_context})")
        bad = next((t for t in ids if not 0 <= t < self.vocab_size), None)
        if bad is not None:
            raise ValueError(f"token id {bad} is outside the vocabulary ({self.vocab_size})")
        sampler = make_sampler(keys, self.vocab_size)
        seed = _num(keys, "seed", int, None)
        job = self.Job(input_ids=torch.tensor([ids], dtype=torch.long), max_new_tokens=max_new, sampler=sampler,
                       seed=seed, stop_conditions=list(self.stop_ids), decode_special_tokens=True)
        t0 = time.perf_counter()
        t_first = None
        generated, reused, sent_resume = 0, 0, False
        finish, terminal = "length", {}
        with torch.inference_mode():
            self.generator.enqueue(job)
            try:
                while self.generator.num_remaining_jobs():
                    cmd = self.cmds.poll()
                    if cmd is None or cmd == "QUIT":
                        self.quit = True
                        finish = "cancel"
                        break
                    if cmd == "STOP":
                        finish = "cancel"
                        break
                    if cmd:
                        log(f"ignored during a request: {cmd[:40]}")
                    for r in self.generator.iterate():
                        if not sent_resume:
                            reused = self._cached(job)
                            if reused:
                                self.proto.send(f"RESUME {reused}")
                            sent_resume = True
                        if r.get("stage") == "prefill":
                            ms = (time.perf_counter() - t0) * 1000.0
                            read, total = int(r.get("curr_progress", 0)), int(r.get("max_progress", len(ids)))
                            rate = (read - reused) / (ms / 1000.0) if ms > 0 and read > reused else 0.0
                            self.proto.send(f"PP {read} {total} {ms:.1f} {rate:.1f}")
                            continue
                        tids = r.get("token_ids")
                        if tids is not None:
                            for t in tids.flatten().tolist():
                                if t_first is None:
                                    t_first = time.perf_counter()
                                generated += 1
                                self.proto.send(f"T {int(t)}")
                        if r.get("eos"):
                            terminal = r
                            reason = r.get("eos_reason")
                            if reason == "stop_token":
                                stop = r.get("eos_triggering_token_id")
                                if isinstance(stop, int):
                                    if t_first is None:
                                        t_first = time.perf_counter()
                                    generated += 1
                                    self.proto.send(f"T {stop}")
                                finish = "stop"
                            elif reason in ("stop_string", "end_filter", "loop_detected"):
                                finish = "stop"
                            else:
                                finish = "length"
            finally:
                if self.generator.num_remaining_jobs():
                    self.generator.cancel(job)
        t_end = time.perf_counter()
        if terminal.get("cached_tokens") is not None:
            reused = int(terminal["cached_tokens"])
        prompt_ms = (terminal.get("time_prefill") or ((t_first or t_end) - t0)) * 1000.0
        decode_ms = (terminal.get("time_generate") or (t_end - (t_first or t_end))) * 1000.0
        acc = int(terminal.get("accepted_draft_tokens") or 0)
        rej = int(terminal.get("rejected_draft_tokens") or 0)
        self._done(generated, len(ids), prompt_ms, decode_ms, finish, acc, acc + rej, reused)

    def _cached(self, job) -> int:
        try:
            from exllamav3.constants import PAGE_SIZE
        except ImportError:
            PAGE_SIZE = 256
        pages = int(getattr(job, "cached_pages", 0) or 0)
        extra = int(getattr(job, "cached_tokens", 0) or 0)
        return pages * PAGE_SIZE + extra

    def _done(self, generated, prompt, prompt_ms, decode_ms, finish, acc, offered, reused):
        # generated prompt_tokens prompt_ms decode_ms finish drafts_accepted drafts_offered reused
        # hits lookups ram_blobs file_blobs file_mb prompt_read offloaded
        read = max(0, prompt - reused)
        self.proto.send(f"DONE {generated} {prompt} {prompt_ms:.1f} {decode_ms:.1f} {finish} {acc} {offered} {reused} "
                        f"0 0 0 0 0.0 {read} 0")

    # ------------------------------------------------------------------ the loop
    def serve(self):
        while not self.quit:
            line = self.cmds.get()
            if line is None or line == "QUIT":
                break
            head = line.split(" ", 1)[0]
            try:
                if head == "GEN":
                    self.generate(line)
                elif head == "GENI":
                    self.proto.send("ERR images are not supported by the EXL3 engine yet (start without vision)")
                elif head == "STOP":
                    pass                                   # nothing running: a late STOP after DONE
                elif head == "VRAM":
                    self.proto.send("ERR the EXL3 engine has no elastic expert cache (VRAM is fixed at load)")
                elif head in ("SAVE", "RESTORE"):
                    self.proto.send("SERR invalid 0 disk sessions are not supported by the EXL3 engine")
                else:
                    self.proto.send(f"ERR unknown command {head[:20]}")
            except (ValueError, TypeError) as e:
                self.proto.send(f"ERR {str(e).replace(chr(10), ' ')[:400]}")
            except Exception as e:  # noqa: BLE001 - keep serving; the server turns ERR into an error reply
                traceback.print_exc()
                msg = f"{type(e).__name__}: {e}".replace("\n", " ")[:400]
                if "out of memory" in msg.lower():
                    msg = "GPU out of memory: " + msg
                self.proto.send(f"ERR {msg}")
        log("quit")


def main():
    proto = Protocol()
    args, ignored = parse_args(sys.argv[1:])
    if ignored:
        log("ignored (not used by the EXL3 engine): " + " ".join(ignored))
    if not args.serve:
        raise SystemExit("strata_exl3_engine: start it with --serve (Strata's server does)")
    cmds = Commands()
    eng = Engine(args, proto, cmds)
    try:
        eng.load()
    except Exception:  # noqa: BLE001 - before READY: the log says why, the server reports the exit
        traceback.print_exc()
        raise SystemExit(3)
    eng.serve()


if __name__ == "__main__":
    main()
