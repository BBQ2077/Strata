"""Offline checks for the EXL3 port (no GPU, no model load): the engine's request parsing, the tokenizer export
against the model's own tokenizer.json, and the server's launcher / top_k / stop-id changes.

python exl3/test_exl3_port.py [--model <EXL3 folder>]
"""
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = os.environ.get("STRATA_EXL3_MODEL", "")
if "--model" in sys.argv:
    _i = sys.argv.index("--model")
    MODEL = sys.argv[_i + 1]
    del sys.argv[_i:_i + 2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def have(mod):
    return importlib.util.find_spec(mod) is not None


eng = load("strata_exl3_engine", HERE / "strata_exl3_engine.py")
hft = load("hf_tokenizer", HERE / "hf_tokenizer.py")


class Request(unittest.TestCase):
    def test_gen_line(self):
        n, keys, ids = eng.parse_request("GEN 128 temperature=0.7 top_k=20 logit_bias=5:-1.5 1,2,3")
        self.assertEqual((n, ids), (128, [1, 2, 3]))
        self.assertEqual(keys, {"temperature": "0.7", "top_k": "20", "logit_bias": "5:-1.5"})

    def test_greedy_line(self):
        self.assertEqual(eng.parse_request("GEN 1 42"), (1, {}, [42]))

    def test_bad_lines(self):
        for line in ("GEN 5", "GEN 0 1,2", "GEN 5 notakey 1,2", "GEN x 1"):
            with self.assertRaises(ValueError, msg=line):
                eng.parse_request(line)

    def test_number_keys(self):
        self.assertEqual(eng._num({"top_k": "7"}, "top_k", int), 7)
        self.assertIsNone(eng._num({}, "seed", int))
        with self.assertRaises(ValueError):
            eng._num({"temperature": "nan"}, "temperature")

    def test_strata_flags_known(self):
        for f in ("--slots", "--expert-profile", "--vision", "--layer-split"):
            self.assertIn(f, eng.STRATA_ONLY)


@unittest.skipUnless(MODEL and pathlib.Path(MODEL, "tokenizer.json").is_file(), "model folder not found (STRATA_EXL3_MODEL or --model)")
class Tokenizer(unittest.TestCase):
    def test_read(self):
        h = hft.read_hf(MODEL)
        self.assertEqual(h["pre_tokenizer"], [hft.HF_QWEN35_PATTERN])
        tok = h["tokens"]
        self.assertEqual(tok.index("<|im_end|>"), 248046)
        self.assertEqual(h["types"][248046], hft.CONTROL)
        self.assertEqual(h["types"][tok.index("<think>")], hft.USER_DEFINED)
        self.assertTrue(h["chat_template"])

    @unittest.skipUnless(have("regex"), "regex is not installed")
    def test_export_matches_hf(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = hft.export(MODEL, d)
            self.assertEqual(cfg["special_ids"]["<|im_end|>"], 248046)
            bad = hft.check(MODEL, d)
            self.assertFalse([b for b in bad if b.startswith("mismatch")], bad)


@unittest.skipUnless(have("jinja2") and have("regex"), "jinja2 / regex are not installed")
class Server(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT))
        cls.S = load("strata_server", ROOT / "serve" / "server.py")

    def test_launcher(self):
        S = self.S
        self.assertEqual(S.engine_launcher({}), [])
        exe = sys.executable
        out = S.engine_launcher({"launcher": [exe, "-u", "runtime.py", "script"], "cwd": str(HERE)})
        self.assertEqual(out[0], exe)
        self.assertEqual(out[1:], ["-u", "runtime.py", "script"])
        with self.assertRaises(ValueError):
            S.engine_launcher({"launcher": "python.exe"})

    def test_top_k_cap(self):
        S = self.S
        old = S.ENGINE_TOP_K_CAP
        try:
            S.ENGINE_TOP_K_CAP = 64
            self.assertIn("top_k=64", S.StrataEngine.sampling_keys({"top_k": 200}))
            S.ENGINE_TOP_K_CAP = 0
            self.assertIn("top_k=200", S.StrataEngine.sampling_keys({"top_k": 200}))
            self.assertIn("top_k=0", S.StrataEngine.sampling_keys({"top_k": 0}))
        finally:
            S.ENGINE_TOP_K_CAP = old

    def test_effort_end_finds_flag(self):
        self.assertIn(b"--tail-role-token", (HERE / "strata_exl3_engine.py").read_bytes())

    def test_env_paths(self):
        S = self.S
        cfg = json.loads((HERE / "strata-exl3.json").read_text(encoding="utf-8"))
        env = {"STRATA_EXL3_RUNTIME": "R", "STRATA_EXL3_MODEL": "M", "STRATA_EXL3_TOKENIZER": "T"}
        old = {k: os.environ.get(k) for k in env}
        try:
            os.environ.update(env)
            out = S.expand_env_paths(cfg)
            self.assertEqual(out["cwd"], "R")
            self.assertEqual(out["exe"], "R/strata_exl3_engine.py")
            self.assertEqual(out["tokenizer"], "T")
            self.assertEqual(out["args"][out["args"].index("-m") + 1], "M")
            self.assertEqual(cfg["cwd"], "${STRATA_EXL3_RUNTIME}")
            del os.environ["STRATA_EXL3_MODEL"]
            with self.assertRaisesRegex(ValueError, "STRATA_EXL3_MODEL"):
                S.expand_env_paths(cfg)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_no_machine_paths(self):
        for p in HERE.iterdir():
            if p.suffix in (".py", ".json", ".bat", ".md"):
                self.assertNotRegex(p.read_text(encoding="utf-8"), r"(?<![A-Za-z])[A-Za-z]:[\\/](?!path[\\/])", p.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
