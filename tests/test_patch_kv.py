#!/usr/bin/env python3
"""patch-kv.py on synthetic caches laid out as huggingface_hub lays them out (the snapshot's
files linked into blobs/): the lane's fp8 KV declaration written into both configs of a
ModelOpt export that declares none, the blobs left as they were, idempotence, the refusals
(another KV scheme, a quantization that is not ModelOpt's), --check, the exact snapshot, and
the order with patch-yarn.py that the switch and the installer keep: a YaRN restore puts back
a config.json saved before the declaration, so the declaration comes after it. main() runs in
this process, for the coverage gate, and as the CLI the scripts call. No network."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "patch-kv.py"
YARN = REPO / "patch-yarn.py"
SHA = "c" * 40
# the lane's declaration, as RadixArk/Qwen3.8-27B-NVFP4 @ 52d1adc writes it in its two files
LANE_SCHEME = {"dynamic": False, "num_bits": 8, "type": "float"}
LANE_ALGO = "FP8"


def load():
    spec = importlib.util.spec_from_file_location("patch_kv_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def nvfp4(kv_scheme=None, kv_algo=None, quant_method="modelopt"):
    """The two configs of an NVFP4 ModelOpt export of the 27B, cut to the keys that matter here."""
    qc = {"config_groups": {"group_0": {"weights": {"num_bits": 4}}}, "quantized_layers": {"x": {"quant_algo": "NVFP4"}},
          "ignore": ["mtp*"], "quant_algo": "MIXED_PRECISION", "producer": {"name": "modelopt"},
          "quant_method": quant_method}
    if kv_scheme is not None:
        qc["kv_cache_scheme"] = kv_scheme
    cfg = {"architectures": ["Qwen3_5ForConditionalGeneration"],
           "text_config": {"max_position_embeddings": 262144,
                           "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"}},
           "quantization_config": qc}
    hfq = {"producer": {"name": "modelopt"},
           "quantization": {"quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": kv_algo,
                            "quantized_layers": {"x": {"quant_algo": "NVFP4"}}, "exclude_modules": ["mtp*"]}}
    return cfg, hfq


class Cache:
    """A throwaway HF cache holding one snapshot whose files are links into blobs/."""

    def __init__(self, test, repo="atakans/Arete-27B-NVFP4", files=None):
        self.base = Path(tempfile.mkdtemp(prefix="kv-"))
        test.addCleanup(shutil.rmtree, self.base, True)
        self.repo = repo
        self.repo_dir = self.base / "hub" / f"models--{repo.replace('/', '--')}"
        self.snap = self.repo_dir / "snapshots" / SHA
        self.snap.mkdir(parents=True)
        (self.repo_dir / "blobs").mkdir()
        for name, data in (files or {}).items():
            raw = json.dumps(data, indent=2).encode()
            blob = self.repo_dir / "blobs" / hashlib.sha1(raw).hexdigest()
            blob.write_bytes(raw)
            (self.snap / name).symlink_to(os.path.relpath(blob, self.snap))

    def json(self, name):
        return json.loads((self.snap / name).read_text())

    def blobs(self):
        return {p.name: p.read_bytes() for p in (self.repo_dir / "blobs").iterdir()}


def cli(*args, script=SCRIPT):
    return subprocess.run([sys.executable, "-I", str(script), *map(str, args)], capture_output=True, text=True)


def inproc(*args):
    """main() in this process: (exit code, stdout, stderr)."""
    mod = load()
    out, err, old = io.StringIO(), io.StringIO(), sys.argv
    sys.argv = ["patch-kv.py", *map(str, args)]
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            mod.main()
        code = 0
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        err.write("" if isinstance(e.code, int) or e.code is None else str(e.code))
    finally:
        sys.argv = old
    return code, out.getvalue(), err.getvalue()


class TheDeclaration(unittest.TestCase):
    def test_an_export_that_declares_none_gets_the_lanes_declaration_in_both_files(self):
        cfg, hfq = nvfp4()
        c = Cache(self, files={"config.json": cfg, "hf_quant_config.json": hfq})
        blobs = c.blobs()
        code, out, err = inproc(c.base, c.repo, SHA)
        self.assertEqual(code, 0, out + err)
        self.assertIn("fp8 KV cache declared", out)
        got, gotq = c.json("config.json"), c.json("hf_quant_config.json")
        self.assertEqual(got["quantization_config"]["kv_cache_scheme"], LANE_SCHEME)
        self.assertEqual(gotq["quantization"]["kv_cache_quant_algo"], LANE_ALGO)
        # SGLang's ModelOpt rule (from_config): a float scheme of 8 bits is an FP8 cache
        scheme = got["quantization_config"]["kv_cache_scheme"]
        self.assertEqual((scheme["type"], scheme["num_bits"]), ("float", 8))
        # nothing else moved: the declaration is the one difference in each file
        del got["quantization_config"]["kv_cache_scheme"]
        self.assertEqual(got, cfg)
        gotq["quantization"]["kv_cache_quant_algo"] = None
        self.assertEqual(gotq, hfq)
        # the links were replaced, the blobs are the checkpoint's own bytes still
        self.assertFalse((c.snap / "config.json").is_symlink())
        self.assertFalse((c.snap / "hf_quant_config.json").is_symlink())
        self.assertEqual(c.blobs(), blobs)
        self.assertEqual(sorted(p.name for p in c.snap.iterdir()), ["config.json", "hf_quant_config.json"])

    def test_a_second_run_writes_nothing(self):
        cfg, hfq = nvfp4()
        c = Cache(self, files={"config.json": cfg, "hf_quant_config.json": hfq})
        self.assertEqual(inproc(c.base, c.repo, SHA)[0], 0)
        before = {n: (c.snap / n).stat().st_ino for n in ("config.json", "hf_quant_config.json")}
        code, out, _ = inproc(c.base, c.repo, SHA)
        self.assertEqual(code, 0)
        self.assertIn("already declared", out)
        self.assertEqual({n: (c.snap / n).stat().st_ino for n in before}, before)

    def test_a_declaration_the_export_carries_already_is_left_as_it_is(self):
        # the lane's own NVFP4: its links stay links, as no file lacks the declaration
        cfg, hfq = nvfp4(LANE_SCHEME, LANE_ALGO)
        c = Cache(self, repo="RadixArk/Qwen3.8-27B-NVFP4", files={"config.json": cfg, "hf_quant_config.json": hfq})
        code, out, _ = inproc(c.base, c.repo, SHA)
        self.assertEqual(code, 0)
        self.assertIn("already declared", out)
        self.assertTrue((c.snap / "config.json").is_symlink())
        self.assertTrue((c.snap / "hf_quant_config.json").is_symlink())

    def test_one_file_lacking_it_is_the_one_written(self):
        cfg, hfq = nvfp4(LANE_SCHEME, None)
        c = Cache(self, files={"config.json": cfg, "hf_quant_config.json": hfq})
        self.assertEqual(inproc(c.base, c.repo, SHA)[0], 0)
        self.assertTrue((c.snap / "config.json").is_symlink())
        self.assertEqual(c.json("hf_quant_config.json")["quantization"]["kv_cache_quant_algo"], LANE_ALGO)

    def test_without_an_hf_quant_config_the_config_alone_carries_it(self):
        cfg, _ = nvfp4()
        c = Cache(self, files={"config.json": cfg})
        self.assertEqual(inproc(c.base, c.repo, SHA)[0], 0)
        self.assertEqual(c.json("config.json")["quantization_config"]["kv_cache_scheme"], LANE_SCHEME)
        self.assertFalse((c.snap / "hf_quant_config.json").exists())


class TheRefusals(unittest.TestCase):
    def assert_refused(self, files, says):
        c = Cache(self, files=files)
        blobs = c.blobs()
        code, out, err = inproc(c.base, c.repo, SHA)
        self.assertEqual(code, 3, out + err)
        self.assertIn(says, err)
        self.assertTrue(all((c.snap / n).is_symlink() for n in files))
        self.assertEqual(c.blobs(), blobs)

    def test_another_kv_scheme_is_not_overwritten(self):
        cfg, hfq = nvfp4({"dynamic": False, "num_bits": 4, "type": "float"}, "NVFP4")
        self.assert_refused({"config.json": cfg, "hf_quant_config.json": hfq}, "another KV cache")

    def test_another_kv_algo_in_hf_quant_config_is_not_overwritten(self):
        cfg, hfq = nvfp4(None, "NVFP4")
        self.assert_refused({"config.json": cfg, "hf_quant_config.json": hfq}, "another KV cache")

    def test_a_quantization_that_is_not_modelopts_is_refused(self):
        # Qwen's FP8 release: its fp8 cache is the unit's flag, never a declaration of this form
        cfg, _ = nvfp4(quant_method="fp8")
        self.assert_refused({"config.json": cfg}, "no ModelOpt quantization")

    def test_an_unquantized_checkpoint_is_refused(self):
        cfg, _ = nvfp4()
        del cfg["quantization_config"]
        self.assert_refused({"config.json": cfg}, "no ModelOpt quantization")

    def test_an_hf_quant_config_without_its_section_is_refused(self):
        cfg, _ = nvfp4()
        self.assert_refused({"config.json": cfg, "hf_quant_config.json": {"producer": {"name": "modelopt"}}},
                            "no quantization section")


class TheCommandLine(unittest.TestCase):
    def test_check_says_absent_then_declared_and_writes_nothing(self):
        cfg, hfq = nvfp4()
        c = Cache(self, files={"config.json": cfg, "hf_quant_config.json": hfq})
        code, out, err = inproc("--check", c.base, c.repo, SHA)
        self.assertEqual(code, 2, out + err)
        self.assertIn("bf16", out)
        self.assertTrue((c.snap / "config.json").is_symlink())
        self.assertEqual(cli(c.base, c.repo, SHA).returncode, 0)
        r = cli("--check", c.base, c.repo, SHA)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_a_ref_name_reads_the_snapshot_it_points_at(self):
        cfg, hfq = nvfp4()
        c = Cache(self, files={"config.json": cfg, "hf_quant_config.json": hfq})
        (c.repo_dir / "refs").mkdir()
        (c.repo_dir / "refs" / "main").write_text(SHA + "\n")
        code, out, err = inproc(c.base, c.repo, "main")
        self.assertEqual(code, 0, out + err)
        self.assertEqual(c.json("config.json")["quantization_config"]["kv_cache_scheme"], LANE_SCHEME)

    def test_a_revision_not_in_the_cache_writes_nothing_anywhere(self):
        # no fallback to another snapshot: the engine reads the one its --revision names
        cfg, hfq = nvfp4()
        c = Cache(self, files={"config.json": cfg, "hf_quant_config.json": hfq})
        code, out, err = inproc(c.base, c.repo, "d" * 40)
        self.assertEqual(code, 1, out + err)
        self.assertIn("fetch the checkpoint first", err)
        self.assertTrue((c.snap / "config.json").is_symlink())

    def test_wrong_argument_counts_exit_with_the_usage(self):
        for argv in ([], ["--check"], ["a", "b"], ["a", "b", "c", "d"], ["--check", "a", "b"]):
            code, _, err = inproc(*argv)
            self.assertEqual(code, 1, argv)
            self.assertIn("Usage: patch-kv.py", err, argv)


class AfterPatchYarn(unittest.TestCase):
    """The order the switch and the installer keep: the window's configs first, then this."""

    def test_the_1m_config_carries_both_and_a_restore_needs_the_declaration_again(self):
        cfg, hfq = nvfp4()
        c = Cache(self, files={"config.json": cfg, "hf_quant_config.json": hfq})
        self.assertEqual(cli(c.base, c.repo, SHA, script=YARN).returncode, 0)
        self.assertEqual(cli(c.base, c.repo, SHA).returncode, 0)
        got = c.json("config.json")
        self.assertEqual(got["text_config"]["max_position_embeddings"], 1010000)
        self.assertEqual(got["quantization_config"]["kv_cache_scheme"], LANE_SCHEME)
        # patch-yarn's check still reads a YaRN patch with its backup
        self.assertEqual(cli("--check", c.base, c.repo, SHA, script=YARN).returncode, 2)
        # back to the native window: the backup predates the declaration
        self.assertEqual(cli("--restore", c.base, c.repo, SHA, script=YARN).returncode, 0)
        self.assertEqual(cli("--check", c.base, c.repo, SHA).returncode, 2)
        self.assertEqual(cli(c.base, c.repo, SHA).returncode, 0)
        got = c.json("config.json")
        self.assertEqual(got["text_config"]["max_position_embeddings"], 262144)
        self.assertEqual(got["quantization_config"]["kv_cache_scheme"], LANE_SCHEME)
        self.assertEqual(c.json("hf_quant_config.json")["quantization"]["kv_cache_quant_algo"], LANE_ALGO)


if __name__ == "__main__":
    unittest.main()
