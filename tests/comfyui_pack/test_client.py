"""The client against the stub server: the handshake and the version
messages, uploads by hash and the 409 path, runs, interrupt, encode, the
VAE pair and the API key. Needs torch and numpy, not ComfyUI."""

from __future__ import annotations

import http.client
import http.server
import json
import os
import tempfile
import threading
import time
import unittest

import numpy as np
import torch

from support import Stub, load_client

C = load_client()


class Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stub = Stub()
        cls.c = C.StepClient(cls.stub.url)
        cls.caps = cls.c.handshake()

    @classmethod
    def tearDownClass(cls):
        cls.stub.close()

    def setUp(self):
        self.stub.reset()

    def test_handshake_reads_the_contract(self):
        self.assertEqual(self.caps["contract"], {"major": 1, "minor": 0})
        self.assertIn("lora", self.caps["patch_kinds"])

    def test_every_versioned_request_names_the_contract(self):
        s, _, d = self.c.request("GET", C.PREFIX + "/models", versioned=False)
        self.assertEqual(s, 400)
        self.assertIn(b"X-Gmlx-Steps-Contract", d)
        self.assertEqual(len(self.c.models()), 4)

    def test_status_and_manifest(self):
        st = self.c.status("z-image-turbo")
        self.assertEqual(st["comfy"]["model_class"], "ZImage")
        m = self.c.manifest("qwen-image-2.1")
        self.assertEqual(m["count"], len(m["tensors"]))
        with self.assertRaises(C.GmlxError) as e:
            self.c.status("no-such-model")
        self.assertIn("404", str(e.exception))

    def test_tensor_upload_is_once_per_hash(self):
        t = torch.randn(1, 5, 7)
        before = self.c.stats["tensor_uploads"]
        h1 = self.c.ensure_tensor(t)
        h2 = self.c.ensure_tensor(t.clone())
        self.assertEqual(h1, h2)
        self.assertEqual(self.c.stats["tensor_uploads"] - before, 1)
        self.assertEqual(len(self.stub.events("put_tensor")), 1)

    def test_a_wrong_hash_is_refused(self):
        with self.assertRaises(C.GmlxError) as e:
            self.c.put_tensor("0" * 64, torch.zeros(2))
        self.assertIn("hash", str(e.exception))

    def test_adapter_upload_probes_first_and_memoizes_by_identity(self):
        a = torch.randn(4, 16, dtype=torch.bfloat16)
        h = self.c.ensure_adapter(a)
        self.c.known_adapters.clear()
        self.assertEqual(self.c.ensure_adapter(a), h)
        self.assertEqual(len(self.stub.events("put_adapter")), 1)
        self.assertEqual(self.c.stats["adapter_uploads"], 1)

    def test_an_inference_tensor_hashes_and_memoizes(self):
        with torch.inference_mode():
            a = torch.randn(4, 8)
        h = self.c.ensure_adapter(a)
        self.assertEqual(self.c._memo_hash(a), h)
        self.assertEqual(h, C.tensor_hash(a))

    def test_table_missing_adapters_answer_409(self):
        table = {"model": "z-image-turbo", "entries": [{"key": "diffusion_model.x.weight", "band": None,
                 "patches": [{"up": "a" * 64, "down": "b" * 64, "alpha": 1.0, "strength": 1.0,
                              "strength_model": 1.0}]}]}
        with self.assertRaises(C.GmlxMissing) as e:
            self.c.define_table("z-image-turbo", table)
        self.assertEqual(sorted(e.exception.missing), ["a" * 64, "b" * 64])

    def test_encode_marks_its_conditioning_held(self):
        out, fields = self.c.encode("z-image-turbo", "a red bicycle")
        self.assertEqual(tuple(out["cond"].shape), (1, 32, 2560))
        self.assertIn(fields["cond"], self.c.known_tensors)
        self.assertEqual(C.tensor_hash(out["cond"]), fields["cond"])

    def test_encode_with_images_sends_a_container(self):
        ims = [np.full((64, 96, 3), 100, np.uint8), np.full((32, 32, 3), 50, np.uint8)]
        out, fields = self.c.encode("qwen-image-2.1", "make it blue", ims)
        self.assertEqual(fields["ref_latents"], [])
        self.assertEqual(fields["image_slots"], [2, 5])
        ev = self.stub.events("encode")[0]
        self.assertEqual((ev["images"], ev["refs"]), (2, False))

    def test_vae_pair(self):
        img = self.c.vae_decode("qwen-image-2.1", torch.zeros(1, 64, 4, 6), alpha=True)
        self.assertEqual(tuple(img.shape), (1, 64, 96, 4))
        lat = self.c.vae_encode("z-image-turbo", torch.rand(2, 64, 32, 3))
        self.assertEqual(tuple(lat.shape), (2, 16, 8, 4))

    def test_run_begin_end_and_step_with_the_run(self):
        run = self.c.run_begin("z-image-turbo")
        out, fields = self.c.encode("z-image-turbo", "x")
        o, h = self.c.step(torch.zeros(1, 16, 4, 4), torch.tensor([0.5]),
                           {"model": "z-image-turbo", "run": run, "convention": "comfy",
                            "patch_table": None, "rows": [{"cond": fields["cond"], "num_tokens": 32}]})
        self.assertEqual(tuple(o.shape), (1, 16, 4, 4))
        self.assertIn("x-gmlx-forward-ms", h)
        self.c.run_end(run)
        kinds = [e["kind"] for e in self.stub.log() if e["kind"] in ("run_begin", "step", "run_end")]
        self.assertEqual(kinds, ["run_begin", "step", "run_end"])
        with self.assertRaises(C.GmlxError):
            self.c.step(torch.zeros(1, 16, 4, 4), torch.tensor([0.5]),
                        {"model": "z-image-turbo", "run": run, "convention": "comfy",
                         "patch_table": None, "rows": [{"cond": fields["cond"]}]})

    def test_a_step_naming_a_dropped_tensor_answers_409(self):
        _, fields = self.c.encode("z-image-turbo", "y")
        self.stub.forget()
        with self.assertRaises(C.GmlxMissing) as e:
            self.c.step(torch.zeros(1, 16, 4, 4), torch.tensor([0.5]),
                        {"model": "z-image-turbo", "run": None, "convention": "comfy",
                         "patch_table": "f" * 64, "rows": [{"cond": fields["cond"]}]})
        self.assertEqual(sorted(e.exception.missing), sorted([fields["cond"], "f" * 64]))


class Interrupt(unittest.TestCase):
    def test_interrupt_and_run_end_stop_an_in_flight_step(self):
        stub = Stub("--step-delay-ms", "3000")
        try:
            c = C.StepClient(stub.url)
            _, fields = c.encode("z-image-turbo", "x")
            for stopper in ("interrupt", "run_end"):
                run = c.run_begin("z-image-turbo")
                got = {}

                def go():
                    try:
                        c.step(torch.zeros(1, 16, 4, 4), torch.tensor([0.5]),
                               {"model": "z-image-turbo", "run": run, "convention": "comfy",
                                "patch_table": None, "rows": [{"cond": fields["cond"]}]})
                    except C.GmlxError as e:
                        got["e"] = e

                th = threading.Thread(target=go)
                t0 = time.monotonic()
                th.start()
                time.sleep(0.2)
                c.interrupt(run) if stopper == "interrupt" else c.run_end(run)
                th.join(5)
                self.assertIsInstance(got.get("e"), C.GmlxInterrupted, stopper)
                self.assertLess(time.monotonic() - t0, 2.0)
                if stopper == "interrupt":
                    c.run_end(run)
            by = [e["by"] for e in stub.events("step_interrupted")]
            self.assertEqual(by, ["interrupt", "run_end"])
        finally:
            stub.close()


class Timeouts(unittest.TestCase):
    def test_a_slow_call_after_a_health_probe_waits_for_its_answer(self):
        stub = Stub("--step-delay-ms", "600")
        try:
            c = C.StepClient(stub.url)
            self.assertTrue(c.health(timeout=0.2))
            _, fields = c.encode("z-image-turbo", "x")
            out, _ = c.step(torch.zeros(1, 16, 4, 4), torch.tensor([0.5]),
                            {"model": "z-image-turbo", "run": None, "convention": "comfy",
                             "patch_table": None, "rows": [{"cond": fields["cond"]}]})
            self.assertEqual(tuple(out.shape), (1, 16, 4, 4))
            self.assertEqual(c.stats["connections"], 1)
        finally:
            stub.close()


class _Idle(http.server.BaseHTTPRequestHandler):
    """Keep-alive answers that close the connection once idle, with no
    Connection header, and a slow path."""
    protocol_version = "HTTP/1.1"
    hits: dict = {}

    def log_message(self, *a):
        pass

    def do_GET(self):
        type(self).hits[self.path] = type(self).hits.get(self.path, 0) + 1
        if self.path.endswith("/slow"):
            time.sleep(0.5)
        self._answer()

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        type(self).hits[self.path] = type(self).hits.get(self.path, 0) + 1
        if self.path.endswith("/hangup"):
            # The whole request was read; close without an answer.
            self.close_connection = True
            return
        self._answer()

    def _answer(self):
        body = b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        if self.path.endswith("/drop"):
            self.close_connection = True


class KeepAlive(unittest.TestCase):
    def setUp(self):
        _Idle.hits = {}
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Idle)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.c = C.StepClient(f"http://127.0.0.1:{self.srv.server_address[1]}")

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    def test_a_pooled_connection_the_server_closed_is_retried_once(self):
        self.assertEqual(self.c.request("GET", "/drop", versioned=False)[0], 200)
        time.sleep(0.1)                 # the server has closed it
        self.assertEqual(self.c.request("GET", "/ok", versioned=False)[0], 200)
        self.assertEqual(self.c.stats["connections"], 2)
        self.assertEqual(_Idle.hits.get("/ok"), 1)

    def test_a_pooled_post_the_server_closed_while_idle_goes_out_once(self):
        self.assertEqual(self.c.request("GET", "/drop", versioned=False)[0], 200)
        time.sleep(0.1)                 # the server has closed it
        self.assertEqual(self.c.request("POST", "/ok", b"x" * 4096, versioned=False)[0], 200)
        self.assertEqual(self.c.stats["connections"], 2)
        self.assertEqual(_Idle.hits.get("/ok"), 1)

    def test_a_post_the_server_read_and_dropped_is_never_sent_again(self):
        for pooled in (False, True):
            _Idle.hits = {}
            c = C.StepClient(self.c.base_url)
            if pooled:
                self.assertEqual(c.request("GET", "/ok", versioned=False)[0], 200)
            with self.assertRaises(C.GmlxError):
                c.request("POST", "/hangup", b"x" * 4096, versioned=False)
            self.assertEqual(_Idle.hits.get("/hangup"), 1, f"pooled {pooled}")
            self.assertEqual(c.stats["connections"], 1, f"pooled {pooled}")

    def test_the_retry_after_a_pooled_send_failure_takes_a_new_connection(self):
        def broken():
            conn = http.client.HTTPConnection(self.c.host, self.c.port, timeout=5)
            conn.connect()

            def send(data):
                raise BrokenPipeError(32, "Broken pipe")
            conn.send = send
            return conn

        self.c._pool[:] = [broken(), broken()]
        self.assertEqual(self.c.request("POST", "/ok", b"{}", versioned=False)[0], 200)
        self.assertEqual(self.c.stats["connections"], 1)
        self.assertEqual(_Idle.hits.get("/ok"), 1)

    def test_a_call_that_timed_out_is_never_sent_again(self):
        self.assertEqual(self.c.request("GET", "/ok", versioned=False)[0], 200)
        with self.assertRaises(C.GmlxError):
            self.c.request("GET", "/slow", versioned=False, timeout=0.2)
        time.sleep(0.8)
        self.assertEqual(_Idle.hits.get("/slow"), 1)


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Versions(unittest.TestCase):
    def setUp(self):
        self.env = os.environ.get("GMLX_COMFY_SETTINGS")
        self.dir = tempfile.mkdtemp()
        os.environ["GMLX_COMFY_SETTINGS"] = os.path.join(self.dir, "comfyui.json")

    def tearDown(self):
        if self.env is None:
            os.environ.pop("GMLX_COMFY_SETTINGS", None)
        else:
            os.environ["GMLX_COMFY_SETTINGS"] = self.env

    def test_a_newer_server_says_to_restart_comfyui(self):
        stub = Stub("--contract", "2.0", "--gmlx-version", "99.0.0")
        try:
            c = C.StepClient(stub.url)
            with self.assertRaises(C.GmlxVersionError) as e:
                c.handshake()
            self.assertIn("Restart ComfyUI", str(e.exception))
            self.assertIn("gmlx 99.0.0", str(e.exception))
            self.assertIn(f"gmlx {C.PACK_VERSION}", str(e.exception))
            with self.assertRaises(C.GmlxVersionError) as e:
                c.models()
            self.assertIn("Upgrade the client", str(e.exception))
        finally:
            stub.close()

    def test_an_older_server_names_its_restart_and_the_upgrade_for_the_install(self):
        with open(os.environ["GMLX_COMFY_SETTINGS"], "w") as f:
            json.dump({"gmlx_install": "uv"}, f)
        stub = Stub("--contract", "0.9")
        try:
            with self.assertRaises(C.GmlxVersionError) as e:
                C.StepClient(stub.url).handshake()
            self.assertIn("gmlx restart", str(e.exception))
            self.assertIn("uv tool upgrade gmlx", str(e.exception))
        finally:
            stub.close()

    def test_a_server_with_no_contract_is_the_one_to_upgrade(self):
        stub = Stub("--no-contract-field", "--gmlx-version", "0.3.0")
        try:
            with self.assertRaises(C.GmlxVersionError) as e:
                C.StepClient(stub.url).handshake()
            self.assertIn("the server names no step contract", str(e.exception))
            self.assertIn("gmlx restart", str(e.exception))
            self.assertNotIn("Restart ComfyUI", str(e.exception))
        finally:
            stub.close()

    def test_a_server_restarted_as_another_gmlx_is_checked_at_the_next_get_client(self):
        """The loaders call get_client once per run, so a server restarted
        after a gmlx upgrade gets its note, and a contract change is caught
        there, by the same client."""
        port = str(_free_port())
        stub = Stub("--port", port, "--gmlx-version", C.PACK_VERSION)
        try:
            c = C.get_client(stub.url)
            self.assertIsNone(c.version_note)
        finally:
            stub.close()
        stub = Stub("--port", port, "--gmlx-version", "99.0.0")
        try:
            self.assertIs(C.get_client(stub.url), c)
            self.assertIn("gmlx 99.0.0", c.version_note)
            self.assertIn("Restart ComfyUI", c.version_note)
        finally:
            stub.close()
        stub = Stub("--port", port, "--gmlx-version", C.PACK_VERSION)
        try:
            C.get_client(stub.url)
            self.assertIsNone(c.version_note)
        finally:
            stub.close()
        stub = Stub("--port", port, "--contract", "2.0", "--gmlx-version", "99.0.0")
        try:
            with self.assertRaises(C.GmlxVersionError):
                C.get_client(stub.url)
        finally:
            stub.close()

    def test_another_gmlx_version_on_the_same_contract_is_a_note(self):
        # A source tree's server reports 0.0.0 or a local version: no note.
        cases = (("99.0.0", "Restart ComfyUI"), ("0.0.1", "gmlx restart"),
                 (C.PACK_VERSION, None), ("stub", None), ("0.0.0+unknown", None),
                 ("0.0.0", None), ("99.0.0+g1a2b3c4", None))
        for version, says in cases:
            stub = Stub("--gmlx-version", version)
            try:
                c = C.StepClient(stub.url)
                c.handshake()
                if says is None:
                    self.assertIsNone(c.version_note, version)
                else:
                    self.assertIn(says, c.version_note)
                    self.assertIn(f"gmlx {version}", c.version_note)
            finally:
                stub.close()


class ApiKey(unittest.TestCase):
    def test_the_key_travels_as_a_bearer_token(self):
        stub = Stub("--api-key", "sekret")
        try:
            with self.assertRaises(C.GmlxError) as e:
                C.StepClient(stub.url).handshake()
            self.assertIn("GMLX_API_KEY", str(e.exception))
            self.assertEqual(C.StepClient(stub.url, api_key="sekret").handshake()["contract"]["major"], 1)
        finally:
            stub.close()


class Discovery(unittest.TestCase):
    def setUp(self):
        self.env = {k: os.environ.get(k) for k in ("GMLX_BASE_URL", "GMLX_COMFY_SETTINGS", "PATH",
                                                    "GMLX_API_KEY")}
        # A 127.0.0.1 address that does not answer gives way to the one found
        # without it: that one answers nowhere either, whatever runs here.
        os.environ["GMLX_BASE_URL"] = _dead_url()

    def tearDown(self):
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_the_order_is_widget_env_settings_status_default(self):
        d = tempfile.mkdtemp()
        os.environ["PATH"] = d
        os.environ["GMLX_COMFY_SETTINGS"] = os.path.join(d, "none.json")
        os.environ.pop("GMLX_BASE_URL", None)
        self.assertEqual(C.resolve_base_url(""), C.DEFAULT_URL)
        exe = os.path.join(d, "gmlx")
        with open(exe, "w") as f:
            f.write("#!/bin/sh\necho '{\"running\": true, \"url\": \"http://127.0.0.1:9123\"}'\n")
        os.chmod(exe, 0o755)
        self.assertEqual(C.resolve_base_url(""), "http://127.0.0.1:9123")
        with open(os.environ["GMLX_COMFY_SETTINGS"], "w") as f:
            json.dump({"base_url": "http://127.0.0.1:9124", "api_key": "k1"}, f)
        self.assertEqual(C.resolve_base_url(""), "http://127.0.0.1:9124")
        self.assertEqual(C.resolve_api_key(), "k1")
        os.environ["GMLX_BASE_URL"] = "http://127.0.0.1:9125/"
        self.assertEqual(C.resolve_base_url(""), "http://127.0.0.1:9125")
        self.assertEqual(C.resolve_base_url("http://h:1"), "http://h:1")

    def _fake_gmlx(self, d):
        exe = os.path.join(d, "gmlx")
        with open(exe, "w") as f:
            f.write("#!/bin/sh\necho '{}'\n")
        os.chmod(exe, 0o755)
        return exe

    def test_a_down_server_without_start_names_the_remedy(self):
        d = tempfile.mkdtemp()
        os.environ["GMLX_COMFY_SETTINGS"] = os.path.join(d, "none.json")
        os.environ["PATH"] = d
        self._fake_gmlx(d)
        with self.assertRaises(C.GmlxError) as e:
            C.get_client("http://127.0.0.1:9", start_server=False)
        self.assertIn("run `gmlx serve`", str(e.exception))

    def test_the_recorded_gmlx_comes_first_then_path(self):
        d, rec = tempfile.mkdtemp(), tempfile.mkdtemp()
        os.environ["GMLX_COMFY_SETTINGS"] = os.path.join(d, "comfyui.json")
        os.environ["PATH"] = d
        on_path = self._fake_gmlx(d)
        recorded = self._fake_gmlx(rec)
        with open(os.environ["GMLX_COMFY_SETTINGS"], "w") as f:
            json.dump({"gmlx_command": recorded}, f)
        self.assertEqual(C.settings.gmlx_command(), recorded)
        os.remove(recorded)
        self.assertEqual(C.settings.gmlx_command(), on_path)

    def test_the_recorded_gmlx_is_found_without_path(self):
        """ComfyUI started from the Dock has no shell PATH: the server address
        and the server start come from the recorded gmlx."""
        d = tempfile.mkdtemp()
        os.environ["GMLX_COMFY_SETTINGS"] = os.path.join(d, "comfyui.json")
        os.environ["PATH"] = "/nonexistent"
        os.environ.pop("GMLX_BASE_URL", None)
        exe = os.path.join(d, "gmlx")
        with open(exe, "w") as f:
            f.write("#!/bin/sh\necho '{\"running\": true, \"url\": \"http://127.0.0.1:9126\"}'\n")
        os.chmod(exe, 0o755)
        with open(os.environ["GMLX_COMFY_SETTINGS"], "w") as f:
            json.dump({"gmlx_command": exe}, f)
        self.assertEqual(C.resolve_base_url(""), "http://127.0.0.1:9126")
        with self.assertRaises(C.GmlxError) as e:
            C.get_client("http://127.0.0.1:9", start_server=False)
        self.assertIn("run `gmlx serve`", str(e.exception))

    def test_a_recorded_gmlx_that_is_a_folder_or_does_not_run_is_named(self):
        d = tempfile.mkdtemp()
        os.environ["GMLX_COMFY_SETTINGS"] = os.path.join(d, "comfyui.json")
        os.environ["PATH"] = "/nonexistent"
        folder = os.path.join(d, "gmlx-folder")
        os.mkdir(folder)
        with open(os.environ["GMLX_COMFY_SETTINGS"], "w") as f:
            json.dump({"gmlx_command": folder}, f)
        self.assertIsNone(C.settings.gmlx_command())
        broken = os.path.join(d, "gmlx")
        with open(broken, "w") as f:
            f.write("#!/nonexistent/python\n")
        os.chmod(broken, 0o755)
        with open(os.environ["GMLX_COMFY_SETTINGS"], "w") as f:
            json.dump({"gmlx_command": broken}, f)
        with self.assertRaises(C.GmlxError) as e:
            C.get_client("http://127.0.0.1:9", start_server=True)
        msg = str(e.exception)
        self.assertIn(f"gmlx is not installed: {broken} does not run", msg)
        self.assertIn("Install gmlx again", msg)

    def test_no_gmlx_at_all_names_the_two_paths_to_delete(self):
        d = tempfile.mkdtemp()
        os.environ["GMLX_COMFY_SETTINGS"] = os.path.join(d, "comfyui.json")
        os.environ["PATH"] = "/nonexistent"
        with open(os.environ["GMLX_COMFY_SETTINGS"], "w") as f:
            json.dump({"gmlx_command": "/gone/bin/gmlx",
                       "pack_dir": "/data/gmlx/comfyui/comfyui-gmlx"}, f)
        with self.assertRaises(C.GmlxError) as e:
            C.get_client("http://127.0.0.1:9", start_server=True)
        msg = str(e.exception)
        self.assertIn("gmlx is not installed: /gone/bin/gmlx does not exist", msg)
        self.assertIn(f"deleting {C.settings.PACK_DIR} and /data/gmlx/comfyui", msg)


def _dead_url() -> str:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}"


class _Cancelled(Exception):
    pass


def _cancel_after(seconds: float):
    """An interrupt check that raises once ``seconds`` have passed."""
    t0 = time.monotonic()

    def check():
        if time.monotonic() - t0 > seconds:
            raise _Cancelled()
    return check


class _NotGmlx(http.server.BaseHTTPRequestHandler):
    """A server that answers /health and nothing else, as llama-server does."""

    def do_GET(self):
        body = b'{"status": "ok"}' if self.path == "/health" else b'{"error": "File Not Found"}'
        self.send_response(200 if self.path == "/health" else 404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def _not_gmlx(status: int = 404, body: bytes = b'{"error": "File Not Found"}'):
    """A server that answers /health, and the capabilities route with
    ``status`` and ``body``."""
    class Handler(_NotGmlx):
        def do_GET(self):
            if self.path == "/health":
                return super().do_GET()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


class StartAndCancel(unittest.TestCase):
    """A serve that fails names its error at once, Cancel stops every wait,
    and the server is the one a typed address names, or for the default
    address the gmlx server found without it."""

    def setUp(self):
        self.env = {k: os.environ.get(k) for k in ("GMLX_BASE_URL", "GMLX_COMFY_SETTINGS", "PATH",
                                                    "GMLX_API_KEY")}
        d = tempfile.mkdtemp()
        os.environ["GMLX_COMFY_SETTINGS"] = os.path.join(d, "comfyui.json")
        os.environ["PATH"] = "/nonexistent"
        os.environ["GMLX_BASE_URL"] = _dead_url()
        os.environ.pop("GMLX_API_KEY", None)
        self.dir = d

    def tearDown(self):
        C.set_interrupt_check(None)
        C.DEFAULT_URL = C.discovery.DEFAULT_URL
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def gmlx(self, serve: str, status: dict | None = None):
        """A recorded gmlx whose `serve` runs the shell ``serve`` and whose
        `status --json` prints ``status``."""
        exe = os.path.join(self.dir, "gmlx")
        with open(exe, "w") as f:
            f.write(f"#!/bin/sh\ncase \"$1\" in\n  serve) {serve} ;;\n"
                    f"  *) echo '{json.dumps(status or {})}' ;;\nesac\n")
        os.chmod(exe, 0o755)
        with open(os.environ["GMLX_COMFY_SETTINGS"], "w") as f:
            json.dump({"gmlx_command": exe}, f)

    def test_a_serve_that_fails_raises_its_error_at_once(self):
        self.gmlx("echo 'error: speech-to-text requires the optional stt extra' >&2; exit 1")
        t0 = time.monotonic()
        with self.assertRaises(C.GmlxError) as e:
            C.get_client(None, start_server=True, start_timeout=20)
        self.assertLess(time.monotonic() - t0, 3.0)
        msg = str(e.exception)
        self.assertIn("`gmlx serve` failed (exit 1)", msg)
        self.assertIn("error: speech-to-text requires the optional stt extra", msg)

    def _port(self) -> int:
        return int(os.environ["GMLX_BASE_URL"].rsplit(":", 1)[1])

    def test_a_server_that_dies_after_serve_exits_0_raises_with_its_log(self):
        """`gmlx serve` gave up waiting and exited 0, then the server died:
        gmlx status says it is not running, and its log names why."""
        logfile = os.path.join(self.dir, "server.log")
        with open(logfile, "w") as f:
            f.write("[server] loading\nMemoryError: the model does not fit\n")
        self.gmlx("exit 0", {"running": False, "port": self._port(), "log": logfile})
        t0 = time.monotonic()
        with self.assertRaises(C.GmlxError) as e:
            C.get_client(None, start_server=True, start_timeout=20)
        self.assertLess(time.monotonic() - t0, 5.0)
        msg = str(e.exception)
        self.assertIn("stopped before it answered", msg)
        self.assertIn("MemoryError: the model does not fit", msg)
        self.assertIn(logfile, msg)

    def test_a_second_serve_while_the_first_loads_keeps_the_wait(self):
        """A server already holds the address and still loads: that launch
        exits 1, and the wait goes on while gmlx status says it runs."""
        url = os.environ["GMLX_BASE_URL"]
        self.gmlx(f"echo 'error: a server already holds {url}' >&2; exit 1",
                  {"running": True, "port": self._port()})
        t0 = time.monotonic()
        with self.assertRaises(C.GmlxError) as e:
            C.get_client(None, start_server=True, start_timeout=2.5)
        self.assertGreater(time.monotonic() - t0, 2.4)
        self.assertIn("did not answer", str(e.exception))

    def test_a_serve_that_exits_0_is_waited_for(self):
        self.gmlx("exit 0", {"running": True, "port": self._port()})
        t0 = time.monotonic()
        with self.assertRaises(C.GmlxError) as e:
            C.get_client(None, start_server=True, start_timeout=1.5)
        self.assertGreater(time.monotonic() - t0, 1.4)
        self.assertIn("did not answer", str(e.exception))

    def test_cancel_stops_the_wait_for_a_starting_server(self):
        self.gmlx("exit 0", {"running": True, "port": self._port()})
        C.set_interrupt_check(_cancel_after(0.5))
        t0 = time.monotonic()
        with self.assertRaises(_Cancelled):
            C.get_client(None, start_server=True, start_timeout=30)
        self.assertLess(time.monotonic() - t0, 2.0)

    def test_cancel_stops_a_request_and_tells_the_server(self):
        stub = Stub("--step-delay-ms", "3000")
        try:
            c = C.StepClient(stub.url)
            _, fields = c.encode("z-image-turbo", "x")
            told = []
            C.set_interrupt_check(_cancel_after(0.3))
            t0 = time.monotonic()
            with self.assertRaises(_Cancelled):
                c.step(torch.zeros(1, 16, 4, 4), torch.tensor([0.5]),
                       {"model": "z-image-turbo", "run": None, "convention": "comfy",
                        "patch_table": None, "rows": [{"cond": fields["cond"]}]},
                       on_interrupt=lambda: told.append(1))
            self.assertLess(time.monotonic() - t0, 1.5)
            self.assertEqual(told, [1])
            # Without a cancel, a call waits for its answer as before.
            C.set_interrupt_check(lambda: None)
            self.assertEqual(c.status("z-image-turbo")["model"], "z-image-turbo")
        finally:
            stub.close()

    def test_a_typed_address_is_used_as_given(self):
        stub = Stub()
        other, other_url = _not_gmlx()
        try:
            os.environ["GMLX_BASE_URL"] = stub.url
            dead = _dead_url()
            with self.assertRaises(C.GmlxError) as e:
                C.get_client(dead, start_server=False)
            self.assertIn(f"no server answers at {dead}", str(e.exception))
            with self.assertRaises(C.GmlxError) as e:
                C.get_client("http://192.0.2.1:9", start_server=False)
            self.assertIn("192.0.2.1", str(e.exception))
            # A server that answers /health but not the handshake is not used.
            with self.assertRaises(C.GmlxError) as e:
                C.get_client(other_url, start_server=True)
            self.assertIn(f"a server answers at {other_url}, but it is not a gmlx image server",
                          str(e.exception))
            c = C.get_client(stub.url + "/", start_server=False)
            self.assertEqual((c.base_url, c.redirect_note), (stub.url, None))
        finally:
            stub.close()
            other.shutdown()
            other.server_close()

    def test_the_default_address_takes_the_gmlx_server_found_without_it(self):
        """llama-server answers /health on the default port; the gmlx server
        runs on another, which GMLX_BASE_URL names here."""
        stub = Stub()
        other, other_url = _not_gmlx()
        try:
            C.DEFAULT_URL = other_url
            os.environ["GMLX_BASE_URL"] = stub.url
            c = C.get_client(other_url, start_server=False)
            self.assertEqual(c.base_url, stub.url)
            self.assertIn(f"the loader's server_url is the default {other_url}, so the nodes "
                          f"use the gmlx server at {stub.url}", c.redirect_note)
            c = C.get_client("", start_server=False)
            self.assertEqual((c.base_url, c.redirect_note), (stub.url, None))
            # With no gmlx server found, the default is not used for one.
            os.environ["GMLX_BASE_URL"] = other_url
            with self.assertRaises(C.GmlxError) as e:
                C.get_client(other_url, start_server=False)
            self.assertIn("is not a gmlx image server", str(e.exception))
            dead = _dead_url()
            os.environ["GMLX_BASE_URL"] = dead
            with self.assertRaises(C.GmlxError) as e:
                C.get_client(other_url, start_server=False)
            msg = str(e.exception)
            self.assertIn(f"no server answers at {dead}", msg)
            self.assertIn(f"Note: the loader's server_url is the default {other_url}", msg)
        finally:
            stub.close()
            other.shutdown()
            other.server_close()

    def test_only_a_server_without_the_step_routes_is_not_gmlx(self):
        cases = [
            (404, b'{"error": "File Not Found"}', C.GmlxNotGmlx, "server.image.steps_api is off"),
            (200, b"<html>hi</html>", C.GmlxNotGmlx, "not a gmlx one"),
            (200, b"[1, 2]", C.GmlxNotGmlx, "not a gmlx one"),
            (200, b'{"contract": "1.0"}', C.GmlxNotGmlx, "not a gmlx one"),
            (200, b'{"contract": {"major": 1}, "min_client": [1]}', C.GmlxNotGmlx,
             "not a gmlx one"),
            (401, b'{"error": "missing or bad API key"}', C.GmlxError, "set GMLX_API_KEY"),
            (403, b'{"error": "the peer 10.0.0.5 is not loopback"}', C.GmlxError,
             "server.image.steps_remote"),
        ]
        for status, body, kind, says in cases:
            srv, url = _not_gmlx(status, body)
            try:
                with self.assertRaises(kind) as e:
                    C.get_client(url, start_server=False)
                msg = str(e.exception)
                self.assertIn(says, msg, (status, body))
                if status in (401, 403):
                    self.assertNotIsInstance(e.exception, C.GmlxNotGmlx)
                    self.assertNotIn("not a gmlx image server", msg)
                else:
                    self.assertIn(f"a server answers at {url}, but it is not a gmlx image "
                                  "server", msg)
            finally:
                srv.shutdown()
                srv.server_close()

    def test_cancel_stops_the_status_call_of_the_wait(self):
        """`gmlx status` hangs after `gmlx serve` exited: Cancel still stops
        the wait at once."""
        self.gmlx("exit 0", None)
        exe = os.path.join(self.dir, "gmlx")
        with open(exe, "w") as f:
            f.write("#!/bin/sh\ncase \"$1\" in\n  serve) exit 0 ;;\n  *) exec /bin/sleep 30 ;;\nesac\n")
        C.set_interrupt_check(_cancel_after(0.8))
        t0 = time.monotonic()
        with self.assertRaises(_Cancelled):
            C.get_client(None, start_server=True, start_timeout=30)
        self.assertLess(time.monotonic() - t0, 2.0)

    def test_the_interrupt_waits_5_s_at_most(self):
        self.assertEqual(C.INTERRUPT_TIMEOUT, 5.0)
        got = {}

        def request(method, path, body=None, headers=None, **kw):
            got.update(path=path, timeout=kw.get("timeout"))
            return 200, {}, b"{}"
        c = C.StepClient("http://127.0.0.1:9")
        c.request = request
        c.interrupt("r1")
        self.assertEqual(got, {"path": C.PREFIX + "/interrupt", "timeout": 5.0})


if __name__ == "__main__":
    unittest.main()
