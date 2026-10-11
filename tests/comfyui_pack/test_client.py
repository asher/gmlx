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


if __name__ == "__main__":
    unittest.main()
