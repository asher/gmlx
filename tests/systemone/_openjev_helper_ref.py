# SPDX-License-Identifier: Apache-2.0
# openjev/openjev helper/shim.py at revision 0b6bb6e (sha256 81a22f1b1b89),
# vendored unmodified below this header as the reference for the letter
# readout parity test (Apache-2.0; see licenses/openjev-LICENSE). The test
# stubs the openai and transformers imports.
"""TypeSafe-shaped /v1/systemone shim over a vLLM letter readout.

Accepts jev-ultrafast's request (state + Choice questions), answers each question with ONE
prefill against the model and a softmax over the option letters. Options beyond 52 are handled
in two stages (per-chunk readout, then a readout over the chunk winners) composed into one full distribution (_distribution).
GET /v1/version reports the served model dir, calibration constants, flags and this file's sha256 (also the `model` of every answer).

Run: VLLM=http://localhost:8000/v1 python shim.py --port 8765
Point jev-ultrafast at http://localhost:8765/v1/systemone.
"""
import argparse, ast, json, math, os, threading, time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from openai import OpenAI

LETTERS = [chr(65 + i) for i in range(26)] + [chr(97 + i) for i in range(26)]
client = OpenAI(base_url=os.environ.get("VLLM", "http://localhost:8000/v1"), api_key="x")
TEMP = float(os.environ.get("READOUT_T", "1.1"))   # fitted on held-out cross-website steps for the merged model
PERMS = int(os.environ.get("READOUT_PERMS", "1"))   # >1 = average over that many letterings (4 ≈ +16 pts acc, 4× cost)
SHIM_TOKEN = os.environ.get("SHIM_TOKEN", "")  # if set, /v1/systemone requires Authorization: Bearer <token> (what the TypeSafe SDK sends as TYPESAFE_API_KEY)
TARGETED = os.environ.get("READOUT_TARGETED") == "1"  # OFF unless set: corrected score extraction (logprob_token_ids); changes probabilities, not the prompt, so READOUT_T and the yes/no slope must be fitted ON it
class ReadoutIncomplete(RuntimeError): pass
NOUL_T = float(os.environ.get("READOUT_NOUL_T", "3.0")); NOUL_BIAS = float(os.environ.get("READOUT_NOUL_BIAS", "-0.4"))  # yes/no calibration fit on 19 boolean sets, checked on 18 (results/noul_calibration.json)
_ids = {}
def letter_ids():
    if not _ids:
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained(os.environ.get("TOKENIZER", "Qwen/Qwen3.8-27B"))
        _ids.update({L: t.encode(L, add_special_tokens=False)[0] for L in LETTERS})
    return _ids

class State(str):
    """The state as text, plus an optional screenshot (data URL) that rides along to the model as an image."""
    image = None; fields = None

COMPACT = os.environ.get("SHIM_COMPACT") == "1"  # lossless: drop geometry/id fields, dedupe identical elements, drop non-interactive textless elements
COMPACT_CAP = int(os.environ.get("SHIM_COMPACT_CAP", "0"))  # >0 additionally truncates strings to this many chars: a lossy cap, off unless a held-out check on real states proves it neutral
DROP_KEYS = {"box", "rect", "bbox", "bounds", "xpath", "selector", "nonce", "n", "fingerprint", "marker", "page_key",
             "elapsed_ms", "started_at", "latency_ms", "executed_ms", "usage"}  # geometry/ids and per-send telemetry: no evidence, and the telemetry changes every send, which defeats prefix reuse across steps
INTERACTIVE = {"a", "button", "input", "select", "textarea", "option", "label", "summary"}

ID_KEYS = ("index", "id", "idx", "n", "marker")  # an element's identity: two same-label controls with different ids are two choices

def compact(state):
    """Drop geometry/id fields; dedupe only elements that repeat the same (tag, text, id) — an element without any id field is
    never deduped, so distinct same-label controls survive; keep interactive or texted/labelled elements; optional string cap."""
    def clean(v):
        if isinstance(v, dict): return {k: clean(x) for k, x in v.items() if k not in DROP_KEYS}
        if isinstance(v, list): return [clean(x) for x in v]
        return v[:COMPACT_CAP] if COMPACT_CAP and isinstance(v, str) and len(v) > COMPACT_CAP else v
    st = clean(state); els = state.get("elements")  # identity is read from the raw elements: clean() drops some id fields
    if isinstance(els, list):
        seen, out = set(), []
        for e in els:
            if not isinstance(e, dict): out.append(e); continue
            ident = next((str(e[k]) for k in ID_KEYS if e.get(k) is not None), None)
            key = (e.get("tag"), (e.get("text") or "").strip(), ident)
            if ident is not None and key in seen: continue
            if e.get("tag") not in INTERACTIVE and not (e.get("text") or e.get("label") or e.get("aria") or e.get("role")): continue
            seen.add(key); out.append(clean(e))
        st["elements"] = out
    return st

LAYOUT = os.environ.get("SHIM_LAYOUT", "")  # page_first: stable page/elements/goal first, mutable history and telemetry last, so consecutive steps of a task share the longest prefix (a change of the trained layout: web-300 check before go-live)
STABLE_FIRST = ("page", "elements", "goal", "task", "plan")
def page_first(state):
    return {**{k: state[k] for k in STABLE_FIRST if k in state}, **{k: v for k, v in state.items() if k not in STABLE_FIRST}}

def with_image(state):
    """state.screenshot / state.image (data URL or raw base64 PNG/JPEG) becomes the image; the rest is the text state."""
    if COMPACT and isinstance(state, dict): state = compact(state)
    if LAYOUT == "page_first" and isinstance(state, dict): state = page_first(state)
    if isinstance(state, dict):
        for key in ("screenshot", "image"):
            v = state.get(key)
            if isinstance(v, str) and (v.startswith("data:image") or len(v) > 2000):
                rest = {k: x for k, x in state.items() if k != key}; st = State(json.dumps(rest, ensure_ascii=False) if rest else "(see screenshot)")
                st.image = v if v.startswith("data:image") else "data:image/png;base64," + v; st.fields = rest; return st
        return State(json.dumps(state, ensure_ascii=False))
    return State(state)

PAD = int(os.environ.get("SHIM_PAD", "0"))  # 784 = the cache block of this model: pad the shared page prefix to a block boundary so every question reuses all page blocks
_hdr = {}
def _header():
    """Chat-template text before the user content, so prefix token counts match what vLLM sees."""
    if not _hdr:
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained(os.environ.get("TOKENIZER", "Qwen/Qwen3.8-27B")); _hdr["t"] = t
        r = t.apply_chat_template([{"role": "user", "content": "X"}], tokenize=False, add_generation_prompt=True, enable_thinking=False); _hdr["h"] = r.split("X")[0]
    return _hdr["t"], _hdr["h"]

def pad_prefix(prefix):
    """Append filler so the chat header + prefix is 2 tokens past a multiple of PAD tokens (the boundary stays inside the shared text)."""
    if not PAD: return prefix
    t, h = _header(); n = len(t.encode(h + prefix, add_special_tokens=False)); k = (2 - n) % PAD
    if k < 2: k += PAD
    out = prefix + "\n" + " pad" * k
    for _ in range(3):  # BPE can merge fillers; correct the count
        m = len(t.encode(h + out, add_special_tokens=False)); d = (m - 2) % PAD
        if d == 0: break
        out = out + " pad" * (PAD - d) if d > PAD // 2 else out[:len(out) - 4 * d]
    return out

def prefix_text(state_text):
    """The shared part of a text-mode prompt (everything before the question)."""
    return pad_prefix(f"State:\n{state_text}")

def prewarm(state_text):
    """One-token prefill of the shared prefix so a following decision call only pays for the question tails."""
    r = client.chat.completions.create(model="qwen", max_tokens=1, temperature=0, messages=[{"role": "user", "content": prefix_text(state_text)}],
                                       extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return r.usage.prompt_tokens

HIST_KEYS = ("previous_actions", "history", "recent_actions")
def _task(fields, instructions):
    """Task line of the trained screenshot layout: state.task (Mind2Web/ops shape) or instructions.goal (harness shape)."""
    if fields.get("task"): return str(fields["task"])
    if instructions.startswith("{"):  # the goal is read from the STRUCTURE whatever the presentation style: JSON text, or the Python-literal text of READOUT_INSTR_STYLE=pyrepr (literal_eval: literals only, never code)
        for parse in (json.loads, ast.literal_eval):
            try: v = parse(instructions)
            except (ValueError, SyntaxError): continue
            return v.get("goal") if isinstance(v, dict) else None
        return None

def _readout_once(state_text, instructions, options):
    """One lettering, one prefill. Returns temperature-scaled probs aligned with options.
    Layouts: (1) image + state.task [+ previous_actions|history] → the trained screenshot layout (image lane: task, previous
    actions, red-letter-marked candidates, question). (2) image + harness shape (page/elements/recent_actions, goal inside the
    question's instructions) → the same layout with page/elements as Context and unmarked candidates; the image lane never
    trained on this shape (the text lane trained on it as JSON, without an image), so it needs the matched check before it is
    relied on. (3) image + any other shape, or no image → the text-lane layout (State JSON, Question, Options)."""
    lines = "\n".join(f"[{LETTERS[i]}] {k}: {d}" for i, (k, d) in enumerate(options))
    image = getattr(state_text, "image", None); fields = getattr(state_text, "fields", None) or {}
    task = _task(fields, instructions) if image else None
    if task:
        hist = next((fields[k] for k in HIST_KEYS if fields.get(k)), [])
        hist = "\n".join(f"- {h if isinstance(h, str) else json.dumps(h, ensure_ascii=False)}" for h in (hist if isinstance(hist, list) else [hist])) or "- (none)"
        extra = {k: v for k, v in fields.items() if k not in ("task",) + HIST_KEYS}
        marks = "\n".join(f"[{LETTERS[i]}] {d if d else k}" for i, (k, d) in enumerate(options))
        shown = "The screenshot shows the current page with candidate elements marked by red letters." if "task" in fields else "The screenshot shows the current page; candidate elements:"
        page = f"{shown}\n{marks}\n"
        task = f"Task: {task}\nPrevious actions:\n{hist}\n" + (f"Context: {json.dumps(extra, ensure_ascii=False)}\n" if extra else "")
        prompt = (page + "\n" + task if LAYOUT == "page_first" else task + "\n" + page) + f"\n{instructions} Answer with the letter only."
    else:
        prompt = ((f"The screenshot shows the current screen.\nState:\n{state_text}" if image else prefix_text(state_text))
                  + f"\n\nQuestion: {instructions}\nOptions:\n{lines}\n\nAnswer with the letter of the best option only.")
    content = [{"type": "image_url", "image_url": {"url": image}}, {"type": "text", "text": prompt}] if image else prompt
    ids = letter_ids(); allowed = [ids[LETTERS[i]] for i in range(len(options))]
    if TARGETED:  # exact scores of exactly the candidate token ids, matched BY ID: top_logprobs is the raw top-K over the whole vocabulary, taken before the allowed-token mask, and drops labels on many-option heads
        assert len(set(allowed)) == len(allowed), "candidate token ids are not unique"
        r = client.chat.completions.create(model="qwen", max_tokens=1, temperature=0, logprobs=True, messages=[{"role": "user", "content": content}],
            extra_body={"allowed_token_ids": allowed, "logprob_token_ids": allowed, "return_tokens_as_token_ids": True, "chat_template_kwargs": {"enable_thinking": False}})
        got = {t.token: t.logprob for t in r.choices[0].logprobs.content[0].top_logprobs}; raw = [got.get(f"token_id:{i}") for i in allowed]
        if any(v is None or not math.isfinite(v) for v in raw): raise ReadoutIncomplete(f"{sum(v is None or not math.isfinite(v) for v in raw)} of {len(allowed)} candidate scores missing or not finite")  # never floored, never a whitespace look-alike
        z = [v / TEMP for v in raw]
    else:
        r = client.chat.completions.create(model="qwen", max_tokens=1, temperature=0, logprobs=True, top_logprobs=max(20, len(options)),
            messages=[{"role": "user", "content": content}],
            extra_body={"allowed_token_ids": allowed, "chat_template_kwargs": {"enable_thinking": False}})
        lp = {}
        for t in r.choices[0].logprobs.content[0].top_logprobs:  # exact letter token wins; ' U' must not overwrite 'U'
            k = t.token if t.token in LETTERS else t.token.strip()
            if k not in lp or t.token in LETTERS: lp[k] = t.logprob
        z = [lp.get(LETTERS[i], -30.0) / TEMP for i in range(len(options))]  # OLD PROTOCOL: a label outside the raw top-K gets the -30 floor
    m = max(z); e = [math.exp(v - m) for v in z]; s = sum(e)
    return [v / s for v in e], r.usage.prompt_tokens

def readout(state_text, instructions, options):
    """options: list of (key, description). Averages over PERMS letterings (per option), so the
    returned probs are aligned with the caller's option order regardless of which letter each got."""
    if PERMS <= 1: return _readout_once(state_text, instructions, options)
    import random
    acc = [0.0] * len(options); tokens = 0
    for j in range(PERMS):
        order = list(range(len(options))); random.Random(j).shuffle(order)
        p, t = _readout_once(state_text, instructions, [options[i] for i in order]); tokens += t
        for pos, i in enumerate(order): acc[i] += p[pos] / PERMS
    return acc, tokens

class BadQuestion(ValueError): pass

def _instr(q):
    i = q.get("instructions")
    if i is None: raise BadQuestion("instructions is required")
    if isinstance(i, str): return i
    return str(i) if os.environ.get("READOUT_INSTR_STYLE") == "pyrepr" else json.dumps(i, ensure_ascii=False)  # pyrepr = what distill_train.build's f-string wrote for dict instructions in the v8 text run (DOM rows); set ONLY on a box serving a model trained that way

def _desc(v):
    return "" if v is None else (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))

# official formulas (typesafe-ai/system-one-adapter-python confidence_metrics.py)
def choice_confidence(p):
    if len(p) == 1: return 1.0
    u = 1.0 / len(p); return max(0.0, (max(p) - u) / (1.0 - u))

def score_confidence(p):
    if len(p) == 1: return 1.0
    mode = max(range(len(p)), key=p.__getitem__)
    dist = sum(pi * abs(i - mode) for i, pi in enumerate(p))
    c = (len(p) - 1) / 2; umad = sum(abs(i - c) for i in range(len(p))) / len(p)
    return max(0.0, 1.0 - dist / umad)

def _distribution(state_text, instr, opts):
    """opts: list of (key, description) → (probs aligned with opts, tokens).
    Above 52 options (one letter each) the readout runs per near-equal chunk of ≤52, then once over the chunk winners, and
    the two are composed: p(i) ∝ p_final(chunk of i) · p_chunk(i) / p_chunk(winner of that chunk), normalised over all options.
    Approximation: an option's letter logit is taken to be the same in its chunk prompt and in the winners prompt, so each
    winner anchors its chunk's mass on the winners' scale; when that holds exactly, p equals the softmax over all options.
    Every option gets a non-zero probability, the distribution sums to 1, it does not depend on option order beyond float
    noise, and confidence is computed on it (not on the winners alone)."""
    if len(opts) <= 52: return readout(state_text, instr, opts)
    k = -(-len(opts) // 52); size = -(-len(opts) // k); chunks = [opts[i:i + size] for i in range(0, len(opts), size)]
    parts, tokens = [], 0
    for chunk in chunks:
        p, t = readout(state_text, instr, chunk); tokens += t; parts.append((p, max(range(len(p)), key=p.__getitem__)))
    pf, t = readout(state_text, instr, [chunk[w] for chunk, (_, w) in zip(chunks, parts)]); tokens += t
    raw = [pf[c] * pi / p[w] for c, (p, w) in enumerate(parts) for pi in p]; s = sum(raw)
    return [v / s for v in raw], tokens

def answer_choice(state_text, q):
    crit = q.get("criteria")
    if not isinstance(crit, dict) or not crit: raise BadQuestion("choice.criteria must be a non-empty map of option -> description|null")
    opts = [(k, _desc(v)) for k, v in crit.items()]
    p, tokens = _distribution(state_text, _instr(q), opts)
    choice = opts[max(range(len(p)), key=lambda i: p[i])][0]; probs = {k: round(v, 4) for (k, _), v in zip(opts, p)}  # winner picked before rounding
    return {"type": "choice", "choice": choice, "probabilities": probs, "confidence": round(choice_confidence(p), 4)}, tokens

def answer_score(state_text, q):
    levels = q.get("criteria")
    if not isinstance(levels, list) or len(levels) < 2: raise BadQuestion("score.criteria must be an ordered array of at least two levels")
    opts = [(str(i), _desc(l)) for i, l in enumerate(levels)]
    p, tokens = _distribution(state_text, _instr(q) + " Rate along the ordered levels below (lowest first).", opts)
    return {"type": "score", "score": round(sum(i * pi for i, pi in enumerate(p)), 4),
            "legend": {str(i): l for i, l in enumerate(levels)}, "probabilities": {str(i): round(pi, 4) for i, pi in enumerate(p)},
            "confidence": round(score_confidence(p), 4)}, tokens

def answer_noul(state_text, q):
    crit = q.get("criteria") or {}
    yes = _desc(crit.get("true")) or "The statement is true."; no = _desc(crit.get("false")) or "The statement is false."
    p, tokens = _distribution(state_text, _instr(q), [("yes", yes), ("no", no)])
    py = min(max(p[0], 1e-4), 1 - 1e-4); z = math.log(py / (1 - py)) / NOUL_T + NOUL_BIAS  # the raw readout over-says yes on skewed sets
    return {"type": "noul", "noul": round(1 / (1 + math.exp(-z)), 4)}, tokens

ANSWER = {"choice": answer_choice, "score": answer_score, "noul": answer_noul}

LOOP_BREAK = os.environ.get("SHIM_LOOP_BREAK") == "1"  # OFF unless set: the owner classed it a benchmark-specific patch, not model behaviour. On: an action repeated ≥3× without the page changing is masked out of the candidates (exact key or exact description match only; never below two candidates)
def loop_break(state, qs):
    ra = state.get("recent_actions") or []
    if len(ra) < 3: return qs
    key = lambda h: (str(h.get("action") or ""), str(h.get("kind") or ""))
    last = ra[-3:]
    if len({key(h) for h in last}) != 1 or any(h.get("page_changed") for h in last): return qs
    act, kind = key(last[-1]); n_rep = 0
    for h in reversed(ra):
        if key(h) == (act, kind) and not h.get("page_changed"): n_rep += 1
        else: break
    if not act: return qs
    out = {}
    for qid, q in qs.items():
        if q.get("type") == "choice" and isinstance(q.get("criteria"), dict):
            crit = q["criteria"]; keys = list(crit)
            drop = {k for k in keys if k == act or _desc(crit[k]) == act}
            if n_rep >= 4 and kind in keys: drop.add(kind)  # the operation itself, after 4 fruitless repeats
            if drop and len(keys) - len(drop) >= 2:
                q = {**q, "criteria": {k: v for k, v in crit.items() if k not in drop}}
                print(json.dumps({"loop_break": qid, "dropped": sorted(drop)[:4], "repeats": n_rep}), flush=True)
        out[qid] = q
    return out


def passthrough_chat(body):
    """Forward a chat completion to vLLM with thinking forced off; drop fields vLLM does not know."""
    import urllib.request
    for k in ("reasoning", "thinking"): body.pop(k, None)
    body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(str(client.base_url).rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer x"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r: return r.read()

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive: far-away clients otherwise pay a full round trip per call
    disable_nagle_algorithm = True  # headers and body go out immediately instead of waiting for the ACK (one round trip less)
    pool = ThreadPoolExecutor(max_workers=int(os.environ.get("SHIM_POOL", "16")))
    STAGGER = os.environ.get("SHIM_STAGGER") == "1"
    STAGGER_MIN_CHARS = int(os.environ.get("SHIM_STAGGER_MIN_CHARS", "16000"))  # ~4k tokens: below that all questions fit one scheduler step and share blocks in flight anyway
    def log_message(self, *a): pass
    def do_POST(self):
        if SHIM_TOKEN and self.headers.get("Authorization", "") != "Bearer " + SHIM_TOKEN:
            return self.send_json(401, {"error": {"code": 401, "message": "missing or invalid bearer token"}})
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        t0 = time.perf_counter()
        if self.path.rstrip("/").endswith("/version"): return self.send_json(200, VERSION)
        if self.path.endswith("/chat/completions"):
            try: data = passthrough_chat(body); code = 200
            except Exception as e: data = json.dumps({"error": str(e)[:300]}).encode(); code = 502
            self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
            print(json.dumps({"chat_ms": round((time.perf_counter() - t0) * 1000), "code": code}), flush=True); return
        if self.path.endswith("/prewarm"):
            state = body.get("state")
            if state is None: return self.send_json(422, {"error": {"code": 422, "message": "state is required"}})
            try: n = prewarm(with_image(state))
            except Exception as e: return self.send_json(502, {"error": {"code": 502, "message": str(e)[:200]}})
            print(json.dumps({"prewarm_tokens": n, "ms": round((time.perf_counter() - t0) * 1000)}), flush=True); return self.send_json(200, {"ok": True, "prompt_tokens": n})
        state = body.get("state"); qs = body.get("questions")
        if state is None or not isinstance(qs, dict) or not qs:
            return self.send_json(422, {"error": {"code": 422, "message": "state and a non-empty questions map are required"}})
        if LOOP_BREAK and isinstance(state, dict):
            try: qs = loop_break(state, qs)
            except Exception as e: print(json.dumps({"loop_break_error": str(e)[:120]}), flush=True)
        if isinstance(state, dict) and isinstance(state.get("recent_actions"), list) and state["recent_actions"]:
            try: print(json.dumps({"trace": [[h.get("kind"), h.get("action"), bool(h.get("page_changed")), h.get("choice")] for h in state["recent_actions"][-5:] if isinstance(h, dict)]}), flush=True)
            except Exception: pass
        state_text = with_image(state)  # str subclass; carries state.screenshot as an image when present
        for qid, q in qs.items():
            if not isinstance(q, dict) or q.get("type") not in ANSWER:
                return self.send_json(422, {"error": {"code": 422, "message": f"questions.{qid}.type must be one of choice, score, noul"}})
        try:
            items = list(qs.items()); f = lambda it: (it[0], ANSWER[it[1]["type"]](state_text, it[1]))
            stagger = self.STAGGER and len(state_text) >= self.STAGGER_MIN_CHARS  # first question alone so its state blocks are cached before the rest run
            results = ([f(items[0])] + list(self.pool.map(f, items[1:]))) if stagger else list(self.pool.map(f, items))
        except BadQuestion as e:
            return self.send_json(422, {"error": {"code": 422, "message": str(e)}})
        answers = {qid: a for qid, (a, _) in results}; toks = int(sum(t for _, (_, t) in results))
        out = {"id": f"shim-{int(time.time() * 1000)}", "model": MODEL_STRING, "answers": answers,
               "usage": {"input_tokens": toks, "output_tokens": 0}}
        self.send_json(200, out)
        print(json.dumps({"questions": len(qs), "tokens": toks, "ms": round((time.perf_counter() - t0) * 1000), "ops": {q: a.get("choice") for q, a in answers.items() if "choice" in a}}), flush=True)

    def do_GET(self):
        if SHIM_TOKEN and self.headers.get("Authorization", "") != "Bearer " + SHIM_TOKEN:
            return self.send_json(401, {"error": {"code": 401, "message": "missing or invalid bearer token"}})
        if self.path.rstrip("/").endswith("/version"): return self.send_json(200, VERSION)
        self.send_json(404, {"error": {"code": 404, "message": "GET /v1/version"}})

    def send_json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

def _version():
    """What is served, so a tester can pin what they measured: model dir, calibration constants, shim flags, this file's sha256. Never the token."""
    import hashlib
    flags = {"perms": PERMS, "stagger": H.STAGGER, "loop_break": LOOP_BREAK, "compact": COMPACT, "compact_cap": COMPACT_CAP, "layout": LAYOUT, "pad": PAD, "targeted": TARGETED, "instr_style": os.environ.get("READOUT_INSTR_STYLE") or "json"}
    model = os.path.basename((os.environ.get("SHIM_MODEL") or os.environ.get("TOKENIZER", "")).rstrip("/")) or "unknown"
    return {"model_dir": model, "T": TEMP, "noul_t": NOUL_T, "noul_bias": NOUL_BIAS, "flags": flags,
            "shim_file": os.path.basename(__file__), "shim_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}
VERSION = _version()
MODEL_STRING = (f"{VERSION['model_dir']} T={TEMP} noul={NOUL_T},{NOUL_BIAS} flags=" + json.dumps(VERSION["flags"], separators=(",", ":"))
                + f" shim={VERSION['shim_file']}@{VERSION['shim_sha256'][:12]}")  # the `model` of every answer

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8765); ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to expose (use SHIM_TOKEN)"); a = ap.parse_args()
    letter_ids(); print(f"shim on :{a.port} -> {client.base_url}", flush=True)
    ThreadingHTTPServer.request_queue_size = 256; ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()
