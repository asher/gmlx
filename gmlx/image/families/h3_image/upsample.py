"""MiniMax-H3-Image prompt upsampling: MiniMax's prompt expander turns a
typed prompt into the caption the DiT was trained on, as
MiniMaxAI/MiniMax-H3-Image-PE lays the turn out.

MiniMax's prompts are under the model's license, so gmlx ships none of
them: ``gmlx pull minimax-h3-image`` fetches ``prompt/manifest.json`` and
the files it lists at the revision the recipe pins, and the entry names
the manifest as ``upsample_prompt``. A prompt without references is the
text to image turn: the system prompt as the system turn, with its
trailing newlines cut, and the task prompt with the typed prompt in its
slot as the user turn. A prompt with references is the editing turn, one
user turn of content parts: the editing task prompt, the picture count,
each picture's label and shape before the picture, then the typed prompt
as a JSON string and the aspect ratio the request selects, ``adaptive``
when the image takes its size from the references. The turn samples as
MiniMax's request does, at temperature 1 with top-p 0.95 and top-k 20,
thinking on at the template's default effort, in 4096 tokens with the
thinking; the image's seed seeds it. The reply's ``expanded_caption`` is
the caption. MiniMax's JSON schema, which its server decodes under, and
the aspect ratio the reply recommends are not read: the image keeps the
size the request asks for.
"""

from __future__ import annotations

import json
import math
import os

from gmlx.image import upsample
from gmlx.image.registry import UpsampleParts

PE_REPO = "MiniMaxAI/MiniMax-H3-Image-PE"
PE_REVISION = "35603d4b63949afa12cb098227fffefedf081afa"
MANIFEST = "prompt/manifest.json"
# The manifest's tasks and the names of the files each reads.
SYSTEM, TASK = "system_prompt.txt", "task_prompt.txt"
_FILES = (("t2i", SYSTEM, "system"), ("t2i", TASK, "task"),
          ("edit", TASK, "edit"))

NEEDS = ("MiniMax-H3-Image ships no upsampler prompt: its upsampler reads "
         f"MiniMax's prompt files, {MANIFEST} and the files it lists, in "
         f"the Hugging Face repository {PE_REPO}, under the MiniMax-H3-Image "
         "Community License. `gmlx pull minimax-h3-image` fetches them and "
         "names the manifest as upsample_prompt on the image_models entry, "
         "or pass --upsample-prompt to the manifest")


def _listed(manifest: dict, task: str, name: str) -> str | None:
    files = (manifest.get(task) or {}).get("files") or {}
    hits = [p for p in files if isinstance(p, str)
            and p.rsplit("/", 1)[-1] == name]
    return hits[0] if len(hits) == 1 else None


def read(path: str, family=None) -> str:
    """The prompts of MiniMax's manifest at ``path``, as one JSON text of
    the text to image system and task prompts and the editing task
    prompt, each as MiniMax's example reads it. The files sit where the
    manifest lists them, under the folder above the manifest's."""
    where = os.path.expanduser(path)
    try:
        with open(where, encoding="utf-8") as f:
            manifest = json.load(f)
    except FileNotFoundError:
        raise upsample.UpsampleError(
            f"upsample prompt {path}: no such file. {NEEDS}") from None
    except (OSError, UnicodeDecodeError) as e:
        raise upsample.UpsampleError(f"upsample prompt {path}: {e}") from None
    except ValueError:
        manifest = None
    root = os.path.dirname(os.path.dirname(os.path.abspath(where)))
    out = {}
    for task, name, key in _FILES:
        rel = _listed(manifest, task, name) if isinstance(manifest, dict) \
            else None
        if rel is None:
            raise upsample.UpsampleError(
                f"upsample prompt {path} is not MiniMax's {MANIFEST}, which "
                f"lists {task}'s {name}. {NEEDS}")
        try:
            with open(os.path.join(root, rel), encoding="utf-8") as f:
                text = f.read()
        except (OSError, UnicodeDecodeError) as e:
            raise upsample.UpsampleError(
                f"upsample prompt {path}: {rel}: {e}") from None
        # MiniMax's script cuts the system prompt's trailing newlines and
        # takes the task prompts as they are.
        out[key] = text.rstrip("\n") if key == "system" else text
    return json.dumps(out, ensure_ascii=False, sort_keys=True)


def _ratio(width: int, height: int) -> str:
    g = math.gcd(width, height)
    return f"{width // g}:{height // g}"


def turn(text: str, prompt: str, pictures: tuple,
         canvas: tuple | None = None) -> tuple:
    """The chat turn of ``prompt`` from the prompts ``read`` gives:
    MiniMax's text to image turn, or with ``pictures`` its editing turn,
    with the aspect ratio of ``canvas`` selected, or ``adaptive``."""
    t = json.loads(text)
    if not pictures:
        return ({"role": "system", "content": t["system"]},
                {"role": "user", "content": t["task"].format(prompt)})
    n = len(pictures)
    content = [
        {"type": "text", "text": t["edit"]},
        {"type": "text", "text": (
            f"Reference Image Count:\n{n} input image(s), labeled "
            f"<Picture 1> through <Picture {n}>.")}]
    for i, pic in enumerate(pictures, 1):
        content += [{"type": "text", "text": (
                        f"Reference Media:\n- <Picture {i}> "
                        f"[aspect_ratio {pic.ratio}]")},
                    {"type": "image"}]
    choice = "adaptive" if canvas is None else _ratio(*canvas)
    content.append({"type": "text", "text": (
        "\nUser Editing Instruction (JSON string):\n"
        f"{json.dumps(prompt, ensure_ascii=False)}\n\n"
        f"User Aspect Ratio Selection:\n{choice}\n\n"
        "Apply the instructions above to the supplied inputs and reference "
        "images. Output only the required two-field JSON object.")})
    return ({"role": "user", "content": content},)


UPSAMPLE = UpsampleParts(
    prompts=None,
    sha256="",
    key=None,
    # prompt/t2i/request_params.json: max_tokens 4096 with the thinking,
    # temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence and
    # repetition penalties off, enable_thinking true, reasoning_effort
    # xhigh (the template's default).
    max_tokens=4096,
    temperature=1.0,
    presence_penalty=0.0,
    thinking="on",
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    repo=f"huggingface.co/{PE_REPO}",
    revision=PE_REVISION,
    suggested="MiniMax-H3-Image-PE",
    parse=upsample.json_field("expanded_caption"),
    read=read,
    turn=turn,
    pictures=True,
    needs=NEEDS,
)
