"""MiniMax-H3 prompt upsampling: a chat model rewrites a typed prompt into
the three fields MiniMax's prompt guide lays out, before the encoder reads
it.

The guide is MiniMax's (``VIDEO_PROMPT_WRITING_GUIDE_base_en.md`` in
MiniMaxAI/MiniMax-H3), under the model's license, so gmlx ships none of
it: ``gmlx video pull`` fetches it at the revision the recipe pins, and
the entry names it as ``upsample_prompt``. The system prompt is a few
lines of gmlx's own that ask for the guide's fields as one JSON object,
then the guide as fetched. A clip with keyframes is the guide's I2VA,
FL2VA or L2VA task, which opens with an instruction line that names the
pictures and the clip's length; the upsampler sees no image, so the
lines name the task and the length, and the keyframes join the prompt
after the rewrite.
"""

from __future__ import annotations

import re

from gmlx.image import upsample
from gmlx.image.registry import UpsampleParts

GUIDE_REPO = "MiniMaxAI/MiniMax-H3"
GUIDE_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
GUIDE_FILE = "docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md"

# The guide's three fields, in its order.
FIELDS = ("integrated_multimodal_description", "overall_soundscape",
          "non_diegetic_music")
INSTRUCTION = "instruction"

_TASKS = {(False, False): ("T2VA", ""),
          (True, False): ("I2VA", "its first frame is the picture the user "
                                  "gives"),
          (True, True): ("FL2VA", "its first and last frames are the two "
                                  "pictures the user gives"),
          (False, True): ("L2VA", "its last frame is the picture the user "
                                  "gives")}


def task(first: bool, last: bool) -> str:
    """The guide's name of the task of a clip with these keyframes."""
    return _TASKS[(bool(first), bool(last))][0]


def seconds(frames: int, fps: int) -> str:
    """The clip's length as the guide's S.SS: the time of its last
    frame."""
    return f"{(frames - 1) / fps:.2f}"


def system(guide: str, *, frames: int | None, fps: int,
           first: bool = False, last: bool = False) -> str:
    """The system prompt of a clip of ``frames`` at ``fps``, None when its
    length is not known, with these keyframes: gmlx's lines, then the
    guide. A keyframe task names its length, which its instruction line
    holds."""
    name, pictures = _TASKS[(bool(first), bool(last))]
    if frames is None and pictures:
        raise ValueError(f"a {name} prompt names the clip's length")
    length = None if frames is None else seconds(frames, fps)
    clip = (f"a {name} video" if length is None
            else f"a {name} video of {length} seconds")
    unseen = ""
    if pictures:
        clip += f", where {pictures}"
        unseen = (" You see no picture: take what the pictures show from "
                  "the user's words.")
        lead = (f'"{INSTRUCTION}" is the guide\'s instruction line for '
                f"{name}, with S.SS as {length}")
    else:
        lead = f'"{INSTRUCTION}" is empty, since {name} has none'
    keys = ", ".join(f'"{k}"' for k in FIELDS)
    return (f"You rewrite a user's idea into a prompt for MiniMax-H3, which "
            f"makes a video with sound, as the guide below describes. The "
            f"prompt is for {clip}.{unseen} Keep every word the user "
            f"wants spoken, sung or shown, as written. Reply with one JSON "
            f'object and nothing else, with the keys "{INSTRUCTION}", '
            f"{keys}: {lead}, and each other key holds that field's text, "
            f"without its name.\n\nThe guide:\n\n{guide.strip()}\n")


def _laid_out(answer: str) -> str | None:
    """The fields of a reply that writes them as the guide lays them out,
    with the instruction paragraph before them when it names a picture."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", answer)]
    paras = [p for p in (re.sub(r"^```\w*\s*|\s*```$", "", q).strip()
                         for q in paras) if p]
    at = next((i for i, p in enumerate(paras)
               if p.startswith(FIELDS[0] + ":")), None)
    if at is None or len(paras) < at + 3 or not all(
            paras[at + j].startswith(k + ":") for j, k in enumerate(FIELDS)):
        return None
    body = paras[at:at + 3]
    if at and "Picture 1" in paras[at - 1]:
        body.insert(0, paras[at - 1])
    return "\n\n".join(body)


def parse(reply: str) -> str | None:
    """The caption of a reply: the guide's fields of its last JSON object
    that has all three, laid out as the guide does, after the instruction
    line when there is one; else the fields as the reply wrote them out.
    None when the reply holds no caption."""
    answer = upsample.after_thinking(reply)
    if answer is None:
        return None
    for obj in upsample._objects(answer):
        vals = [obj.get(k) for k in FIELDS]
        if not all(isinstance(v, str) and v.strip() for v in vals):
            continue
        out = [f"{k}: {v.strip()}" for k, v in zip(FIELDS, vals)]
        lead = obj.get(INSTRUCTION)
        if isinstance(lead, str) and lead.strip():
            out.insert(0, " ".join(lead.split()))
        return "\n\n".join(out)
    return _laid_out(answer)


UPSAMPLE = UpsampleParts(
    prompts=None,
    sha256="",
    key=None,
    # The guide's cases, 5 to 8 seconds, run 150 to 260 words; a 15 s clip
    # of several shots is the long case.
    max_tokens=6144,
    temperature=0.0,
    presence_penalty=0.0,
    thinking="off",
    repo=f"huggingface.co/{GUIDE_REPO}",
    revision=GUIDE_REVISION,
    suggested="Qwen3.5-9B",
    parse=parse,
    needs=("MiniMax-H3 ships no upsampler prompt: its upsampler reads "
           f"MiniMax's prompt guide, {GUIDE_FILE} in the Hugging Face "
           f"repository {GUIDE_REPO}, under the model's license. `gmlx "
           "video pull h3` fetches it and names it as upsample_prompt on the "
           "video_models entry, or pass --upsample-prompt to the file"),
)
