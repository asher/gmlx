"""gmlx/video/families/h3/upsample.py: the MiniMax-H3 upsampler's system
prompt, which names the guide's task and the clip's length, and its parse
of the reply into the guide's three fields. The guide itself is a stand-in:
gmlx ships none of MiniMax's text."""
from __future__ import annotations

import json

import pytest

from gmlx.image.upsample import UpsampleError, chat_request, upsample
from gmlx.video.families.h3 import family as hf
from gmlx.video.families.h3 import upsample as hu

GUIDE = "# A stand-in guide\n\nWrite the three fields."
FIELDS = {"integrated_multimodal_description": "[Shot 1] Live-action, a fox "
          "runs through snow.",
          "overall_soundscape": "Wind and soft footsteps in snow.",
          "non_diegetic_music": "N/A"}
LAID_OUT = ("integrated_multimodal_description: [Shot 1] Live-action, a fox "
            "runs through snow.\n\noverall_soundscape: Wind and soft "
            "footsteps in snow.\n\nnon_diegetic_music: N/A")
# An instruction line of gmlx's own invention, in the shape a keyframe
# caption opens with; no text of MiniMax's guide ships in gmlx.
LINE = ("Keyframe note: <Picture 1> opens [Shot 1] at 0.00 s and holds its "
        "look.")


@pytest.mark.parametrize("first, last, task, length", [
    (False, False, "T2VA", "5.17"),
    (True, False, "I2VA", "5.17"),
    (True, True, "FL2VA", "5.17"),
    (False, True, "L2VA", "5.17"),
])
def test_the_system_prompt_names_the_task_and_the_clips_length(
        first, last, task, length):
    """124 frames at 24 fps last 5.17 s, the S.SS a keyframe task's
    instruction line holds; the guide follows gmlx's lines whole."""
    text = hf.FAMILY.upsample_system(GUIDE, 124, first, last)
    article = "a" if task == "T2VA" else "an"
    assert f"{article} {task} video of {length} seconds" in text
    assert text.endswith(f"The guide:\n\n{GUIDE}\n")
    if task == "T2VA":
        assert '"instruction" is empty, since T2VA has none' in text
        assert "You see no picture" not in text
    else:
        assert (f'"instruction" is the guide\'s instruction line for {task}, '
                f"with S.SS as {length}") in text
        assert "You see no picture" in text
    assert text[:text.index("The guide:")].isascii()


def test_the_guides_eight_second_case_ends_at_its_length():
    """The guide's 8 s FL2VA case puts its last picture at 8.00: 192
    frames at 24 fps, frames over fps as ComfyUI's node counts it."""
    assert hu.seconds(192, 24) == "8.00"


def test_a_prompt_of_no_known_length_is_a_text_to_video_one():
    """The pack's encode knows no clip length: its prompt names none, and
    a keyframe task, whose instruction line holds the length, needs it."""
    assert "a T2VA video. " in hf.FAMILY.upsample_system(GUIDE, None,
                                                        False, False)
    with pytest.raises(ValueError, match="^an I2VA prompt names the "
                                         "clip's length$"):
        hf.FAMILY.upsample_system(GUIDE, None, True, False)


@pytest.mark.parametrize("reply, want", [
    (json.dumps({"instruction": "", **FIELDS}), LAID_OUT),
    (json.dumps({"instruction": LINE, **FIELDS}), f"{LINE}\n\n{LAID_OUT}"),
    # Its reasoning first, then the object in a code fence.
    ("<think>the fox</think>```json\n" + json.dumps(FIELDS) + "\n```",
     LAID_OUT),
    # The fields written out as the guide lays them out, after a heading.
    (f"Here is the prompt:\n\n{LINE}\n\n{LAID_OUT}\n", f"{LINE}\n\n{LAID_OUT}"),
    (f"```text\n{LAID_OUT}\n```", LAID_OUT),
])
def test_the_reply_becomes_the_guides_three_fields(reply, want):
    assert hu.parse(reply) == want


@pytest.mark.parametrize("reply", [
    json.dumps({**FIELDS, "overall_soundscape": ""}),
    json.dumps({"caption": "a fox"}),
    "<think>the fox runs and",
    "a fox in snow",
])
def test_a_reply_without_the_three_fields_holds_no_caption(reply):
    assert hu.parse(reply) is None


def test_the_upsampler_turn_and_its_refusal():
    """The turn decodes greedily with thinking off, at most 6144 tokens,
    and a reply without the fields names the upsampler and the way out.
    With no guide file, the refusal says where to get it."""
    system = hf.FAMILY.upsample_system(GUIDE, 124, False, False)
    req = chat_request(hf.FAMILY, "a fox in snow", system)
    assert (req.max_tokens, req.temperature, req.thinking) == (6144, 0.0,
                                                               "off")
    assert req.user == "a fox in snow" and req.system == system
    assert upsample(hf.FAMILY, "a fox", lambda r: json.dumps(FIELDS),
                    system=system) == LAID_OUT
    with pytest.raises(UpsampleError, match="the upsampler wrote a reply "
                       "with no caption in it"):
        upsample(hf.FAMILY, "a fox", lambda r: "a fox", system=system)
    from gmlx.video import run

    with pytest.raises(UpsampleError, match="gmlx video pull h3` fetches it"):
        run.upsample_system(hf.FAMILY, None, frames=124)
