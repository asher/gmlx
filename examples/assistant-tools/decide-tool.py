# /// script
# dependencies = ["mcp<2"]
# ///
"""A decide tool for the gmlx assistant: asks /v1/systemone and returns the
top answer with its score."""
import json
import os
import urllib.request

from mcp.server.fastmcp import FastMCP

URL = os.environ.get("GMLX_URL", "http://127.0.0.1:8080") + "/v1/systemone"
mcp = FastMCP("decide")


@mcp.tool()
def decide(question: str, options: list[str], context: str = "") -> str:
    """Pick the best of a fixed set of options with the decision model. For a
    yes or no question, pass the options ["yes", "no"]. Put any facts the
    user gave in context."""
    if sorted(o.lower() for o in options) == ["no", "yes"]:
        q = {"type": "noul", "instructions": question}
    else:
        q = {"type": "choice", "instructions": question,
             "criteria": {o: None for o in options}}
    body = {"state": context or question, "questions": {"q": q}}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        a = json.load(r)["answers"]["q"]
    if a["type"] == "noul":
        top, score = ("yes", a["noul"]) if a["noul"] >= 0.5 else ("no", 1 - a["noul"])
    else:
        top, score = a["choice"], a["probabilities"][a["choice"]]
    return f"{top}, score {score:.2f}"


mcp.run()
