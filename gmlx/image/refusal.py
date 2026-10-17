"""The error an image primitive raises for a request it refuses."""
from __future__ import annotations


class RequestRefusal(ValueError):
    """A request an image primitive refuses by name, such as a step count,
    a guidance or a reference image its model does not take. The image
    routes answer it with a 400, and any other error a job raises with a
    500 and its trace in the log."""
