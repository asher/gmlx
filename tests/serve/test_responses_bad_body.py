"""A Responses body that is not a valid request is a client error."""

import asyncio

import pytest
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request

from gmlx._vendor.mlx_vlm.server import openai


def _request(body: bytes) -> Request:
    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request({"type": "http", "method": "POST", "headers": []}, receive)


@pytest.mark.parametrize("endpoint", [openai.responses_endpoint,
                                      openai.responses_input_tokens_endpoint])
@pytest.mark.parametrize("body, error", [
    (b"", HTTPException), (b"{", HTTPException), (b"[1]", HTTPException),
    (b'{"input": 5}', RequestValidationError),
])
def test_a_bad_body_is_a_client_error(endpoint, body, error):
    with pytest.raises(error) as e:
        asyncio.run(endpoint(_request(body)))
    if error is HTTPException:
        assert e.value.status_code == 400
