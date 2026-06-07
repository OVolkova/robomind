import base64
import io
import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def app():
    from robomind.app import app as flask_app

    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


def _b64(data: bytes = b"RIFF\x00\x00\x00\x00WAVEfmt ") -> str:
    return base64.b64encode(data).decode()


def _mock_processor(text: str = "Hello!", action: str | None = None) -> MagicMock:
    proc = MagicMock()
    proc.process.return_value = (io.BytesIO(b"WAVOUT"), text, action)
    return proc


# ── health check (regression) ────────────────────────────────────────────────


def test_health_check_unchanged(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


# ── input validation ─────────────────────────────────────────────────────────


def test_missing_body_returns_400(client):
    r = client.post("/process", content_type="application/json", data="{}")
    assert r.status_code == 400


def test_missing_audio_field_returns_400(client):
    r = client.post(
        "/process",
        content_type="application/json",
        data=json.dumps({"current_action": "sit"}),
    )
    assert r.status_code == 400


def test_no_content_type_returns_4xx(client):
    r = client.post("/process", data=b"raw bytes")
    assert 400 <= r.status_code < 500


# ── success response shape ────────────────────────────────────────────────────


def test_response_is_wav(client):
    with patch("robomind.app._get_sts_processor", return_value=_mock_processor()):
        r = client.post(
            "/process",
            content_type="application/json",
            data=json.dumps({"audio": _b64()}),
        )
    assert r.status_code == 200
    assert r.content_type == "audio/wav"


def test_response_body_is_wav_bytes(client):
    with patch("robomind.app._get_sts_processor", return_value=_mock_processor()):
        r = client.post(
            "/process",
            content_type="application/json",
            data=json.dumps({"audio": _b64()}),
        )
    assert r.data == b"WAVOUT"


def test_response_text_in_header(client):
    with patch("robomind.app._get_sts_processor", return_value=_mock_processor("Woof!")):
        r = client.post(
            "/process",
            content_type="application/json",
            data=json.dumps({"audio": _b64()}),
        )
    assert r.headers["X-Response-Text"] == "Woof!"


def test_new_action_in_header(client):
    with patch(
        "robomind.app._get_sts_processor", return_value=_mock_processor(action="ksit")
    ):
        r = client.post(
            "/process",
            content_type="application/json",
            data=json.dumps({"audio": _b64()}),
        )
    assert r.headers["X-New-Action"] == "ksit"


def test_new_action_header_absent_when_no_action(client):
    with patch(
        "robomind.app._get_sts_processor", return_value=_mock_processor(action=None)
    ):
        r = client.post(
            "/process",
            content_type="application/json",
            data=json.dumps({"audio": _b64()}),
        )
    assert "X-New-Action" not in r.headers


# ── current_action default ────────────────────────────────────────────────────


def test_current_action_defaults_to_balance(client):
    proc = _mock_processor()
    with patch("robomind.app._get_sts_processor", return_value=proc):
        client.post(
            "/process",
            content_type="application/json",
            data=json.dumps({"audio": _b64()}),
        )
    _, call_action = proc.process.call_args[0][0], proc.process.call_args[0][1]
    assert call_action == "balance"


def test_current_action_forwarded(client):
    proc = _mock_processor()
    with patch("robomind.app._get_sts_processor", return_value=proc):
        client.post(
            "/process",
            content_type="application/json",
            data=json.dumps({"audio": _b64(), "current_action": "trF"}),
        )
    _, call_action = proc.process.call_args[0][0], proc.process.call_args[0][1]
    assert call_action == "trF"


# ── error handling ────────────────────────────────────────────────────────────


def test_processor_error_returns_500(client):
    proc = MagicMock()
    proc.process.side_effect = RuntimeError("model crashed")
    with patch("robomind.app._get_sts_processor", return_value=proc):
        r = client.post(
            "/process",
            content_type="application/json",
            data=json.dumps({"audio": _b64()}),
        )
    assert r.status_code == 500
    assert "error" in r.get_json()
