from unittest.mock import MagicMock, patch

from robomind.v2_process import V2Processor

_DUMMY_WAV = b"RIFF\x00\x00\x00\x00WAVEfmt "


def _make_processor(
    llm_text: str = "Hello!", llm_action: str | None = None
) -> V2Processor:
    """Return a V2Processor with all heavy deps replaced by lightweight mocks."""
    proc = V2Processor.__new__(V2Processor)
    proc.speech_to_text = MagicMock(return_value="sit down please")
    proc.text_to_speech = MagicMock()
    proc.text_to_speech.generate.return_value = (MagicMock(), 16000)
    proc.llm = MagicMock()
    proc.llm.model = "gpt-5.4-nano"
    proc.llm.complete.return_value = (llm_text, llm_action)
    proc.history = []
    return proc


def _patch_io():
    """Context manager that patches torchaudio and soundfile in v2_process."""
    torchaudio_patch = patch("robomind.v2_process.torchaudio")
    sf_patch = patch("robomind.v2_process.sf")
    return torchaudio_patch, sf_patch


def _run(proc: V2Processor, current_action: str = "balance"):
    with (
        patch("robomind.v2_process.torchaudio") as mock_ta,
        patch("robomind.v2_process.sf") as mock_sf,
    ):
        mock_ta.load.return_value = (MagicMock(), 16000)
        mock_sf.write.side_effect = lambda buf, *a, **kw: buf.write(b"WAVDATA")
        return proc.process(_DUMMY_WAV, current_action=current_action)


# ── return shape ─────────────────────────────────────────────────────────────


def test_process_returns_three_tuple():
    result = _run(_make_processor())
    assert len(result) == 3


def test_process_audio_is_bytesio():
    import io as _io

    out_buf, _, _ = _run(_make_processor())
    assert isinstance(out_buf, _io.BytesIO)
    assert len(out_buf.read()) > 0


def test_process_response_text():
    _, text, _ = _run(_make_processor(llm_text="Woof!"))
    assert text == "Woof!"


def test_process_action_returned():
    _, _, action = _run(_make_processor(llm_action="sit"))
    assert action == "ksit"


def test_process_no_action_is_none():
    _, _, action = _run(_make_processor(llm_action=None))
    assert action is None


# ── current_action forwarded to LLM ──────────────────────────────────────────


def test_current_action_appears_in_user_message():
    proc = _make_processor()
    _run(proc, current_action="trF")
    messages = proc.llm.complete.call_args[0][0]
    user_msg = next(m for m in messages if m["role"] == "user")
    assert "trF" in user_msg["content"]


def test_system_message_is_first():
    proc = _make_processor()
    _run(proc)
    messages = proc.llm.complete.call_args[0][0]
    assert messages[0]["role"] == "system"


# ── conversation history ──────────────────────────────────────────────────────


def test_history_empty_before_first_call():
    proc = _make_processor()
    assert len(proc.history) == 0


def test_history_has_two_entries_after_one_call():
    proc = _make_processor()
    _run(proc)
    assert len(proc.history) == 2


def test_history_roles_alternate():
    proc = _make_processor()
    _run(proc)
    roles = [m["role"] for m in proc.history]
    assert roles == ["user", "assistant"]


def test_history_summarised_when_full():
    # After 5 calls (10 messages) summarisation fires: 1 summary + 4 recent = 5
    proc = _make_processor()
    for _ in range(5):
        _run(proc)
    from robomind.v2_process import _KEEP_RECENT

    assert len(proc.history) == _KEEP_RECENT + 1


def test_summary_message_is_system_role():
    proc = _make_processor()
    for _ in range(5):
        _run(proc)
    assert proc.history[0]["role"] == "system"
    assert "Summary" in proc.history[0]["content"]


def test_summary_calls_llm_with_old_messages():
    proc = _make_processor()
    for _ in range(5):
        _run(proc)
    # The last llm.complete call is the summarisation call (no tools arg)
    last_call_args, last_call_kwargs = proc.llm.complete.call_args
    assert "tools" not in last_call_kwargs


def test_history_stays_bounded_across_multiple_summarisations():
    proc = _make_processor()
    for _ in range(12):  # enough to trigger summarisation twice
        _run(proc)
    from robomind.v2_process import _MAX_HISTORY

    assert len(proc.history) < _MAX_HISTORY


def test_history_fed_into_subsequent_call():
    proc = _make_processor()
    _run(proc)
    _run(proc)
    messages = proc.llm.complete.call_args[0][0]
    # system + 2 history entries + new user message = 4
    assert len(messages) == 4


# ── fallback spoken text ──────────────────────────────────────────────────────


def test_fallback_spoken_text_when_llm_returns_empty():
    proc = _make_processor(llm_text="")
    _run(proc)
    proc.text_to_speech.generate.assert_called_once_with("Okay.")


def test_no_fallback_when_llm_returns_whitespace_only():
    proc = _make_processor(llm_text="   ")
    _run(proc)
    proc.text_to_speech.generate.assert_called_once_with("Okay.")


def test_no_fallback_when_llm_has_text():
    proc = _make_processor(llm_text="Good boy!")
    _run(proc)
    proc.text_to_speech.generate.assert_called_once_with("Good boy!")
