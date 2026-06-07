import io as _io
from unittest.mock import MagicMock, patch

from robomind.speech_to_speech import (
    SpeechToSpeechActionProcessor,
    TextToText,
    _KEEP_RECENT,
    _MAX_HISTORY,
)

_DUMMY_WAV = b"RIFF\x00\x00\x00\x00WAVEfmt "


def _make_processor(
    llm_text: str = "Hello!", llm_action: str | None = None
) -> SpeechToSpeechActionProcessor:
    """Return a SpeechToSpeechActionProcessor with all heavy deps replaced by mocks."""
    proc = SpeechToSpeechActionProcessor.__new__(SpeechToSpeechActionProcessor)
    proc.speech_to_text = MagicMock(return_value="sit down please")
    proc.text_to_speech = MagicMock()
    proc.text_to_speech.generate.return_value = (MagicMock(), 16000)
    proc.text_to_text = MagicMock()
    proc.text_to_text.generate.return_value = (llm_text, llm_action)
    return proc


def _run(proc: SpeechToSpeechActionProcessor, current_action: str = "balance"):
    with (
        patch("robomind.speech_to_speech.torchaudio") as mock_ta,
        patch("robomind.speech_to_speech.sf") as mock_sf,
    ):
        mock_ta.load.return_value = (MagicMock(), 16000)
        mock_sf.write.side_effect = lambda buf, *a, **kw: buf.write(b"WAVDATA")
        return proc.process(_DUMMY_WAV, current_action=current_action)


def _make_text_to_text(
    llm_text: str = "Hello!", llm_action: str | None = None
) -> TextToText:
    """Return a TextToText with the LLM replaced by a lightweight mock."""
    tt = TextToText.__new__(TextToText)
    tt.llm = MagicMock()
    tt.llm.complete.return_value = (llm_text, llm_action)
    tt.history = []
    return tt


# ── SpeechToSpeechActionProcessor.process() return shape ─────────────────────


def test_process_returns_three_tuple():
    result = _run(_make_processor())
    assert len(result) == 3


def test_process_audio_is_bytesio():
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


# ── current_action forwarded ──────────────────────────────────────────────────


def test_current_action_forwarded_to_text_to_text():
    proc = _make_processor()
    _run(proc, current_action="trF")
    _, call_action = proc.text_to_text.generate.call_args[0]
    assert call_action == "trF"


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


# ── TextToText: conversation history ─────────────────────────────────────────


def test_history_empty_before_first_call():
    tt = _make_text_to_text()
    assert len(tt.history) == 0


def test_history_has_two_entries_after_one_call():
    tt = _make_text_to_text()
    tt.generate("hello", "balance")
    assert len(tt.history) == 2


def test_history_roles_alternate():
    tt = _make_text_to_text()
    tt.generate("hello", "balance")
    roles = [m["role"] for m in tt.history]
    assert roles == ["user", "assistant"]


def test_history_summarised_when_full():
    tt = _make_text_to_text()
    for _ in range(5):
        tt.generate("hello", "balance")
    assert len(tt.history) == _KEEP_RECENT + 1


def test_summary_message_is_system_role():
    tt = _make_text_to_text()
    for _ in range(5):
        tt.generate("hello", "balance")
    assert tt.history[0]["role"] == "system"
    assert "Summary" in tt.history[0]["content"]


def test_summary_calls_llm_with_old_messages():
    tt = _make_text_to_text()
    for _ in range(5):
        tt.generate("hello", "balance")
    last_call_args, last_call_kwargs = tt.llm.complete.call_args
    assert "tools" not in last_call_kwargs


def test_history_stays_bounded_across_multiple_summarisations():
    tt = _make_text_to_text()
    for _ in range(12):
        tt.generate("hello", "balance")
    assert len(tt.history) < _MAX_HISTORY


def test_history_fed_into_subsequent_call():
    tt = _make_text_to_text()
    tt.generate("hello", "balance")
    tt.generate("world", "sit")
    messages = tt.llm.complete.call_args[0][0]
    # system + 2 history entries + new user = 4
    assert len(messages) == 4


# ── TextToText: LLM message construction ─────────────────────────────────────


def test_current_action_appears_in_user_message():
    tt = _make_text_to_text()
    tt.generate("hello", "trF")
    messages = tt.llm.complete.call_args[0][0]
    user_msg = next(m for m in messages if m["role"] == "user")
    assert "trF" in user_msg["content"]


def test_system_message_is_first():
    tt = _make_text_to_text()
    tt.generate("hello", "balance")
    messages = tt.llm.complete.call_args[0][0]
    assert messages[0]["role"] == "system"
