import base64
import io

from flask import Flask, Response, request
import logging

from robomind.speech_to_speech import SpeechToSpeechActionProcessor

CHUNK_SIZE = 2048

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_sts_processor: SpeechToSpeechActionProcessor | None = None


def _get_sts_processor() -> SpeechToSpeechActionProcessor:
    global _sts_processor
    if _sts_processor is None:
        _sts_processor = SpeechToSpeechActionProcessor()
    return _sts_processor


@app.route("/process", methods=["POST"])
def process_route():
    """LLM-powered speech processing with robot action output."""
    data = request.get_json()
    if not data or "audio" not in data:
        return {"error": "Missing 'audio' field in JSON body"}, 400

    audio_bytes = base64.b64decode(data["audio"])
    current_action = data.get("current_action", "balance")

    logger.info(
        f"process: current_action={current_action!r}, audio={len(audio_bytes)} bytes"
    )

    try:
        wav_buf, response_text, new_action = _get_sts_processor().process(
            audio_bytes, current_action
        )

        def generate():
            while chunk := wav_buf.read(CHUNK_SIZE):
                yield chunk

        response = Response(generate(), mimetype="audio/wav")
        response.headers["Content-Type"] = "audio/wav"
        response.headers["X-Response-Text"] = response_text.encode("ascii", errors="replace").decode("ascii")
        if new_action:
            response.headers["X-New-Action"] = new_action
        return response
    except Exception as e:
        logger.error(f"Error in process: {e}")
        return {"error": str(e)}, 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for ESP32 to verify server is running"""
    logger.info("Health check requested")
    return {"status": "ok", "service": "robomind"}, 200


if __name__ == "__main__":
    # Set host='0.0.0.0' to make it accessible to other devices in the network
    logger.info("Starting robomind Flask server on port 7777...")
    logger.info("Audio format: WAV (PCM 16-bit, simplified - no MP3!)")
    logger.info("Endpoints available:")
    logger.info("  POST /process - Process audio and return AI response (WAV)")
    logger.info("  GET /health - Health check")
    app.run(host="0.0.0.0", port=7777, debug=True, threaded=True)
