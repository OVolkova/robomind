import base64

from flask import Flask, Response, jsonify, request
import logging

from robomind.process import process, random_answer
from robomind.v2_process import V2Processor

CHUNK_SIZE = 2048

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_v2_processor: V2Processor | None = None


def _get_v2_processor() -> V2Processor:
    global _v2_processor
    if _v2_processor is None:
        _v2_processor = V2Processor()
    return _v2_processor


@app.route("/random", methods=["POST"])
def random_wav_stream():
    logger.info("Received random request")
    try:
        processed_wav = random_answer()

        def generate():
            while chunk := processed_wav.read(CHUNK_SIZE):
                yield chunk

        response = Response(generate(), mimetype="audio/wav")
        response.headers["Content-Type"] = "audio/wav"
        return response
    except Exception as e:
        logger.error(f"Error in random endpoint: {e}")
        return {"error": str(e)}, 500


@app.route("/process", methods=["POST"])
def process_wav_stream():
    """Receive audio stream (WAV), process it, and return processed WAV as a stream."""
    logger.info(f"Received process request, Content-Type: {request.content_type}")

    # Read incoming audio stream
    audio_data = request.stream.read()

    if not audio_data:
        logger.error("No audio data received")
        return {"error": "No audio data received"}, 400

    logger.info(f"Received audio data size: {len(audio_data)} bytes")

    try:
        processed_wav = process(audio_data)

        def generate():
            while chunk := processed_wav.read(CHUNK_SIZE):
                yield chunk

        response = Response(generate(), mimetype="audio/wav")
        response.headers["Content-Type"] = "audio/wav"
        return response
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return {"error": str(e)}, 500


@app.route("/v2/process", methods=["POST"])
def v2_process_route():
    """LLM-powered speech processing with robot action output."""
    data = request.get_json()
    if not data or "audio" not in data:
        return {"error": "Missing 'audio' field in JSON body"}, 400

    audio_bytes = base64.b64decode(data["audio"])
    current_action = data.get("current_action", "balance")

    logger.info(
        f"v2/process: current_action={current_action!r}, audio={len(audio_bytes)} bytes"
    )

    try:
        out_audio, response_text, new_action = _get_v2_processor().process(
            audio_bytes, current_action
        )
        return jsonify(
            {
                "audio": base64.b64encode(out_audio).decode(),
                "response_text": response_text,
                "new_action": new_action,
            }
        )
    except Exception as e:
        logger.error(f"Error in v2/process: {e}")
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
    logger.info("  POST /random - Get random AI response (WAV)")
    logger.info("  GET /health - Health check")
    app.run(host="0.0.0.0", port=7777, debug=True, threaded=True)
