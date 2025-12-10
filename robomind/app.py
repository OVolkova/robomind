from flask import Flask, Response, request
import logging

from robomind.process import process, random_answer


app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route("/random", methods=["POST"])
def random_wav_stream():
    logger.info("Received random request")
    try:
        processed_wav = random_answer()

        def generate():
            while chunk := processed_wav.read(4096):
                yield chunk

        return Response(generate(), mimetype="audio/wav")
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
            while chunk := processed_wav.read(4096):
                yield chunk

        return Response(generate(), mimetype="audio/wav")
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return {"error": str(e)}, 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for ESP32 to verify server is running"""
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
