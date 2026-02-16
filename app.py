from flask import Flask, request, jsonify
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.metrics.pairwise import cosine_similarity
import whisper
import numpy as np
import tempfile
import os

app = Flask(__name__)

encoder = VoiceEncoder()
whisper_model = whisper.load_model("base")

SIMILARITY_THRESHOLD = 0.75

def transcribe_audio(path):
    result = whisper_model.transcribe(path)
    return result

@app.route("/analyze", methods=["POST"])
def analyze():
    fileA = request.files["fileA"]
    fileB = request.files["fileB"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpA:
        fileA.save(tmpA.name)
        wavA = preprocess_wav(tmpA.name)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpB:
        fileB.save(tmpB.name)
        wavB = preprocess_wav(tmpB.name)

    # Voice embeddings
    embedA = encoder.embed_utterance(wavA)
    embedB = encoder.embed_utterance(wavB)

    similarity = float(cosine_similarity([embedA], [embedB])[0][0])
    match = similarity >= SIMILARITY_THRESHOLD

    # Full transcription
    transcriptA = transcribe_audio(tmpA.name)
    transcriptB = transcribe_audio(tmpB.name)

    # If match, extract matched speech segments
    matched_speech = []
    if match:
        for segment in transcriptA["segments"]:
            matched_speech.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })

    response = {
        "similarity_score": similarity,
        "match_detected": match,
        "confidence": "High" if match else "Low",
        "fileA_full_transcript": transcriptA["text"],
        "fileB_full_transcript": transcriptB["text"],
        "matched_segments_from_fileA": matched_speech
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run()
