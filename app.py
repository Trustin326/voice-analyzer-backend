from flask import Flask, request, jsonify
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tempfile
import os

app = Flask(__name__)
encoder = VoiceEncoder()

@app.route("/analyze", methods=["POST"])
def analyze():
    fileA = request.files["fileA"]
    fileB = request.files["fileB"]

    with tempfile.NamedTemporaryFile(delete=False) as tmpA:
        fileA.save(tmpA.name)
        wavA = preprocess_wav(tmpA.name)

    with tempfile.NamedTemporaryFile(delete=False) as tmpB:
        fileB.save(tmpB.name)
        wavB = preprocess_wav(tmpB.name)

    embedA = encoder.embed_utterance(wavA)
    embedB = encoder.embed_utterance(wavB)

    similarity = float(cosine_similarity([embedA], [embedB])[0][0])

    return jsonify({
        "similarity_score": similarity,
        "confidence": "High" if similarity > 0.75 else "Low"
    })

if __name__ == "__main__":
    app.run()
