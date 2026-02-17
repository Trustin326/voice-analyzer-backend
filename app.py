from flask import Flask, request, jsonify
from flask_cors import CORS
from pyannote.audio import Pipeline
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.metrics.pairwise import cosine_similarity
import whisper
import numpy as np
import tempfile
import threading
import uuid
import time

app = Flask(__name__)
CORS(app)

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
encoder = VoiceEncoder()
whisper_model = whisper.load_model("base")

jobs = {}

def diarize(path):
    diarization = pipeline(path)
    speakers = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.setdefault(speaker, []).append((turn.start, turn.end))
    return speakers

def extract_embedding(path, segments):
    wav = preprocess_wav(path)
    embeds = []
    for start, end in segments:
        clip = wav[int(start * 16000):int(end * 16000)]
        embeds.append(encoder.embed_utterance(clip))
    return np.mean(embeds, axis=0)

@app.route("/start-analysis", methods=["POST"])
def start_analysis():

    fileA = request.files["fileA"]
    fileB = request.files["fileB"]

    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "result": None,
        "created_at": time.time()
    }

    tmpA = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmpB = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    fileA.save(tmpA.name)
    fileB.save(tmpB.name)

    def run_analysis():

        jobs[job_id]["status"] = "processing"

        try:
            jobs[job_id]["progress"] = 10
            speakersA = diarize(tmpA.name)

            jobs[job_id]["progress"] = 30
            speakersB = diarize(tmpB.name)

            jobs[job_id]["progress"] = 60
            transcriptA = whisper_model.transcribe(tmpA.name)

            jobs[job_id]["progress"] = 80
            transcriptB = whisper_model.transcribe(tmpB.name)

            embA = extract_embedding(tmpA.name, list(speakersA.values())[0])
            embB = extract_embedding(tmpB.name, list(speakersB.values())[0])

            similarity = float(
                cosine_similarity([embA], [embB])[0][0]
            )

            jobs[job_id]["result"] = {
                "highest_similarity": similarity,
                "speaker_count_fileA": len(speakersA),
                "speaker_count_fileB": len(speakersB),
                "transcriptA": transcriptA["text"],
                "transcriptB": transcriptB["text"]
            }

            jobs[job_id]["progress"] = 100
            jobs[job_id]["status"] = "complete"

        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["result"] = {"error": str(e)}

    threading.Thread(target=run_analysis).start()

    return jsonify({"job_id": job_id})


@app.route("/job-status/<job_id>")
def job_status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Not found"})
    return jsonify({
        "status": jobs[job_id]["status"],
        "progress": jobs[job_id]["progress"]
    })


@app.route("/job-result/<job_id>")
def job_result(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Not found"})
    return jsonify(jobs[job_id]["result"])


@app.route("/all-jobs")
def all_jobs():
    return jsonify(jobs)


@app.route("/")
def health():
    return jsonify({"status": "VoiceAI Multi-Job Backend Running"})


if __name__ == "__main__":
    app.run()
