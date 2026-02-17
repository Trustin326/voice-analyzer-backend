from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pyannote.audio import Pipeline
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.metrics.pairwise import cosine_similarity
import whisper
import numpy as np
import tempfile
import threading
import uuid
import hashlib
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import datetime

app = Flask(__name__)
CORS(app)

# Load heavy models once
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
encoder = VoiceEncoder()
whisper_model = whisper.load_model("base")

jobs = {}

SIMILARITY_THRESHOLD = 0.75


# ------------------ HELPER FUNCTIONS ------------------

def generate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


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


def generate_pdf(job_id, result):
    filename = f"report_{job_id}.pdf"
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>FORENSIC VOICE ANALYSIS REPORT</b>", styles['Heading1']))
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph(f"Generated: {datetime.datetime.now()}", styles['Normal']))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"Similarity Score: {round(result['highest_similarity'] * 100, 2)}%", styles['Normal']))
    elements.append(Paragraph(f"Speakers File A: {result['speaker_count_fileA']}", styles['Normal']))
    elements.append(Paragraph(f"Speakers File B: {result['speaker_count_fileB']}", styles['Normal']))
    elements.append(Spacer(1, 0.5 * inch))

    elements.append(Paragraph("<b>Transcript File A</b>", styles['Normal']))
    elements.append(Paragraph(result['transcriptA'], styles['Normal']))
    elements.append(Spacer(1, 0.5 * inch))

    elements.append(Paragraph("<b>Transcript File B</b>", styles['Normal']))
    elements.append(Paragraph(result['transcriptB'], styles['Normal']))

    doc.build(elements)
    return filename


# ------------------ BACKGROUND JOB ROUTES ------------------

@app.route("/start-analysis", methods=["POST"])
def start_analysis():

    fileA = request.files["fileA"]
    fileB = request.files["fileB"]

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "processing",
        "progress": 0,
        "result": None
    }

    tmpA = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmpB = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    fileA.save(tmpA.name)
    fileB.save(tmpB.name)

    def run_analysis():
        try:
            jobs[job_id]["progress"] = 10

            speakersA = diarize(tmpA.name)
            jobs[job_id]["progress"] = 30

            speakersB = diarize(tmpB.name)
            jobs[job_id]["progress"] = 50

            transcriptA = whisper_model.transcribe(tmpA.name)
            jobs[job_id]["progress"] = 70

            transcriptB = whisper_model.transcribe(tmpB.name)
            jobs[job_id]["progress"] = 85

            # Simple first speaker comparison
            embA = extract_embedding(tmpA.name, list(speakersA.values())[0])
            embB = extract_embedding(tmpB.name, list(speakersB.values())[0])

            similarity = float(
                cosine_similarity([embA], [embB])[0][0]
            )

            result_data = {
                "highest_similarity": similarity,
                "speaker_count_fileA": len(speakersA),
                "speaker_count_fileB": len(speakersB),
                "transcriptA": transcriptA["text"],
                "transcriptB": transcriptB["text"]
            }

            jobs[job_id]["result"] = result_data
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
        return jsonify({"error": "Job not found"})
    return jsonify({
        "status": jobs[job_id]["status"],
        "progress": jobs[job_id]["progress"]
    })


@app.route("/job-result/<job_id>")
def job_result(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"})
    return jsonify(jobs[job_id]["result"])


@app.route("/download/<job_id>")
def download(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"})

    result = jobs[job_id]["result"]
    pdf_file = generate_pdf(job_id, result)
    return send_file(pdf_file, as_attachment=True)


@app.route("/export-csv/<job_id>")
def export_csv(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"})

    result = jobs[job_id]["result"]

    csv_content = "Metric,Value\n"
    csv_content += f"Similarity,{result['highest_similarity']}\n"
    csv_content += f"Speakers File A,{result['speaker_count_fileA']}\n"
    csv_content += f"Speakers File B,{result['speaker_count_fileB']}\n"

    filename = f"results_{job_id}.csv"
    with open(filename, "w") as f:
        f.write(csv_content)

    return send_file(filename, as_attachment=True)


# ------------------ HEALTH CHECK ------------------

@app.route("/")
def health():
    return jsonify({"status": "VoiceAI Backend Running"})


if __name__ == "__main__":
    app.run()
