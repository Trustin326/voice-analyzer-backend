from flask import Flask, request, jsonify, send_file
from pyannote.audio import Pipeline
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.metrics.pairwise import cosine_similarity
import whisper
import numpy as np
import tempfile
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import datetime

app = Flask(__name__)

# Load models
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
encoder = VoiceEncoder()
whisper_model = whisper.load_model("base")

SIMILARITY_THRESHOLD = 0.75

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
        clip = wav[int(start*16000):int(end*16000)]
        embeds.append(encoder.embed_utterance(clip))
    return np.mean(embeds, axis=0)

def transcribe(path):
    return whisper_model.transcribe(path)

def generate_pdf(results):
    filename = "Forensic_Voice_Report.pdf"
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>FORENSIC VOICE ANALYSIS REPORT</b>", styles['Heading1']))
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph(f"Generated: {datetime.datetime.now()}", styles['Normal']))
    elements.append(Spacer(1, 0.3 * inch))

    for line in results:
        elements.append(Paragraph(line, styles['Normal']))
        elements.append(Spacer(1, 0.2 * inch))

    doc.build(elements)
    return filename

@app.route("/analyze", methods=["POST"])
def analyze():
    fileA = request.files["fileA"]
    fileB = request.files["fileB"]

    tmpA = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmpB = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    fileA.save(tmpA.name)
    fileB.save(tmpB.name)

    speakersA = diarize(tmpA.name)
    speakersB = diarize(tmpB.name)

    embeddingsA = {spk: extract_embedding(tmpA.name, segs) for spk, segs in speakersA.items()}
    embeddingsB = {spk: extract_embedding(tmpB.name, segs) for spk, segs in speakersB.items()}

    match_results = []
    report_lines = []

    for spkA, embA in embeddingsA.items():
        for spkB, embB in embeddingsB.items():
            sim = float(cosine_similarity([embA], [embB])[0][0])
            if sim >= SIMILARITY_THRESHOLD:
                match_results.append((spkA, spkB, sim))
                report_lines.append(
                    f"Speaker {spkA} matches Speaker {spkB} with similarity {round(sim,3)}"
                )

    transcriptA = transcribe(tmpA.name)
    transcriptB = transcribe(tmpB.name)

    report_lines.append("----- FULL TRANSCRIPT FILE A -----")
    report_lines.append(transcriptA["text"])

    report_lines.append("----- FULL TRANSCRIPT FILE B -----")
    report_lines.append(transcriptB["text"])

    pdf_path = generate_pdf(report_lines)

    return jsonify({
        "speaker_count_fileA": len(speakersA),
        "speaker_count_fileB": len(speakersB),
        "matches": match_results,
        "pdf_download": "/download"
    })

@app.route("/download")
def download():
    return send_file("Forensic_Voice_Report.pdf", as_attachment=True)

if __name__ == "__main__":
    app.run()
