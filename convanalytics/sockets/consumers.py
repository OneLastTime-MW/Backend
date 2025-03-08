import os
import json
import time
import pyaudio
import numpy as np
import nltk
import threading
from datetime import datetime
from vosk import Model, KaldiRecognizer
from nltk.sentiment import SentimentIntensityAnalyzer
from better_profanity import profanity

from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import async_to_sync

# Download required NLTK data (only once)
nltk.download("vader_lexicon")

# Initialize VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Load Vosk model from project folder
model_path = "vosk-model-small-en-us-0.15"
if not os.path.exists(model_path):
    print(f"❌ Error: Vosk model not found at '{model_path}'.")
    exit(1)
model = Model(model_path)
rec = KaldiRecognizer(model, 16000)
rec.SetWords(True)

# Load profanity filter
profanity.load_censor_words()

# Initialize PyAudio for live recording
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=4000
)

# Speaker tracking variables
energy_thresholds = []
energy_window_size = 5
last_speaker = None
speaker_count = 1

def check_profanity(text):
    """Detect but do NOT censor profanity. Returns original text and a flag."""
    contains_profanity = profanity.contains_profanity(text)
    return text, contains_profanity

def analyze_advanced_sentiment(text):
    """Analyze sentiment and detect emotional tone."""
    scores = sia.polarity_scores(text)
    compound = scores["compound"]
    if scores["pos"] > 0.3 and compound < 0:
        sentiment = "Sarcasm"
    elif compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.6:
        sentiment = "Anger"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment

def identify_speaker(audio_chunk):
    """Improved speaker detection using rolling energy averages."""
    global last_speaker, speaker_count, energy_thresholds
    audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
    energy = np.sum(np.abs(audio_data))
    energy_thresholds.append(energy)
    if len(energy_thresholds) > energy_window_size:
        energy_thresholds.pop(0)
    avg_energy = np.mean(energy_thresholds)
    if last_speaker is None or energy > avg_energy * 1.5:
        last_speaker = f"Speaker {speaker_count}"
        speaker_count = 1 if speaker_count == 2 else 2
    return last_speaker

class STTConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.running = True
        # Start the transcription pipeline in a background thread
        self.transcription_thread = threading.Thread(target=self.run_transcription, daemon=True)
        self.transcription_thread.start()

    def run_transcription(self):
        while self.running:
            try:
                data = stream.read(4000, exception_on_overflow=False)
            except IOError as e:
                print("⚠ Audio Overflow Error:", e)
                continue

            speaker = identify_speaker(data)

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text:
                    original_text, has_profanity = check_profanity(text)
                    sentiment = analyze_advanced_sentiment(original_text)
                    timestamp = datetime.utcnow().isoformat() + "Z"
                    entry = {
                        "speaker": speaker,
                        "text": original_text,
                        "profanity_detected": has_profanity,
                        "sentiment": sentiment,
                        "timestamp": timestamp
                    }
                    # Send the transcription result to the WebSocket client
                    async_to_sync(self.send)(text_data=json.dumps(entry))
                    print(f"{speaker}: {original_text}")
            else:
                partial = json.loads(rec.PartialResult())
                text = partial.get("partial", "")
                if len(text.split()) > 3:
                    # Optionally, send partial results too if desired
                    pass

            time.sleep(0.1)

    async def disconnect(self, close_code):
        self.running = False
        # Cleanup resources when disconnecting
        stream.stop_stream()
        stream.close()
        p.terminate()
