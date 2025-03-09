import os
import json
import wave
import io
import nltk
import numpy as np
from datetime import datetime
from vosk import Model, KaldiRecognizer
from nltk.sentiment import SentimentIntensityAnalyzer
from better_profanity import profanity
from channels.generic.websocket import AsyncWebsocketConsumer

# Download NLTK VADER (first-time only)
nltk.download("vader_lexicon")

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Load Vosk model for STT
MODEL_PATH = "vosk-model-small-en-us-0.15"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Vosk model not found at '{MODEL_PATH}'. Please ensure it is downloaded correctly.")

model = Model(MODEL_PATH)

# Load Profanity Detector
profanity.load_censor_words()

def check_profanity(text):
    """Detect but do NOT censor profanity. Returns (original text, flag)."""
    return text, profanity.contains_profanity(text)

def analyze_advanced_sentiment(text):
    """Analyze sentiment and detect emotional tone including sarcasm and anger."""
    scores = sia.polarity_scores(text)
    compound = scores["compound"]
    
    if scores["pos"] > 0.3 and compound < 0:
        return "Sarcasm"
    elif compound >= 0.05:
        return "Positive"
    elif compound <= -0.6:
        return "Anger"
    elif compound <= -0.05:
        return "Negative"
    return "Neutral"

class STTConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        # Create a new recognizer for each connection
        self.rec = KaldiRecognizer(model, 16000)
        self.rec.SetWords(True)
        
        # Buffer for audio chunks
        self.buffer = bytearray()
        
        # Accumulated text buffer for partial results
        self.current_text = ""
        self.silent_chunks = 0
        
        await self.accept()
        print("✅ WebSocket Connection Established.")

    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data:
            # Process received audio data (raw PCM Int16 format)
            if len(bytes_data) > 0:
                # Add to buffer
                self.buffer.extend(bytes_data)
                
                # Process in chunks of 8000 bytes (4000 samples at 16-bit)
                chunk_size = 8000
                while len(self.buffer) >= chunk_size:
                    # Convert to proper WAV format that Vosk can process
                    chunk = self.buffer[:chunk_size]
                    self.buffer = self.buffer[chunk_size:]
                    
                    # Process audio with Vosk
                    if self.rec.AcceptWaveform(bytes(chunk)):
                        result = json.loads(self.rec.Result())
                        text = result.get("text", "").strip()
                        
                        # Only process if we have text and it's different from previous
                        if text and text != self.current_text:
                            self.current_text = text
                            self.silent_chunks = 0
                            
                            # Check for profanity & analyze sentiment
                            original_text, has_profanity = check_profanity(text)
                            sentiment = analyze_advanced_sentiment(original_text)
                            timestamp = datetime.utcnow().isoformat() + "Z"

                            # Prepare response payload
                            entry = {
                                "speaker": "Client",
                                "text": original_text,
                                "profanity_detected": has_profanity,
                                "sentiment": sentiment,
                                "timestamp": timestamp
                            }

                            # Send response back to frontend
                            await self.send(text_data=json.dumps(entry))
                            print(f"📝 Transcribed: {original_text} | Sentiment: {sentiment} | Profanity: {has_profanity}")
                    else:
                        # Check partial results
                        partial = json.loads(self.rec.PartialResult())
                        partial_text = partial.get("partial", "").strip()
                        
                        # If no speech detected for a while, reset
                        if not partial_text:
                            self.silent_chunks += 1
                            if self.silent_chunks > 10:  # Reset after ~2 seconds of silence
                                self.current_text = ""
                                self.silent_chunks = 0
        elif text_data:
            # Handle text commands if needed
            try:
                data = json.loads(text_data)
                cmd = data.get("command")
                
                if cmd == "reset":
                    # Reset the recognizer state
                    self.rec = KaldiRecognizer(model, 16000)
                    self.rec.SetWords(True)
                    self.buffer = bytearray()
                    self.current_text = ""
                    self.silent_chunks = 0
                    await self.send(text_data=json.dumps({"status": "reset_complete"}))
            except Exception as e:
                print(f"Error processing text command: {e}")

    async def disconnect(self, close_code):
        print(f"❌ WebSocket Disconnected: {close_code}")