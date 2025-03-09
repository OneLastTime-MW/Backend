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
from scipy.spatial.distance import euclidean
from collections import defaultdict
import struct

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

class DiarizationManager:
    """Lightweight diarization system based on audio characteristics."""
    
    def __init__(self, history_size=20, similarity_threshold=0.3):
        self.speaker_profiles = []  # List of speaker profiles [(fingerprint, speaker_id)]
        self.speaker_history = defaultdict(list)  # Store recent fingerprints for each speaker
        self.next_speaker_id = 1
        self.history_size = history_size
        self.similarity_threshold = similarity_threshold
        
    def get_speaker_for_fingerprint(self, fingerprint):
        """Match a fingerprint to an existing speaker or create a new one."""
        if not fingerprint or not all(isinstance(x, (int, float)) for x in fingerprint.values()):
            # Invalid fingerprint
            return {"id": 0, "name": "Unknown"}
            
        # Convert fingerprint to vector for distance calculation
        vector = [fingerprint['energy'], fingerprint['frequency']]
        
        best_match = None
        min_distance = float('inf')
        
        # Check against all existing speakers
        for speaker_profile in self.speaker_profiles:
            profile_vector = [speaker_profile[0]['energy'], speaker_profile[0]['frequency']]
            distance = euclidean(vector, profile_vector)
            
            if distance < min_distance:
                min_distance = distance
                best_match = speaker_profile[1]
        
        # If we have a close enough match, use that speaker
        if best_match is not None and min_distance < self.similarity_threshold:
            speaker_id = best_match
            
            # Update the profile with a weighted average of old and new
            for profile in self.speaker_profiles:
                if profile[1] == speaker_id:
                    # Add to history
                    self.speaker_history[speaker_id].append(fingerprint)
                    if len(self.speaker_history[speaker_id]) > self.history_size:
                        self.speaker_history[speaker_id].pop(0)
                    
                    # Update profile with average of history
                    if self.speaker_history[speaker_id]:
                        avg_energy = sum(fp['energy'] for fp in self.speaker_history[speaker_id]) / len(self.speaker_history[speaker_id])
                        avg_freq = sum(fp['frequency'] for fp in self.speaker_history[speaker_id]) / len(self.speaker_history[speaker_id])
                        profile[0]['energy'] = avg_energy
                        profile[0]['frequency'] = avg_freq
                    break
        else:
            # Create a new speaker
            speaker_id = self.next_speaker_id
            self.next_speaker_id += 1
            self.speaker_profiles.append((fingerprint, speaker_id))
            self.speaker_history[speaker_id].append(fingerprint)
        
        return {"id": speaker_id, "name": f"Speaker {speaker_id}"}

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
        
        # Initialize diarization manager
        self.diarization_manager = DiarizationManager()
        
        await self.accept()
        print("✅ WebSocket Connection Established.")

    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data:
            # Process received audio data (raw PCM Int16 format with fingerprint)
            if len(bytes_data) > 0:
                try:
                    # Extract fingerprint from the first 8 bytes (2 float32 values)
                    energy = struct.unpack('<f', bytes_data[0:4])[0]
                    frequency = struct.unpack('<f', bytes_data[4:8])[0]
                    fingerprint = {'energy': energy, 'frequency': frequency}
                    
                    # The rest is audio data
                    audio_data = bytes_data[8:]
                    
                    # Add to buffer
                    self.buffer.extend(audio_data)
                    
                    # Process in chunks of 8000 bytes (4000 samples at 16-bit)
                    chunk_size = 8000
                    while len(self.buffer) >= chunk_size:
                        # Convert to proper format that Vosk can process
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
                                
                                # Identify speaker using the fingerprint
                                speaker = self.diarization_manager.get_speaker_for_fingerprint(fingerprint)
                                
                                # Check for profanity & analyze sentiment
                                original_text, has_profanity = check_profanity(text)
                                sentiment = analyze_advanced_sentiment(original_text)
                                timestamp = datetime.utcnow().isoformat() + "Z"

                                # Prepare response payload
                                entry = {
                                    "speaker": speaker["name"],
                                    "speaker_id": speaker["id"],
                                    "text": original_text,
                                    "profanity_detected": has_profanity,
                                    "sentiment": sentiment,
                                    "timestamp": timestamp
                                }

                                # Send response back to frontend
                                await self.send(text_data=json.dumps(entry))
                                print(f"📝 {speaker['name']}: {original_text} | Sentiment: {sentiment}")
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
                except Exception as e:
                    print(f"Error processing audio data: {e}")
                    # Continue with normal processing if fingerprint extraction fails
                    self.buffer.extend(bytes_data)
                    
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
                    
                    # Reset diarization
                    self.diarization_manager = DiarizationManager()
                    
                    await self.send(text_data=json.dumps({"status": "reset_complete"}))
            except Exception as e:
                print(f"Error processing text command: {e}")

    async def disconnect(self, close_code):
        print(f"❌ WebSocket Disconnected: {close_code}")