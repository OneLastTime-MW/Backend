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
from scipy.spatial.distance import cosine
from collections import defaultdict
import struct
import librosa
import torch
from resemblyzer import VoiceEncoder, preprocess_wav
import queue
import asyncio
import threading

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

# Initialize the voice encoder for speaker embeddings (loads lazily)
voice_encoder = None
voice_encoder_lock = threading.Lock()

def get_voice_encoder():
    global voice_encoder
    with voice_encoder_lock:
        if voice_encoder is None:
            # This runs on first use only
            print("Loading voice encoder model...")
            voice_encoder = VoiceEncoder()
    return voice_encoder

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

class EmbeddingDiarizationManager:
    """Diarization system based on speaker embeddings."""
    
    def __init__(self, similarity_threshold=0.75):
        self.speaker_profiles = []  # List of speaker profiles [(embedding, speaker_id)]
        self.next_speaker_id = 1
        self.similarity_threshold = similarity_threshold
        self.embedding_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self._process_embeddings, daemon=True)
        self.processing_thread.start()
        
    def _process_embeddings(self):
        """Background thread to process audio and extract embeddings."""
        while True:
            try:
                audio_data, callback = self.embedding_queue.get()
                if audio_data is None:  # Shutdown signal
                    break
                    
                # Generate embedding
                encoder = get_voice_encoder()
                
                # Convert int16 PCM to float32 in range [-1, 1]
                float_audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Ensure audio is long enough (at least 1 second)
                if len(float_audio) < 16000:
                    # Pad with zeros if too short
                    float_audio = np.pad(float_audio, (0, 16000 - len(float_audio)))
                
                # Preprocess for the encoder
                preprocessed_wav = preprocess_wav(float_audio, source_sr=16000)
                
                # Get embedding
                embedding = encoder.embed_utterance(preprocessed_wav)
                
                # Execute callback with the embedding
                callback(embedding)
                
            except Exception as e:
                print(f"Error in embedding processing: {e}")
            finally:
                self.embedding_queue.task_done()
    
    def queue_audio_for_embedding(self, audio_data, callback):
        """Queue audio data for embedding extraction in background thread."""
        self.embedding_queue.put((audio_data, callback))
    
    def get_speaker_for_embedding(self, embedding):
        """Match an embedding to an existing speaker or create a new one."""
        if embedding is None:
            return {"id": 0, "name": "Unknown"}
        
        best_match = None
        max_similarity = -1
        
        # Check against all existing speakers
        for speaker_profile in self.speaker_profiles:
            profile_embedding = speaker_profile[0]
            # Use cosine similarity (higher is better)
            similarity = 1 - cosine(embedding, profile_embedding)
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = speaker_profile[1]
        
        # If we have a close enough match, use that speaker
        if best_match is not None and max_similarity > self.similarity_threshold:
            speaker_id = best_match
        else:
            # Create a new speaker
            speaker_id = self.next_speaker_id
            self.next_speaker_id += 1
            self.speaker_profiles.append((embedding, speaker_id))
        
        return {"id": speaker_id, "name": f"Speaker {speaker_id}"}

class AudioBuffer:
    """Buffer for collecting audio chunks until enough for speaker embedding."""
    
    def __init__(self, min_size=32000):  # 2 seconds at 16kHz
        self.buffer = bytearray()
        self.min_size = min_size
        self.last_reset = datetime.now()
        
    def add(self, audio_data):
        """Add audio data to the buffer."""
        self.buffer.extend(audio_data)
        
    def is_ready(self):
        """Check if enough audio has been collected."""
        return len(self.buffer) >= self.min_size
    
    def get_and_clear(self):
        """Get the buffer contents and clear it."""
        data = bytes(self.buffer)
        self.buffer = bytearray()
        self.last_reset = datetime.now()
        return data
    
    def should_reset(self, silence_duration=3):
        """Check if buffer should be reset due to silence."""
        seconds_since_reset = (datetime.now() - self.last_reset).total_seconds()
        return seconds_since_reset > silence_duration
    
    def reset(self):
        """Reset the buffer."""
        self.buffer = bytearray()
        self.last_reset = datetime.now()

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
        self.diarization_manager = EmbeddingDiarizationManager()
        
        # Audio buffer for collecting enough data for speaker embeddings
        self.audio_buffer = AudioBuffer()
        
        # Keep track of pending transcriptions waiting for speaker ID
        self.pending_transcriptions = []
        
        # For batching multiple transcriptions
        self.batch_lock = asyncio.Lock()
        
        await self.accept()
        print("✅ WebSocket Connection Established.")

    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data:
            # Process received audio data (raw PCM Int16 format)
            if len(bytes_data) > 0:
                # Add to diarization audio buffer
                self.audio_buffer.add(bytes_data)
                
                # Add to transcription buffer
                self.buffer.extend(bytes_data)
                
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
                            
                            # Process text
                            original_text, has_profanity = check_profanity(text)
                            sentiment = analyze_advanced_sentiment(original_text)
                            timestamp = datetime.utcnow().isoformat() + "Z"
                            
                            # If we have enough audio for an embedding, process it
                            # Otherwise, queue as pending
                            if self.audio_buffer.is_ready():
                                audio_data = self.audio_buffer.get_and_clear()
                                
                                # Create an entry with temporary ID
                                entry = {
                                    "speaker": "Processing...",
                                    "speaker_id": 0,  # Temporary
                                    "text": original_text,
                                    "profanity_detected": has_profanity,
                                    "sentiment": sentiment,
                                    "timestamp": timestamp
                                }
                                
                                # Queue for diarization
                                async def process_with_embedding(entry, audio_data):
                                    def callback(embedding):
                                        asyncio.run_coroutine_threadsafe(
                                            self.update_with_speaker(entry, embedding),
                                            asyncio.get_event_loop()
                                        )
                                    
                                    self.diarization_manager.queue_audio_for_embedding(
                                        audio_data, callback
                                    )
                                
                                # Queue the initial entry to show something immediately
                                asyncio.create_task(self.send(text_data=json.dumps(entry)))
                                # Then process with embedding
                                asyncio.create_task(process_with_embedding(entry, audio_data))
                            else:
                                # Not enough audio yet, use temporary speaker
                                entry = {
                                    "speaker": "Processing...",
                                    "speaker_id": 0,
                                    "text": original_text,
                                    "profanity_detected": has_profanity,
                                    "sentiment": sentiment,
                                    "timestamp": timestamp
                                }
                                await self.send(text_data=json.dumps(entry))
                                self.pending_transcriptions.append(entry)
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
                                
                                # Also check if we should process any pending entries
                                if self.audio_buffer.is_ready() and self.pending_transcriptions:
                                    audio_data = self.audio_buffer.get_and_clear()
                                    
                                    # Use the same embedding for all pending entries
                                    async def process_pending_with_embedding(audio_data):
                                        def callback(embedding):
                                            asyncio.run_coroutine_threadsafe(
                                                self.update_pending_entries(embedding),
                                                asyncio.get_event_loop()
                                            )
                                        
                                        self.diarization_manager.queue_audio_for_embedding(
                                            audio_data, callback
                                        )
                                    
                                    asyncio.create_task(process_pending_with_embedding(audio_data))
                                # Or if we've been silent too long, reset the audio buffer
                                elif self.audio_buffer.should_reset():
                                    self.audio_buffer.reset()
                                    # Clear pending entries with a default speaker
                                    if self.pending_transcriptions:
                                        async def assign_default_speaker():
                                            await self.update_pending_entries(None)
                                        asyncio.create_task(assign_default_speaker())
                
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
                    self.audio_buffer.reset()
                    self.pending_transcriptions = []
                    
                    # No need to reset diarization_manager - keep speaker profiles
                    
                    await self.send(text_data=json.dumps({"status": "reset_complete"}))
            except Exception as e:
                print(f"Error processing text command: {e}")
    
    async def update_with_speaker(self, entry, embedding):
        """Update an entry with speaker information based on embedding."""
        speaker = self.diarization_manager.get_speaker_for_embedding(embedding)
        
        # Update the entry
        entry["speaker"] = speaker["name"]
        entry["speaker_id"] = speaker["id"]
        
        # Send the updated entry
        await self.send(text_data=json.dumps({
            "update": True,
            "speaker": speaker["name"],
            "speaker_id": speaker["id"],
            "text": entry["text"],
            "profanity_detected": entry["profanity_detected"],
            "sentiment": entry["sentiment"],
            "timestamp": entry["timestamp"]
        }))
        
        print(f"📝 {speaker['name']}: {entry['text']} | Sentiment: {entry['sentiment']}")
    
    async def update_pending_entries(self, embedding):
        """Update all pending entries with the same speaker ID."""
        async with self.batch_lock:
            if not self.pending_transcriptions:
                return
                
            speaker = self.diarization_manager.get_speaker_for_embedding(embedding)
            
            for entry in self.pending_transcriptions:
                # Update and send
                entry["speaker"] = speaker["name"]
                entry["speaker_id"] = speaker["id"]
                
                await self.send(text_data=json.dumps({
                    "update": True,
                    "speaker": speaker["name"],
                    "speaker_id": speaker["id"],
                    "text": entry["text"],
                    "profanity_detected": entry["profanity_detected"],
                    "sentiment": entry["sentiment"],
                    "timestamp": entry["timestamp"]
                }))
            
            # Clear the list
            self.pending_transcriptions = []

    async def disconnect(self, close_code):
        print(f"❌ WebSocket Disconnected: {close_code}")