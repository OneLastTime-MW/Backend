import json
from channels.generic.websocket import AsyncWebsocketConsumer

class STTConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()  # Accept WebSocket connection
        await self.send(text_data=json.dumps({"message": "Connected to STT WebSocket!"}))

    async def disconnect(self, close_code):
        print(f"WebSocket disconnected with code: {close_code}")

    async def receive(self, text_data):
        data = json.loads(text_data)
        print("Received data:", data)

        # Simulate STT response (Replace with actual STT model integration)
        transcription = f"Transcribed: {data.get('text', 'No text received')}"
        
        # Send back response
        await self.send(text_data=json.dumps({"transcription": transcription}))
