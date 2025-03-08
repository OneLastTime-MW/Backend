from django.urls import path
from .consumers import STTConsumer

websocket_urlpatterns = [
    path("ws/stt/", STTConsumer.as_asgi()),  # WebSocket endpoint
]
