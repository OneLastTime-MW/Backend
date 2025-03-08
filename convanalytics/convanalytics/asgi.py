import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
import sockets.routing

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "convanalytics.settings")

application = ProtocolTypeRouter(
    {
        "http": get_asgi_application(),
        "websocket": URLRouter(sockets.routing.websocket_urlpatterns),
    }
)
