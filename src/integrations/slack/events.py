from __future__ import annotations
import json
import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger('slack.events')

DEDUPE_TTL_SECONDS = 60 * 60

EventHandler = Callable[[dict[str, Any], dict[str, Any]], None]

class EventDeduplicator:
    def __init__(self, ttl_seconds: int = DEDUPE_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._seen: dict[str, float] = {}

    def is_duplicate(self, event_id: str | None) -> bool:
        if not event_id:
            return False
        
        self._evict_expired()

        if event_id in self._seen:
            return True

        self._seen[event_id] = time.time()
        return False

    def _evict_expired(self) -> None:
        cutoff = time.time() - self._ttl
        for event_id in [e for e, seen in self._seen.items() if seen < cutoff]:
            del self._seen[event_id]

    def clear(self) -> None:
        self._seen.clear()


class EventDispatcher:
    def __init__(self) -> None:
        self._handlers: dict[str, EventHandler] = {}

    def register(self, event_type: str) -> Callable[[EventHandler], EventHandler]:
        def decorator(handler: EventHandler) -> EventHandler:
            self._handlers[event_type] = handler
            return handler

        return decorator

    def dispatch(self, envelope: dict[str, Any]) -> str:
        event = envelope.get('event') or {}
        event_type = event.get('type')

        if not event_type:
            logger.warning('Received event_callback with no event.type')
            return 'malformed'

        handler = self._handlers.get(event_type)
        if handler is None:
            logger.info(f'No handler registered for event type: {event_type}')
            return 'unhandled'

        handler(event, envelope)
        return 'handled'


deduplicator = EventDeduplicator()
dispatcher = EventDispatcher()

@dispatcher.register('message')
def handle_message(event: dict[str, Any], envelope: dict[str, Any]) -> None:
    """Log an inbound channel message as one structured line."""
    if event.get('bot_id') or event.get('subtype') == 'bot_message':
        logger.debug('Ignoring bot-authored message')
        return

    logger.info(json.dumps(
        {
            'event': 'slack.message_received',
            'team_id': envelope.get('team_id'),
            'channel': event.get('channel'),
            'user': event.get('user'),
            'ts': event.get('ts'),
            'text_length': len(event.get('text', ''))
        }
    ))