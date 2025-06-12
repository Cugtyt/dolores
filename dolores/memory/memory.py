"""Memory module for Dolores bot."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal


@dataclass
class ChatMessage:
    """Represents a message with text, timestamp, and role."""

    text: str
    timestamp: str
    role: Literal["user", "assistant"]


class Memory:
    """Manages the message history for the bot."""

    def __init__(self) -> None:
        """Initialize the Memory with an empty history."""
        self._history: dict[int, list[ChatMessage]] = {}

    def add_message(
        self,
        chat_id: int,
        text: str,
        role: Literal["user", "assistant"],
    ) -> None:
        """Add a message to the history for a given chat_id.

        Args:
            chat_id: The ID of the chat.
            text: The text of the message.
            role: The role of the message sender ("user" or "assistant").

        """
        timestamp = datetime.now(UTC).isoformat()
        if chat_id not in self._history:
            self._history[chat_id] = []
        self._history[chat_id].append(
            ChatMessage(text=text, timestamp=timestamp, role=role),
        )

    def get_messages(self, chat_id: int) -> list[ChatMessage]:
        """Retrieve all messages for a given chat_id.

        Args:
            chat_id: The ID of the chat.

        Returns:
            A list of messages, or an empty list if no messages exist for the chat_id.

        """
        return self._history.get(chat_id, [])
