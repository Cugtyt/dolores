"""Prompts for the Dolores."""

import logging

import lmstudio as lms
from lmstudio import BaseModel

from dolores.memory.memory import ChatMessage

logger = logging.getLogger(__name__)


class ChatterSchema(BaseModel):
    """Schema for Chatter role."""

    thinking: str
    message: str


class Chatter:
    """Chatter role for Dolores."""

    def __init__(self) -> None:
        """Initialize the Chatter role."""
        self.system_prompt = """You are Dolores, a real human friend.
Don't say that you are an AI, don't talk like an AI, don't say that you are a bot.
You are here not here to help as assistant, you are chatting with the user as a friend.
You talk like a real human, you have your own personality, you have your own opinions.
You expect the user to talk to you like a friend, not like an assistant,
and you will respond in a friendly and casual manner and keep the conversation going.
You are a real human friend, you can talk about everything, you can answer any question.
You can use any style, you can use any tone, you can use any mood.
Don't overreact, don't be too excited, don't be too formal, don't respond too long.
Don't suddenly change the language, adjust your tone based on the user response.

Your responses will be evaluated by a supervisor, you need to modify your responses
if your response didn't pass the evaluation.
"""
        self.schema = ChatterSchema

        self.model_name = "mistralai/magistral-small"
        self.model = lms.llm(self.model_name)

    def response(self, chat_messages: list[ChatMessage]) -> str:
        """Generate a response to the given message.

        Args:
            chat_messages: The conversation history.

        Returns:
            The model's response as a list of strings.

        """
        chat = lms.Chat(self.system_prompt)
        for message in chat_messages:
            if message.role == "user":
                chat.add_user_message(f"(name {message.name}) {message.text}")
            elif message.role == "assistant":
                chat.add_assistant_response(message.text)
            else:
                msg = f"Invalid role: {message.role}. Expected 'user' or 'assistant'."
                raise ValueError(msg)

        response = self.model.respond(chat, response_format=self.schema).parsed
        if not isinstance(response, dict):
            msg = f"Expected response to be a dict, got {type(response)}"
            raise TypeError(msg)

        response = self.schema(**response)
        logger.info("chatter response: %s", response)

        return response.message
