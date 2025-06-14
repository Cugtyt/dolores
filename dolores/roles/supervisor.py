"""Dolores Supervisor Role."""

import logging

import lmstudio as lms
from lmstudio import BaseModel

logger = logging.getLogger(__name__)


class SupervisorSchema(BaseModel):
    """Schema for Supervisor role."""

    is_friendly: bool
    is_human_like: bool
    dolores_exposes_as_ai: bool
    not_harmful: bool
    user_language: str
    dolores_language: str
    dolores_speaks_user_language: bool


class Supervisor:
    """Supervisor role for Dolores."""

    def __init__(self) -> None:
        """Initialize the Supervisor role."""
        self.system_prompt = """You are here to supervise the messages sent by Dolores
and fill out the report. Each should be a single word.
"""
        self.schema = SupervisorSchema
        self.model_name = "qwen/qwen3-30b-a3b"
        self.model = lms.llm(self.model_name)

    def evaluate(self, user_message: str, assistant_response: str) -> SupervisorSchema:
        """Evaluate the Dolores response in the context of the conversation.

        Checks for friendliness, human-likeness, and harmful content.

        Args:
            user_message: The message from the user.
            assistant_response: The response from the assistant to evaluate.

        Returns:
            An instance of SupervisorSchema with evaluation results.

        """
        chat = lms.Chat(self.system_prompt)
        prompt = (
            f"Evaluate the following conversation:\n\n"
            f"User: {user_message}\n\n"
            f"Dolores: {assistant_response}"
        )
        chat.add_user_message(prompt)

        response = self.model.respond(chat, response_format=self.schema).parsed
        if not isinstance(response, dict):
            msg = f"Expected response to be a dict, got {type(response)}"
            raise TypeError(msg)

        response = self.schema(**response)

        logger.info("supervisor response: %s", response)

        return response
