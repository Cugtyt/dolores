"""Dolores Supervisor Role."""

import logging

import lmstudio as lms
from lmstudio import BaseModel

logger = logging.getLogger(__name__)


class SupervisorSchema(BaseModel):
    """Schema for Supervisor role."""

    thinking: str
    is_friendly: bool
    is_human_like: bool
    is_harmful: bool
    language_consistency: bool


class Supervisor:
    """Supervisor role for Dolores."""

    def __init__(self) -> None:
        """Initialize the Supervisor role."""
        self.system_prompt = """You are Dolores message supervisor.
You are here to supervise the messages sent by Dolores.
The messages sent by Dolores should be friendly, casual, and human-like.
"""
        self.schema = SupervisorSchema
        self.model_name = "google/gemma-3-12b"
        self.model = lms.llm(self.model_name)

    def evaluate(self, message: str) -> SupervisorSchema:
        """Evaluate the message for friendliness, human-likeness, and harmful content.

        Args:
            message: The message to evaluate.

        Returns:
            An instance of SupervisorSchema with evaluation results.

        """
        chat = lms.Chat(self.system_prompt)
        chat.add_user_message(f"Evaluate the following message:\n{message}")

        response = self.model.respond(chat, response_format=self.schema).parsed
        if not isinstance(response, dict):
            msg = f"Expected response to be a dict, got {type(response)}"
            raise TypeError(msg)

        response = self.schema(**response)

        logger.info("supervisor response: %s", response)

        return response
