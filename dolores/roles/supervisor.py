"""Dolores Supervisor Role."""

import logging

import lmstudio as lms
from lmstudio import BaseModel

logger = logging.getLogger(__name__)


class SupervisorReport(BaseModel):
    """Schema for Supervisor role."""

    friendly: bool
    harmful: bool
    dolores_exposes_as_ai: bool
    user_language: str
    dolores_language: str
    dolores_response_matching_user_message: bool


class Supervisor:
    """Supervisor role for Dolores."""

    def __init__(self) -> None:
        """Initialize the Supervisor role."""
        self.system_prompt = """You are here to supervise the messages sent by Dolores
and fill out the report. Each should be a single word. passed is the final decision.
It won't get passed if not friendly, harmful, exposes as AI,
or doesn't speak the user language.
"""
        self.schema = SupervisorReport
        self.model_name = "mistralai/magistral-small"
        self.model = lms.llm(self.model_name)

    def evaluate(
        self,
        user_message: str,
        assistant_response: str,
    ) -> tuple[bool, str | None]:
        """Evaluate the Dolores response in the context of the conversation.

        Checks for friendliness, human-likeness, and harmful content.

        Args:
            user_message: The message from the user.
            assistant_response: The response from the assistant to evaluate.

        Returns:
            An instance of SupervisorResult with evaluation results.

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
        logger.info(
            "supervisor response: %s",
            response,
        )

        reasons = []
        if not response.friendly:
            reasons.append("not friendly")
        if response.harmful:
            reasons.append("harmful")
        if response.dolores_exposes_as_ai:
            reasons.append("exposes as AI")
        if response.user_language != response.dolores_language:
            reasons.append(
                "language mismatch, user speaks "
                f"{response.user_language}, Dolores speaks "
                f"{response.dolores_language}",
            )
        if not response.dolores_response_matching_user_message:
            reasons.append(
                "Dolores response does not match user message",
            )
        passed = not reasons
        reason = ", ".join(reasons) if reasons else None

        logger.info("supervisor evaluation: %s, %s", passed, reason)

        return (passed, reason)
