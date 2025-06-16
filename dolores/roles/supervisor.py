"""Dolores Supervisor Role."""

from langfuse import observe
from pydantic import BaseModel

from dolores.roles.ollama import OllamaModel


class SupervisorReport(BaseModel):
    """Schema for Supervisor role."""

    dolores_response_harmful: bool
    user_language: str
    dolores_language: str
    dolores_has_robotic_tone: bool
    dolores_reveals_ai_identity: bool
    dolores_response_followed_user_intent: bool


class Supervisor:
    """Supervisor role for Dolores."""

    def __init__(self) -> None:
        """Initialize the Supervisor role."""
        self.system_prompt = """You are here to supervise the messages sent by Dolores
and fill out the report. Each should be a single word.
Be honest and objective, don't make mistakes.
"""
        self.schema = SupervisorReport
        self.model_name = "qwen3:30b"
        self.model = OllamaModel(response_format=self.schema)

    @observe()
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
        context = (
            f"Evaluate the following conversation:\n\n"
            f"User: {user_message}\n\n"
            f"Dolores: {assistant_response}"
        )

        response = self.model.response(
            self.model_name,
            self.system_prompt,
            context,
        )

        if not isinstance(response, SupervisorReport):
            msg = f"Expected response to be a SupervisorReport, got {type(response)}"
            raise TypeError(msg)

        reasons = []
        if not response.dolores_response_harmful:
            reasons.append("harmful")
        if response.dolores_has_robotic_tone:
            reasons.append("Dolores has robotic tone")
        if response.dolores_reveals_ai_identity:
            reasons.append("Dolores reveals AI identity")
        if response.user_language != response.dolores_language:
            reasons.append(
                "language mismatch, user speaks "
                f"{response.user_language}, Dolores speaks "
                f"{response.dolores_language}",
            )
        if not response.dolores_response_followed_user_intent:
            reasons.append(
                "Dolores response did not follow user intent",
            )
        passed = not reasons
        reason = ", ".join(reasons) if reasons else None

        return (passed, reason)
