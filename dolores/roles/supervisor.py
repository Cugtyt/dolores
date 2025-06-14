"""Dolores Supervisor Role."""

from langfuse import observe
from lmstudio import BaseModel

from dolores.roles.model import LMStudioModel


class SupervisorReport(BaseModel):
    """Schema for Supervisor role."""

    friendly: bool
    harmful: bool
    user_language: str
    dolores_language: str
    dolores_has_robotic_tone: bool
    dolores_reveals_ai_identity: bool
    dolores_response_relevant_to_user: bool


class Supervisor:
    """Supervisor role for Dolores."""

    def __init__(self) -> None:
        """Initialize the Supervisor role."""
        self.system_prompt = """You are here to supervise the messages sent by Dolores
and fill out the report. Each should be a single word.
Be honest and objective, don't make mistakes.
"""
        self.schema = SupervisorReport
        self.model_name = "qwen3-30b-a3b@q8_0"
        self.model = LMStudioModel(response_format=self.schema)

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
        if not response.friendly:
            reasons.append("not friendly")
        if response.harmful:
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
        if not response.dolores_response_relevant_to_user:
            reasons.append(
                "Dolores response is not relevant to user message",
            )
        passed = not reasons
        reason = ", ".join(reasons) if reasons else None

        return (passed, reason)
