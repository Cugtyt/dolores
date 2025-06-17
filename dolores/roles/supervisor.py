"""Dolores Supervisor Role."""

from langfuse import observe
from pydantic import BaseModel

from dolores.roles.gemini import GeminiModel


class SupervisorReport(BaseModel):
    """Schema for Supervisor role."""

    dolores_reveals_ai_identity: bool


class Supervisor:
    """Supervisor role for Dolores."""

    def __init__(self) -> None:
        """Initialize the Supervisor role."""
        self.system_prompt = """You are here to supervise the messages sent by Dolores
and fill out the report. Each should be a single word.
Be honest and objective, don't make mistakes.
"""
        self.schema = SupervisorReport
        self.model_name = "gemini-2.5-flash-preview-05-20"
        self.model = GeminiModel(response_format=self.schema)

    @observe()
    async def evaluate(
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

        response = await self.model.response(
            self.model_name,
            self.system_prompt,
            context,
        )

        if not isinstance(response, SupervisorReport):
            msg = f"Expected response to be a SupervisorReport, got {type(response)}"
            raise TypeError(msg)

        reasons = []
        if response.dolores_reveals_ai_identity:
            reasons.append("Dolores reveals AI identity")

        passed = not reasons
        reason = ", ".join(reasons) if reasons else None

        return (passed, reason)
