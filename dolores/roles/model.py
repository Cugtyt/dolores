"""Dolores local model."""

import lmstudio as lms
from langfuse import observe
from lmstudio import BaseModel


class LMStudioModel:
    """Base class for LM Studio models."""

    def __init__(self, response_format: type[BaseModel]) -> None:
        """Initialize the LM Studio model."""
        self.response_format = response_format

    @observe()
    def response(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
    ) -> BaseModel:
        """Generate a response based on the conversation history.

        Args:
            model_name: The name of the model to use.
            system_prompt: The system prompt to set the context.
            user_prompt: The user's message to respond to.

        Returns:
            The model's response as an instance of the specified response format.

        """
        chat = lms.Chat()
        chat.add_system_prompt(system_prompt)
        chat.add_user_message(user_prompt)

        model = lms.llm(model_name)
        response = model.respond(
            chat,
            response_format=self.response_format,
        ).parsed
        if not isinstance(response, dict):
            msg = f"Expected response to be a dict, got {type(response)}"
            raise TypeError(msg)

        return self.response_format(**response)
