"""Dolores ollama model."""

from langfuse import observe
from ollama import chat
from pydantic import BaseModel


class OllamaModel:
    """Base class for Ollama models."""

    def __init__(self, response_format: type[BaseModel]) -> None:
        """Initialize the Ollama model."""
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
        response = chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            format=self.response_format.model_json_schema(),
        )
        if not isinstance(response.message.content, str):
            msg = (
                f"Expected response to be a string, got {type(response.message.content)}"
            )
            raise TypeError(msg)

        return self.response_format.model_validate_json(response.message.content)
