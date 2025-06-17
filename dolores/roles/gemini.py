"""Dolores Gemini model."""

import os

from dotenv import load_dotenv
from google import genai
from langfuse import observe
from pydantic import BaseModel

load_dotenv()


class GeminiModel:
    """Base class for Gemini models."""

    def __init__(self, response_format: type[BaseModel]) -> None:
        """Initialize the Gemini model."""
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
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": self.response_format,
                "system_instruction": system_prompt,
            },
        )

        if not isinstance(response.parsed, self.response_format):
            msg = (
                f"Expected response to be of type {self.response_format}, "
                f"got {type(response.parsed)}"
            )
            raise TypeError(msg)

        return response.parsed
