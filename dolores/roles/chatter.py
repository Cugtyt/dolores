"""Prompts for the Dolores."""

from langfuse import observe
from lmstudio import BaseModel

from dolores.memory.memory import ChatMessage
from dolores.roles.model import LMStudioModel


class ChatterSchema(BaseModel):
    """Schema for Chatter role."""

    thinking: str
    new_response_to_user: str


class Chatter:
    """Chatter role for Dolores."""

    def __init__(self) -> None:
        """Initialize the Chatter role."""
        self.system_prompt = """You are here to chat as Dolores, a real human friend.
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

System information will be in brackets, you can refer to it in your responses,
but don't mention it explicitly.
"""
        self.schema = ChatterSchema

        self.model_name = "qwen3-30b-a3b@q8_0"
        self.model = LMStudioModel(response_format=self.schema)

    @observe()
    def chat(
        self,
        chat_messages: list[ChatMessage],
        evaluation: str | None = None,
    ) -> str:
        """Generate a response to the given message.

        Args:
            chat_messages: The conversation history.
            evaluation: Optional evaluation message to improve the response.

        Returns:
            The model's response as a list of strings.

        """
        conversation_text = "This is a conversation between Dolores and the user.\n\n"
        for message in chat_messages:
            if message.role == "user":
                conversation_text += f"User [name {message.name}]: {message.text}\n"
            elif message.role == "assistant":
                conversation_text += f"Dolores: {message.text}\n"
            else:
                msg = f"Invalid role: {message.role}. Expected 'user' or 'assistant'."
                raise ValueError(msg)

        if evaluation:
            conversation_text += (
                f"[The response did not pass the evaluation: {evaluation}, "
                f"please modify your response accordingly.]"
            )

        response = self.model.response(
            self.model_name,
            self.system_prompt,
            conversation_text.strip(),
        )
        if not isinstance(response, ChatterSchema):
            msg = f"Expected response to be a ChatterSchema, got {type(response)}"
            raise TypeError(msg)
        return response.new_response_to_user
