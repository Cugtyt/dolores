"""Dolores Telegram Bot Main Module."""

import logging
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    filters,
)

from dolores.memory.dict_memory import ChatMessage, DictMemory
from dolores.roles.chatter import Chatter
from dolores.roles.supervisor import Supervisor

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

memory = DictMemory()
chatter = Chatter()
supervisor = Supervisor()


async def handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process user message, store it in memory, and generate AI response."""
    if not update.effective_chat:
        logger.error("Error: update.effective_chat is None")
        return
    if not update.message or not update.message.text:
        logger.error("Error: update.message or update.message.text is None")
        return

    chat_id = update.effective_chat.id
    message_text = update.message.text

    username = (
        update.message.from_user.username
        if update.message.from_user and update.message.from_user.username
        else "unknown_user"
    )

    memory.add_message(chat_id, message_text, role="user", name=username)
    logger.info(
        "Stored message for chat_id %s from user %s: %s",
        chat_id,
        username,
        message_text,
    )

    history = memory.get_messages(chat_id) or []

    ai_response = await chatter.chat(history)
    passed, evaluation = await supervisor.evaluate(message_text, ai_response)
    while not passed:
        logger.info(
            "Response did not pass evaluation, retrying with updated response.",
        )

        ai_response = await chatter.chat(
            [
                *history,
                ChatMessage(
                    text=ai_response,
                    timestamp=history[-1].timestamp if history else "",
                    role="assistant",
                    name="Dolores",
                ),
            ],
            evaluation=evaluation,
        )
        passed, evaluation = await supervisor.evaluate(message_text, ai_response)

    memory.add_message(chat_id, ai_response, role="assistant", name="Dolores")
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=ai_response,
    )


if __name__ == "__main__":
    application = (
        ApplicationBuilder().token(os.getenv("TELEGRAM_BOT_TOKEN", "")).build()
    )
    echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handler)

    application.add_handler(echo_handler)

    logger.info("Starting Dolores bot...")
    application.run_polling()
