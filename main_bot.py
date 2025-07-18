import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.filters.command import Command
from aiogram.types import InlineQuery, InlineQueryResultArticle, InputTextMessageContent, Message

from config import BOT_TOKEN
from utils.quotes_search import QuotesSearch

logging.basicConfig(level=logging.INFO)

searcher = QuotesSearch()

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


@dp.message(Command("start"))
async def cmd_start(message: Message) -> None:
    await message.answer("Hi, I'm a bot that you can use to search for quotes via inline mode")


@dp.inline_query()
async def show_autocomplete(inline_query: InlineQuery) -> None:
    search_results = searcher.search(inline_query.query)
    inline_results = []
    for quote, similarity in search_results:
        full_text = quote.quote
        author = quote.author
        truncated_text = full_text[:150] + "..." if len(full_text) > 150 else full_text
        inline_results.append(InlineQueryResultArticle(
            id=str(quote.id),
            title=f"{author} ({similarity * 100:.2f}%)",
            description=truncated_text,
            input_message_content=InputTextMessageContent(
                message_text=f'"{full_text}" - {author}'
            )
        ))
    await inline_query.answer(inline_results, cache_time=3600)


async def main() -> None:
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
