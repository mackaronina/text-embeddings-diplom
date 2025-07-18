from typing import Any, Callable

from sqlalchemy import Integer, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

from config import DB_URL

engine = create_engine(DB_URL)


def connection(method: Callable) -> Callable:
    def wrapper(*args, **kwargs) -> Any:
        with Session(engine) as session:
            try:
                return method(*args, session=session, **kwargs)
            except Exception as e:
                session.rollback()
                raise e

    return wrapper


class Base(DeclarativeBase):
    pass


class Quote(Base):
    __tablename__ = "quotes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    quote: Mapped[str]
    author: Mapped[str]
