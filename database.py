from sqlalchemy import Integer, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from config import DB_URL

engine = create_engine(DB_URL)


class Base(DeclarativeBase):
    pass


class Quote(Base):
    __tablename__ = "quotes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    quote: Mapped[str]
    author: Mapped[str]
