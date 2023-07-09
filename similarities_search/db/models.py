from sqlalchemy import Column, ARRAY, Date, Integer, Text, VARCHAR, DateTime
from sqlalchemy.dialects.postgresql import BYTEA, UUID
from sqlalchemy.orm import declarative_base
import uuid

Base = declarative_base()


class TradeNewsRelevant(Base):
    __tablename__ = "trade_news_relevant"

    # id = Column("id", Integer, primary_key=True)
    id = Column("id", UUID(as_uuid=True), primary_key=True)
    classes = Column("classes", ARRAY(Text()))
    itc_codes = Column("itc_codes", Text())
    locations = Column("locations", Text())
    title = Column("title", Text())
    url = Column("url", Text())
    dates = Column("dates", ARRAY(Text()))
    article_ids = Column("article_ids", ARRAY(Text()))
    product = Column("product", Text())
    user_checked = Column("user_checked", Text())
    date_checked = Column("date_checked", Date())
    user_approved = Column("user_approved", Text())
    date_approved = Column("date_approved", Date())


class TradeNewsEmbeddings(Base):
    __tablename__ = "trade_news_embeddings"

    id = Column("id", UUID(as_uuid=True), primary_key=True)
    embedding = Column(BYTEA, nullable=False)
    article_id = Column(VARCHAR, nullable=True)
    model = Column(VARCHAR, nullable=False)
    date_added = Column(DateTime)

class TradeNewsForApproval(Base):
    __tablename__ = "trade_news_for_approval"

    id = Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    classes = Column("classes", ARRAY(Text()))
    itc_codes = Column("itc_codes", Text())
    locations = Column("locations", Text())
    title = Column("title", Text())
    url = Column("url", Text())
    dates = Column("dates", ARRAY(Text()))
    article_ids = Column("article_ids", ARRAY(Text()))
    product = Column("product", Text())
    user_checked = Column("user_checked", Text())
    date_checked = Column("date_checked", Date())
    status = Column("status", Text())
