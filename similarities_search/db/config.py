import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")

DB_STRING = (
    f"{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)
DB_URI = f"postgresql+psycopg2://{DB_STRING}"
engine = create_engine(DB_URI)

Session = sessionmaker(engine)
