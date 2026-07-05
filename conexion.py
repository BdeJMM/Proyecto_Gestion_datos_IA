import os
import sqlalchemy
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

_engine = None

def get_engine():
    global _engine
    if _engine is not None:
        return _engine

    _engine = sqlalchemy.create_engine(
        DATABASE_URL,
        pool_size    = 2,
        max_overflow = 0,
        pool_pre_ping= True,
    )
    return _engine
