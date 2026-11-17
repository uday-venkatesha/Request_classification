import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()

database_url = os.environ["DATABASE_URL"]

# Hosted databases such as Neon hand out URLs that start with "postgresql://".
# SQLAlchemy needs to be told which driver to use, so we add it here.
if database_url.startswith("postgresql://"):
    database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)

# pool_pre_ping checks a connection before using it, because hosted databases close idle ones
engine = create_engine(database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass
