import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Default backend is sqlite
DB_BACKEND = os.getenv("DB_BACKEND", "sqlite")
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:  # fallback if DATABASE_URL is not provided
    if DB_BACKEND == "postgres":
        DATABASE_URL = "postgresql://user:pass@localhost/predictions"
    else:
        DATABASE_URL = "sqlite:///./predictions.db"

print("Using DB:", DATABASE_URL)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    from models import PredictionSession, DetectionObject, User
    print("Initializing DB with models:", Base.metadata.tables.keys())

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(engine)
