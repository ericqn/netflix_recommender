import os
import torch

from sqlalchemy import create_engine, Column, Integer, String, MetaData, Engine, text
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from pgvector.sqlalchemy import Vector

from dotenv import load_dotenv

load_dotenv()
USERNAME = os.getenv('POSTGRES_USER')
DB_NAME = os.getenv('DB_NAME')
PORT = 5432

engine = create_engine(f'postgresql+psycopg2://{USERNAME}@localhost:{PORT}/{DB_NAME}', echo=True)
# engine = create_engine(f'sqlite+pysqlite:///:memory:', echo=True)
Base = declarative_base()

EMBD_DIM = 384

class MovieInfo(Base):
    __tablename__ = 'movie_info'

    id = Column(Integer, primary_key=True)
    movie_title = Column(String, unique=True, nullable=False)
    description = Column(String, nullable=True)
    director = Column(String, nullable=True)
    cast = Column(String, nullable=True)
    genres = Column(String, nullable=True) #listed_in
    country_origin = Column(String, nullable=True)
    date_added = Column(String, nullable=True)
    release_year = Column(String, nullable=True)
    rating = Column(String, nullable=True)
    duration = Column(String, nullable=True)
    show_type = Column(String, nullable=True) # classifies as movie or tv show


class TitleDatabase(Base):
    __tablename__ = 'title_embds'

    id = Column(Integer, primary_key=True)
    movie_title = Column(String, unique=True, nullable=False)
    title_embedding = Column(Vector(EMBD_DIM))

class MetadataDatabase(Base):
    __tablename__ = 'metadata_embds'

    id = Column(Integer, primary_key=True)
    movie_title = Column(String, unique=True, nullable=False)
    movie_metadata = Column(String, nullable=False)
    metadata_embedding = Column(Vector(EMBD_DIM))

Base.metadata.create_all(engine)

SessionLocal: Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def clear_title_db(db: Session):
    delete_rows = 0
    try:
        delete_rows = (
            db.query(TitleDatabase)
            .delete()
        )
        db.commit()

        delete_metadata_rows = (
            db.query(MetadataDatabase)
            .delete()
        )
    except:
        db.rollback()
    return delete_rows, delete_metadata_rows

if __name__ == 'test':
    with SessionLocal() as db:
        result = clear_title_db(db)
        print(f'\n\n\n\nRESULT = {result}\n\n\n\n')