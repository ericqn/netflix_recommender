import os
import torch

from sqlalchemy import create_engine, Column, Integer, String, MetaData, Engine, text
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from pgvector.sqlalchemy import Vector

from dotenv import load_dotenv

load_dotenv()
USERNAME = os.getenv('POSTGRES_USER')
DB_NAME = 'netflix_recommender'
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
    genres = Column(String, nullable=True)

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

if __name__ == '__main__':
    with SessionLocal() as db:
        result = clear_title_db(db)
        print(f'\n\n\n\nRESULT = {result}\n\n\n\n')

if __name__ == 'dis da noobie':
    with SessionLocal() as session:
        dummy_embd = torch.randn(EMBD_DIM, dtype=torch.float)
        dummy_title = 'this is the best noobie 1'
        test_data = TitleDatabase(
            movie_title = dummy_title,
            title_embedding = dummy_embd
        )

        session.add(test_data)
        session.commit()
        session.refresh(test_data)

        query = (
            session.query(TitleDatabase)
            .filter(TitleDatabase.movie_title == dummy_title)
            .all()
        )

        print('Printing query results...\n')
        for row in query:
            print(f'title = {row.movie_title} | embedding shape = {row.title_embedding.shape}')