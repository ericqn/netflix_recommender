from pathlib import Path
import pandas as pd

from sqlalchemy import func, event, select, text
from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

# import models
from models import (
    TitleDatabase,
    MetadataDatabase,
    get_db, 
    engine, 
    SessionLocal  
)
from embedding_utils import (
    create_movie_embeddings, 
    create_title_embeddings, 
    preprocess_into_df, 
    retrieve_metadata_strs,
    create_sentence_embedding
)

def populate_with_title_embds(
        embeddings: torch.Tensor, 
        idx_to_title:dict, 
        db: Session
    ):
    """
    Adds title embeddings into TitleDatabase database table
    """
    for idx, embedding in enumerate(embeddings):
        movie_title = idx_to_title[idx]
        if not get_embd('vector', movie_title, db=db):
            embeddings_entry = TitleDatabase(
                movie_title = movie_title,
                title_embedding = embedding
            )

            db.add(embeddings_entry)
            db.commit()
            db.refresh(embeddings_entry)        
        else:
            # TODO: Handle this error when frontend comes
            print(f'Movie title already found')
    

def populate_with_metadata_embds(
        metadata_strings: dict,
        embeddings: torch.Tensor, 
        idx_to_title:dict, 
        db: Session
    ):
    """
    Adds title embeddings into TitleDatabase database table
    """
    for idx, embedding in enumerate(embeddings):
        movie_title = idx_to_title[idx]
        if not get_embd('metadata', movie_title, db=db):
            movie_metadata = metadata_strings[movie_title]
            embeddings_entry = MetadataDatabase(
                movie_title = movie_title,
                movie_metadata = movie_metadata,
                metadata_embedding = embedding
            )

            db.add(embeddings_entry)
            db.commit()
            db.refresh(embeddings_entry)        
        else:
            # TODO: Handle this error when frontend comes
            print(f'Movie title already found')
    

def get_embd(table_name, movie_title, db: Session=get_db):
    """
    Checks if an entry for movie title exists in the table with (table_name). Used as a helper function for
    populate_with_metadata_embds and populate_with_title_embds.
    """
    if table_name.lower() == 'vector':
        return (
            db.query(TitleDatabase)
            .filter(TitleDatabase.movie_title == movie_title)
            .all()
        )
    elif table_name.lower() == 'metadata':
        return (
            db.query(MetadataDatabase)
            .filter(MetadataDatabase.movie_title == movie_title)
            .all()
        )
    else:
        return None

if __name__ == '__main__':
    file_path = Path('raw_data') / 'netflix_titles_nov_2019.csv'

    df = preprocess_into_df(file_path)
    metadata_strs = retrieve_metadata_strs(df, limit=10)

    embd_model = SentenceTransformer("all-MiniLM-L6-v2")
    title_embds, idx_to_title_0 = create_title_embeddings(metadata_strs, model=embd_model)
    metadata_embds, idx_to_title_metadata = create_movie_embeddings(metadata_strs, model=embd_model)

    with SessionLocal() as session:
        populate_with_title_embds(title_embds, idx_to_title_0, db=session)
        populate_with_metadata_embds(metadata_strs, metadata_embds, idx_to_title_metadata, db=session)
    