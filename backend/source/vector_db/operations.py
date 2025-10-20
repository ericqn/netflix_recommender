from fastapi import Depends

import numpy as np
import torch
from sqlalchemy import text, select, func
from sqlalchemy.orm import Session

from sentence_transformers import SentenceTransformer

# import models
# import embedding_utils
from . import models
from . import embedding_utils

def get_top_k_title_embds(
        movie_title: str,
        input_embd: torch.Tensor, 
        k: int, 
        db: Session
    ):
    
    query = (
        db.query(
            models.TitleDatabase,
            (models.TitleDatabase.title_embedding.cosine_distance(input_embd) / 2
             + (1 - func.similarity(models.TitleDatabase.movie_title, movie_title)) / 2)
            .label('sim_score')
        )
        .order_by(text('sim_score'))
        .limit(k)
        .all()
    )

    return query


def get_top_k_metadata_embds(
    movie_title: str,
    k: int,
    db: Session
):
    metadata_embd = (
        db.query(models.MetadataDatabase)
        .filter(models.MetadataDatabase.movie_title == movie_title)
        .one_or_none()
    ).metadata_embedding
    
    similarity_threshold = 0.7
    top_k_similar = (
        db.query(
            models.MetadataDatabase,
            models.MetadataDatabase.metadata_embedding.cosine_distance(metadata_embd).label('sim_score')
        )
        .filter(models.MetadataDatabase.metadata_embedding.cosine_distance(metadata_embd) <= similarity_threshold)
        .order_by(text('sim_score'))
        .limit(k)
        .all()
    )

    return top_k_similar


# testing:
if __name__ == '__main__':
    embd_model = SentenceTransformer("all-MiniLM-L6-v2")

    movie_title = 'breakn ba'
    sample_embedding = embedding_utils.create_sentence_embedding(movie_title, embd_model)

    with models.SessionLocal() as session:
        query = get_top_k_title_embds(
            movie_title,
            sample_embedding,
            k=5,
            db=session
        )

        print(query)
        print(type(query))

        for item, sim_score in query:
            print(f'MOVIE TITLE = {item.movie_title} | sim_score = {sim_score}')
            
            # similar_movies = get_top_k_metadata_embds(item.movie_title, 3, session)
            # print(f'\n\nTOP 3 SIMILAR MOVIES TO {item.movie_title}:\n\n')
            # for sim_item, score in similar_movies:
            #     print(f'{sim_item.movie_title} | score = {score}')

# sample titles in database (try searching for them and getting recommendations!)
#  Mi Obra Maestra
#  Bad Seeds
#  Free Rein: The Twelve Neighs of Christmas
#  Oddbods: The Festive Menace
#  Errementari: The Blacksmith and the Devil
#  Dare to Be Wild
#  John Mulaney: New in Town
#  Steel Rain
#  Shuddhi
#  The Jack King Affair
#  City of Tiny Lights
#  Miss Panda & Mr. Hedgehog
#  Frozen Planet: The Epic Journey
#  SMOSH: The Movie
#  Yellowbird
#  Oh My Ghost (2015)