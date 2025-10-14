from fastapi import Depends

import numpy as np
import torch
from sqlalchemy import text, select, func
from sqlalchemy.orm import Session

from sentence_transformers import SentenceTransformer
from embedding_utils import create_sentence_embedding

from models import SessionLocal, TitleDatabase, get_db

# 
def get_top_k_title_embds(
        input_embd: torch.Tensor, 
        k: int, 
        db: Session = Depends(get_db)
    ):
    
    query = (
        db.query(
            TitleDatabase,
            TitleDatabase.title_embedding.cosine_distance(input_embd).label('sim_score')
        )
        .order_by(text('sim_score'))
        .limit(k)
        .all()
    )

    return query


if __name__ == '__main__':
    embd_model = SentenceTransformer("all-MiniLM-L6-v2")

    sample_embedding = create_sentence_embedding('bad seeds', embd_model)

    with SessionLocal() as session:
        query = get_top_k_title_embds(
            sample_embedding,
            k=5,
            db=session
        )

        print(query)
        print(type(query))

        for item, sim_score in query:
            print(f'MOVIE TITLE = {item.movie_title} | sim_score = {sim_score}')

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