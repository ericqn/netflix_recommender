from sqlalchemy.orm import Session

from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware

from sentence_transformers import SentenceTransformer

from source.vector_db.embedding_utils import create_sentence_embedding
from source.vector_db.operations import get_top_k_title_embds, get_top_k_metadata_embds, show_info
from source.vector_db.models import get_db

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

sentence_embd_model = SentenceTransformer("all-MiniLM-L6-v2")

@app.get('/get-title')
async def get_title(
    user_input: str, 
    top_k: int, 
    db: Session = Depends(get_db)
):
    input_embd = create_sentence_embedding(user_input, sentence_embd_model)

    query_res = get_top_k_title_embds(user_input, input_embd, k=top_k, db=db)
    
    titles = []
    # main_title = None
    min_score = 2
    for item, sim_score in query_res:
        if sim_score < min_score:
            # main_title = item.movie_title
            min_score = sim_score
        titles.append(item.movie_title)
    
    if min_score <= 0.6:
        return {
            'status': 'success',
            'title': titles,
            'sim_score': min_score
        }
    else:
        return {
            'status': 'failed',
            'title': None,
            'sim_score': min_score
        }


@app.get('/get-recommendations')
async def get_recommendations(
    movie_title: str, 
    top_k: int, 
    db: Session = Depends(get_db)
):
    query_result = get_top_k_metadata_embds(movie_title, top_k+1, db)

    movie_names = []
    sim_scores = []

    for item, sim_score in query_result:
        if sim_score <= 0.05:
            continue
        movie_names.append(item.movie_title)
        sim_scores.append(sim_score)
        
        if len(movie_names) == top_k:
            break

    # TODO: Return descriptions.
    return {
        'movie_names': movie_names,
        'sim_scores': sim_scores
    }


@app.get('/get-show-info')
async def get_show_info(movie_title: str, db: Session = Depends(get_db)):
    show = show_info(movie_title, db)

    return {
        'description' : show.description,
        'director' : show.director,
        'cast' : show.cast,
        'genres' : show.genres,
        'country_origin' : show.country_origin,
        'date_added' : show.date_added,
        'release_year' : show.release_year,
        'rating' : show.rating,
        'duration' : show.duration,
        'show_type' : show.show_type
    }

@app.get('/debug-endpoint')
async def debug(item: str):
    return {'retrieved_item' : item}