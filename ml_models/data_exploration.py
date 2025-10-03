import pandas as pd
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import torch
import torch.nn.functional as F

# TODO: Expand docstring
def create_metadata_strs(data_csv_path):
    """
    Returns: 
        Dict[str, str]: Dictionary containing the titles as keys and the metadata string as the values.
    """
    raw_df = pd.read_csv(data_csv_path)
    relevant_metadata = raw_df.get(['title', 'description', 'listed_in', 'director', 'cast', 'duration'])
    print(f'FULL DATA:\n{raw_df}')
    print(f'columns: {raw_df.columns}')
    print(f'num rows/entries in data: {len(raw_df)}')

    metadata_strings = {}
    duplicate_titles = set()
    metadata_cols = relevant_metadata.columns

    # TODO: Deal with duplicates by indicating the release date
    # In the case where there are more than 1 duplicate, we can fix that by adding the title to the
    # duplicate titles set once the first dupe is found
    # We will append to our metadata_strings dictionary as show_name (release_year)
    end = 0
    for i in range(len(relevant_metadata)):
        
        movie_str = ''
        title = None
        for column in metadata_cols:
            column_entry = relevant_metadata.loc[i, column]

            if column == 'title' and title not in duplicate_titles:
                release_year = raw_df.loc[i, 'release_year']
                title = f'{column_entry} ({release_year})'
                column_entry = f'{column_entry} ({release_year})'
            elif column == 'title' and title in duplicate_titles:
                release_year = raw_df.loc[i, 'release_year']
                title = column_entry + raw_df.loc[i, 'release_year']
                release_year = raw_df.loc[i, 'release_year']
                column_entry = title

            movie_str += f'{column}:{column_entry},'

        if title in metadata_strings:
            duplicate_titles.add(title)
            print(f'\nduplicate title error: {title}')
            print(f'\n========= current entry =========\n{metadata_strings[title]}')
            print(f'\n========= new entry =========\n{movie_str}')

        else:
            metadata_strings[title] = movie_str
        
    print(f'length of metadata_strings = {len(metadata_strings)}')
    return metadata_strings

def retrieve_titles(data_csv_path):
    raw_df = pd.read_csv(data_csv_path)

    titles = set()
    for i in range(len(raw_df)):
        curr_title = raw_df.loc[i, 'title']
        if curr_title not in titles:
            titles.add(curr_title)

    return sorted(list(titles))

def create_sentence_embedding(input: str, model: SentenceTransformer):
    """
    Given some string and sentence transformer model, will return a vector embedding as a Torch tensor.
    """
    embedding = model.encode(input, convert_to_tensor=True)
    return embedding

def create_title_embeddings(titles, model):
    """
    Returns 
        Torch.Tensor: A (N, D) vector where N is the number of titles and D is the embedding dimension specified by the model.
        dict: The index (order) in which the title is computed. To be used when retrieving top k similar titles.
    """
    N = len(titles)
    embeddings = None
    idx_to_title = {}
    for idx, title in enumerate(tqdm(titles)):
        embd: torch.Tensor = create_sentence_embedding(title, model)
        embd = embd.unsqueeze(dim=0)
        if embeddings is None:
            embeddings = embd
        else:
            embeddings = torch.cat((embeddings, embd), dim=0)
        
        idx_to_title[idx] = title
    
    return embeddings, idx_to_title

def compute_similarity(embd_1, embd_2):
    return F.cosine_similarity(embd_1, embd_2, dim=0)

def top_k_similar_embeddings(
        input_embd:torch.Tensor, 
        embd_list: torch.Tensor, 
        idx_to_title:dict, 
        k:int = 3
    ):
    """
    Computes top k similar embeddings.
    """
    # Hyperparameters:
    THRESHOLD = (0.015 * len(embd_list)) / 100
    print(f'THRESHOLD VALUE = {THRESHOLD}')
    TEMPERATURE = 4.0

    sim_scores = []
    for embedding in embd_list:
        score = compute_similarity(input_embd, embedding)
        sim_scores.append(score)
    
    sim_scores = F.softmax(TEMPERATURE * torch.tensor(sim_scores, dtype=torch.float), dim=0)
    top_sim_scores, top_sim_indices = torch.topk(sim_scores, k, largest=True, sorted=True)

    # Convert top sim scores to probability distribution:
    print(f'TOP {k} SIM SCORES: {top_sim_scores}')

    if top_sim_scores[0] < THRESHOLD:
        # TODO: figure out how to handle this
        print(f'No great matches found. Did you mean {idx_to_title[top_sim_indices[0].item()]}')
        return None

    similar_titles = []
    for idx in top_sim_indices:
        similar_titles.append(idx_to_title[idx.item()])
    
    return similar_titles

if __name__ == '__main__':
    file_path = Path('data') / 'netflix_titles_nov_2019.csv'

    # metadata_strings = create_metadata_strs(file_path)
    titles = retrieve_titles(file_path)
    embd_model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings, idx_to_title = create_title_embeddings(titles[0:200], embd_model)
    print(idx_to_title)
    chosen_embedding = embeddings[-1]
    titles_with_noise = [
        'glimpse inside chrles 3',
        'rucker 50',
        'reality high',
        'one chance to dance',
        'seven years',
        'last men in the phillipines',
        'series of unfortunate events',
        'six years movie',
        'fog patch'
    ]

    for title in titles_with_noise:
        embedding = create_sentence_embedding(title, embd_model)
        top_titles_good = top_k_similar_embeddings(embedding, embeddings, idx_to_title, k=5)
        print(f' {title} | Top similar titles: {top_titles_good}')
    
    # for title in list(metadata_strings.keys())[0:5]:
    #     print('\n' + metadata_strings[title] + '\n')
    
            
        