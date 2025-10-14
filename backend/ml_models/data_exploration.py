import pandas as pd
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import torch
import torch.nn.functional as F

def preprocess_into_df(csv_file: Path):
    """
    Returns a pandas data frame that removes duplicate title entries using the following:
    1. If the title is duplicate, then add the year to it
    2. If the title + year is still a duplicate, then add the country to it
    """
    raw_df = pd.read_csv(csv_file)

    duplicates = raw_df[raw_df.duplicated(subset='title', keep=False)].copy()
    duplicates.loc[:, 'title_year'] = duplicates.loc[:, 'title'] + ' (' + duplicates.loc[:, 'release_year'].astype(str) + ')'
    duplicates.loc[:, 'title'] = duplicates.loc[:, 'title_year']
    duplicates.drop(columns=['title_year'], inplace=True)
    
    duplicates_2 = duplicates[duplicates.duplicated(subset='title', keep=False)].copy()
    duplicates_2.loc[:, 'title_year_country'] = duplicates_2.loc[:, 'title'] + ' (' + duplicates_2.loc[:, 'country'].astype(str) + ')'
    duplicates_2.loc[:, 'title'] = duplicates_2.loc[:, 'title_year_country']
    duplicates_2.drop(columns=['title_year_country'], inplace=True)

    raw_df = raw_df.drop_duplicates(subset='title', keep=False)
    duplicates = duplicates.drop_duplicates(subset='title', keep=False)
    duplicates_2.drop_duplicates(subset='title', inplace=True)

    modified_df = pd.concat([raw_df, duplicates, duplicates_2], ignore_index=True)
    modified_df.replace({np.nan: None})
    return modified_df

def retrieve_metadata_strs(
        dataframe: pd.DataFrame,
        limit: int = None
    ):
    """
    Returns: 
        Dict[str, str]: Dictionary containing the titles as keys and the metadata string as the values.
    """
    relevant_metadata = dataframe.get(['title', 'description', 'listed_in', 'director', 'cast', 'duration'])

    metadata_strings = {}
    metadata_cols = relevant_metadata.columns

    if limit is None or limit > len(relevant_metadata):
        limit = len(relevant_metadata)

    for i in range(limit):
        movie_str = ''
        title = None
        for column in metadata_cols:
            column_entry = relevant_metadata.loc[i, column]

            if column == 'title':
                title = column_entry

            if not pd.isna(column_entry):
                movie_str += f'{column}:{column_entry},'
        metadata_strings[title] = movie_str
    
    return metadata_strings

def create_sentence_embedding(input: str, model: SentenceTransformer):
    """
    Given some string and sentence transformer model, will return a vector embedding as a Torch tensor.
    """
    embedding = model.encode(input, convert_to_tensor=True)
    return embedding

def create_title_embeddings(metadata_strs:dict, model):
    """
    Takes in dict of metadata strings

    Returns 
        Torch.Tensor: A (N, D) vector where N is the number of titles and D is the embedding dimension specified by the model.
        dict: The index (order) in which the title is computed. To be used when retrieving top k similar titles.
    """
    # TODO: Figure out how to combine title embeddings with metadata strings. Should I create class to do this for me?
    titles = list(metadata_strs.keys())

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

def create_movie_embeddings(metadata_strs:dict, model):
    """
    Takes in dict of metadata strings

    Returns 
        Torch.Tensor: A (N, D) vector where N is the number of titles and D is the embedding dimension specified by the model.
        dict: The index (order) in which the title is computed. To be used when retrieving top k similar titles.
    """

    embeddings = None
    idx_to_title = {}
    for idx, title in enumerate(tqdm(metadata_strs.keys())):
        metadata = metadata_strs[title]
        embd: torch.Tensor = create_sentence_embedding(metadata, model)
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

    df = preprocess_into_df(file_path)

    metadata_strings = retrieve_metadata_strs(df, limit=20)
    # title_embds = create_title_embeddings(metadata_strings)

    for title in metadata_strings.keys():
        print(metadata_strings[title])
    
    print(len(metadata_strings))

    # for title in list(metadata_strings.keys())[0:5]:
        # print('\n' + metadata_strings[title] + '\n')
    
            
        