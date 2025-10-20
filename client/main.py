import streamlit as st
import requests

from utils.api import send_get_request

class TitleDisplay:
    def __init__(self, movie_name):
        self.movie_name = movie_name
    
    def display(self):
        pass

def main():
    st.title('Next'+ ':red[flix]')

    col1, col2 = st.columns([2, 1])


    with col1:
        movie_name = st.text_input(
            'Search for a show or movie!', 
            placeholder='What was your most recently watched show?',
        )
    
    with col2:
        top_k = st.number_input(
            'Search Titles Displayed',
            min_value = 1,
            max_value = 10,
            value = "min"
        )


    if movie_name is not None and len(movie_name) > 0:
        payload = {'user_input': movie_name, 'top_k': top_k}
        similar_movies = send_get_request('/get-title', request=payload)
        if similar_movies.get('status') == 'success':
            sim_score = similar_movies.get('sim_score')
            st.write(f'Sim score result = {sim_score}')
            user_choice = st.selectbox(label='Found shows', options=similar_movies.get('title'))
            st.write(f'Your choice was {user_choice}')

            payload = {'movie_title': user_choice, 'top_k': 5}
            recommendations = send_get_request('/get-recommendations', request=payload)
            for movie_title, sim_score in zip(recommendations.get('movie_names'), recommendations.get('sim_scores')):
                # st.write(f'Recommended: {movie_title} | sim score = {sim_score}')
                st.write(f'Recommended: {movie_title}')
        else:
            sim_score = similar_movies.get('sim_score')
            st.write(f'Sim score result = {sim_score}')
            st.write('Could not find movie in our database. Please try again.')


if __name__ == "__main__":
    main()
