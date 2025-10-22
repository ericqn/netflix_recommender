import streamlit as st
import requests

from utils.api import send_get_request

class TitleDisplay:
    def __init__(self, movie_name):
        self.movie_name = movie_name
    
    def display(self):
        pass

def score_to_text(sim_score):
    if sim_score < 0.5:
        return ':green[high]'
    elif sim_score < 0.6:
        return ':yellow[medium]'
    elif sim_score < 0.7:
        return ':orange[subpar]'
    else:
        return ':red[low]'

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
            min_value = 3,
            max_value = 10,
            value = "min"
        )

    desired_info = ['Description', 'Director', 'Cast', 'Rating']
    if movie_name is not None and len(movie_name) > 0:
        payload = {'user_input': movie_name, 'top_k': top_k}
        similar_movies = send_get_request('/get-title', request=payload)
        if similar_movies.get('status') == 'success':
            sim_score = similar_movies.get('sim_score')
            # st.write(f'Sim score result = {sim_score}')
            user_choice = st.selectbox(label='Found shows', options=similar_movies.get('title'))
            information = send_get_request('/get-show-info', request = {'movie_title' : user_choice})
            
            st.markdown(f'## {user_choice}')
            for item in desired_info:
                    retrieved_info = information.get(item.lower())
                    if retrieved_info is not None:
                        st.write(f'{item} : {retrieved_info}')

            rec_payload = {'movie_title': user_choice, 'top_k': 5}
            recommendations = send_get_request('/get-recommendations', request=rec_payload)


            for movie_title, sim_score in zip(recommendations.get('movie_names'), recommendations.get('sim_scores')):
                info_payload = {'movie_title': movie_title}
                information = send_get_request('/get-show-info', request = info_payload)
                st.markdown(f'### :red[Recommended:] {movie_title}')

                for item in desired_info:
                    retrieved_info = information.get(item.lower())
                    if retrieved_info is not None:
                        st.write(f'{item} : {retrieved_info}')

                st.write(f'Similarity to {user_choice} : {score_to_text(sim_score)}')

        else:
            sim_score = similar_movies.get('sim_score')
            # st.write(f'Sim score result = {sim_score}')
            st.write('Could not find movie in our database. Please try again.')


if __name__ == "__main__":
    main()
