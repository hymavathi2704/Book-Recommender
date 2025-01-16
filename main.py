import numpy as np
import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# collaborative filtering works perfectly on local
# from surprise import Reader, Dataset, SVD


# @st.cache()
# def read_ratings_data():
#     return pd.read_csv('data/ratings.csv')


# Updated caching for data loading and processing
@st.cache_data
def read_book_data():
    return pd.read_csv(r'C:\Users\Hyma vathi\Desktop\App\books_cleaned.csv')


@st.cache_data
def content(books):
    try:
        # Create the 'content' column
        books['content'] = (pd.Series(books[['authors', 'title', 'genres', 'description']]
                                      .fillna('')
                                      .values.tolist())
                            .str.join(' '))
        
        # Initialize TF-IDF Vectorizer with valid min_df
        tf_content = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, stop_words='english')
        tfidf_matrix = tf_content.fit_transform(books['content'])
        st.write("TF-IDF matrix created successfully.")
        
        # Compute cosine similarity
        cosine = linear_kernel(tfidf_matrix, tfidf_matrix)
        index = pd.Series(books.index, index=books['title'])
        return cosine, index
    except Exception as e:
        st.error(f"An error occurred in the content function: {str(e)}")
        raise




def simple_recommender(books, n=5):
    v = books['ratings_count']
    m = books['ratings_count'].quantile(0.95)
    R = books['average_rating']
    C = books['average_rating'].median()
    score = (v / (v + m) * R) + (m / (m + v) * C)
    books['score'] = score
    qualified = books.sort_values('score', ascending=False)
    return qualified[['book_id', 'title', 'authors', 'score']].head(n)


def content_recommendation(books, title, n=5):
    try:
        # Step 1: Generate cosine similarity matrix and indices
        cosine_sim, indices = content(books)
        st.write("Cosine similarity matrix and indices created successfully.")
        
        # Step 2: Check if the title exists in indices
        if title not in indices:
            st.error(f"Book title '{title}' not found in the dataset.")
            return pd.DataFrame()  # Return an empty DataFrame
        
        # Step 3: Retrieve index and similarity scores
        idx = indices[title]
        st.write(f"Index of selected book: {idx}")
        
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n + 1]  # Top N recommendations excluding the selected book
        book_indices = [i[0] for i in sim_scores]
        
        # Step 4: Return recommendations
        st.write("Recommendations generated successfully.")
        return books[['book_id', 'title', 'authors', 'average_rating', 'ratings_count']].iloc[book_indices]
    except Exception as e:
        st.error(f"An error occurred in Content-Based Filtering: {str(e)}")
        return pd.DataFrame()  # Return an empty DataFrame



def improved_recommendation(books, title, n=5):
    cosine_sim, indices = content(books)
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    book_indices = [i[0] for i in sim_scores]
    books2 = books.iloc[book_indices][['book_id', 'title', 'authors', 'average_rating', 'ratings_count']]

    v = books2['ratings_count']
    m = books2['ratings_count'].quantile(0.75)  # here the minimum rating is quantile 75
    R = books2['average_rating']
    C = books2['average_rating'].median()
    books2['weighted_rating'] = (v / (v + m) * R) + (m / (m + v) * C)

    high_rating = books2[books2['ratings_count'] >= m]
    high_rating = high_rating.sort_values('weighted_rating', ascending=False)

    return high_rating[['book_id', 'title', 'authors', 'average_rating', 'ratings_count']].head(n)



# App declaration
def main():
    st.set_page_config(page_title="Book Recommender", page_icon="📔", layout="centered", initial_sidebar_state="auto",
                       menu_items=None)

    # Header contents
    st.write('# Book Recommender')
    with st.expander("See explanation"):
        st.write("""
            In this book recommender, there are three models available.
            1. Simple Recommender
            This model offers generalized recommendations to every user based on popularity and average rating of 
            the book. This model does not provide user-specific recommendations.
            
            2.  Content Based Filtering
            To personalise our recommendations, you need to pick your favorite book. The cosine similarity between 
            books are measured, and then the model will suggest books that are most similar to a particular book that 
            a user liked.
            
            3. Content Based Filtering+
            The mechanism to remove books with low ratings has been added on top of the content based filtering.
            This model will return books that are similar to your input, are popular and have high ratings.

            """)

    books = read_book_data().copy()

    # User input
    model, book_num = st.columns((2, 1))
    selected_model = model.selectbox('Select model',
                                     options=['Simple Recommender', 'Content Based Filtering',
                                              'Content Based Filtering+'])
    selected_book_num = book_num.selectbox('Number of books',
                                           options=[5, 10, 15, 20, 25])

    if selected_model == 'Simple Recommender':
        if st.button('Recommend'):
            try:
                recs = simple_recommender(books=books,
                                          n=selected_book_num)
                st.write(recs)
            except:
                st.error('Oops!. I need to fix this algorithm.')

   
    else:
        options = np.concatenate(([''], books["title"].unique()))
        book_title = st.selectbox('Pick your favorite book', options, 0)

        if selected_model == 'Content Based Filtering':
            if st.button('Recommend'):
                if book_title == '':
                    st.write('Please pick a book or use Rating-Popularity Model')
                    return
                try:
                    recs = content_recommendation(books=books,
                                                  title=book_title,
                                                  n=selected_book_num)
                    st.write(recs)
                except:
                    st.error('Oops! I need to fix this algorithm.')

        elif selected_model == 'Content Based Filtering+':
            if book_title == '':
                st.write('Please pick a book or use Simple Recommender')
                return
            if st.button('Recommend'):
                try:
                    recs = improved_recommendation(books=books,
                                                   title=book_title,
                                                   n=selected_book_num)
                    st.write(recs)
                except:
                    st.error('Oops! I need to fix this algorithm.')


if __name__ == '__main__':
    main()
