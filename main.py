import numpy as np
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

@st.cache_data
def read_book_data():
    """Load the book dataset."""
    return pd.read_csv(r'books_cleaned.csv')


@st.cache_data
def content(books):
    """Generate a cosine similarity matrix for content-based filtering."""
    try:
        books['content'] = (pd.Series(books[['authors', 'title', 'genres', 'description']]
                                      .fillna('')
                                      .values.tolist())
                            .str.join(' '))
        
        tf_content = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, stop_words='english')
        tfidf_matrix = tf_content.fit_transform(books['content'])
        cosine = linear_kernel(tfidf_matrix, tfidf_matrix)
        index = pd.Series(books.index, index=books['title'])
        return cosine, index
    except Exception as e:
        st.error(f"Error in content function: {str(e)}")
        raise


def simple_recommender(books, n=5):
    """Generate simple recommendations based on popularity and average ratings."""
    v = books['ratings_count']
    m = books['ratings_count'].quantile(0.95)
    R = books['average_rating']
    C = books['average_rating'].median()
    score = (v / (v + m) * R) + (m / (m + v) * C)
    books['score'] = score
    qualified = books.sort_values('score', ascending=False)
    return qualified[['book_id', 'title', 'authors', 'score']].head(n)


def content_recommendation(books, title, n=5):
    """Generate recommendations using content-based filtering."""
    try:
        cosine_sim, indices = content(books)
        if title not in indices:
            st.error(f"Book title '{title}' not found in the dataset.")
            return pd.DataFrame()

        idx = indices[title]
        sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n + 1]
        book_indices = [i[0] for i in sim_scores]
        return books[['book_id', 'title', 'authors', 'average_rating', 'ratings_count']].iloc[book_indices]
    except Exception as e:
        st.error(f"Error in content recommendation: {str(e)}")
        return pd.DataFrame()


def improved_recommendation(books, title, n=5):
    """Generate enhanced recommendations with weighted ratings."""
    try:
        cosine_sim, indices = content(books)
        if title not in indices:
            st.error(f"Book title '{title}' not found in the dataset.")
            return pd.DataFrame()

        idx = indices[title]
        sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:26]
        book_indices = [i[0] for i in sim_scores]
        books2 = books.iloc[book_indices][['book_id', 'title', 'authors', 'average_rating', 'ratings_count']]

        v = books2['ratings_count']
        m = books2['ratings_count'].quantile(0.75)
        R = books2['average_rating']
        C = books2['average_rating'].median()
        books2['weighted_rating'] = (v / (v + m) * R) + (m / (m + v) * C)

        high_rating = books2[books2['ratings_count'] >= m]
        high_rating = high_rating.sort_values('weighted_rating', ascending=False)
        return high_rating[['book_id', 'title', 'authors', 'average_rating', 'ratings_count']].head(n)
    except Exception as e:
        st.error(f"Error in improved recommendation: {str(e)}")
        return pd.DataFrame()


def main():
    st.set_page_config(page_title="Book Recommender", page_icon="📔", layout="centered")

    st.write('# Book Recommender')
    with st.expander("See explanation"):
        st.write("""
            This app provides three recommendation models:
            1. **Simple Recommender**: Based on popularity and average ratings.
            2. **Content-Based Filtering**: Suggests books similar to the selected one.
            3. **Enhanced Content-Based Filtering**: Filters out low-rated books for better recommendations.
        """)

    books = read_book_data().copy()

    model, book_num = st.columns((2, 1))
    selected_model = model.selectbox('Select model', options=['Simple Recommender', 'Content-Based Filtering', 'Enhanced Content-Based Filtering'])
    selected_book_num = book_num.selectbox('Number of books', options=[5, 10, 15, 20, 25])

    if selected_model == 'Simple Recommender':
        if st.button('Recommend'):
            recs = simple_recommender(books=books, n=selected_book_num)
            st.write(recs)

    else:
        options = np.concatenate(([''], books['title'].unique()))
        book_title = st.selectbox('Pick your favorite book', options, 0)

        if st.button('Recommend'):
            if book_title == '':
                st.error('Please pick a book.')
                return

            if selected_model == 'Content-Based Filtering':
                recs = content_recommendation(books=books, title=book_title, n=selected_book_num)
            elif selected_model == 'Enhanced Content-Based Filtering':
                recs = improved_recommendation(books=books, title=book_title, n=selected_book_num)
            
            st.write(recs)


if __name__ == '__main__':
    main()
