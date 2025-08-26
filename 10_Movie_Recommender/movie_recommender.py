"""
Movie Recommender System using Embedding Model
==============================================

This module implements a movie recommendation system using neural network embeddings
for movies, genres, and users. The system learns embeddings through the correlation
between movies and genres, and then between movies and users.

Author: AI Projects Collection
Date: August 2025
"""

import pandas as pd
import numpy as np
import itertools
import os
import logging
import time
from typing import List, Tuple, Generator

# Suppress TensorFlow warnings and info logs
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class MovieRecommender:
    """
    A movie recommendation system using neural network embeddings.
    
    This class implements a two-stage embedding learning approach:
    1. Learn movie-genre embeddings
    2. Learn user-movie embeddings using the learned movie embeddings
    """
    
    def __init__(self, data_dir: str, state_size: int = 10):
        """
        Initialize the MovieRecommender.
        
        Args:
            data_dir (str): Path to the directory containing MovieLens data
            state_size (int): Size of the embedding vectors
        """
        self.data_dir = data_dir
        self.state_size = state_size
        self.genres = [
            'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
            'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
            'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
            'Thriller', 'War', 'Western'
        ]
        
        # Initialize data containers
        self.ratings_df = None
        self.movies_df = None
        self.users_df = None
        self.movies_genres_df = None
        self.user_movie_rating_df = None
        
        # Initialize models
        self.m_g_model = None
        self.u_m_model = None
        
        # Initialize embeddings
        self.embedded_movie_df = None
        self.embedded_user_df = None
    
    def load_data(self):
        """Load and preprocess the MovieLens dataset."""
        print("Loading MovieLens dataset...")
        
        # Read the ratings data
        ratings_list = [
            i.strip().split("::") 
            for i in open(os.path.join(self.data_dir, 'ratings.dat'), 'r').readlines()
        ]
        
        # Read the users data
        users_list = [
            i.strip().split("::") 
            for i in open(os.path.join(self.data_dir, 'users.dat'), 'r').readlines()
        ]
        
        # Read the movies data
        movies_list = [
            i.strip().split("::") 
            for i in open(os.path.join(self.data_dir, 'movies.dat'), encoding='latin-1').readlines()
        ]
        
        # Convert to DataFrames
        self.ratings_df = pd.DataFrame(
            ratings_list, 
            columns=['UserID', 'MovieID', 'Rating', 'Timestamp'], 
            dtype=np.float32
        )
        
        self.movies_df = pd.DataFrame(
            movies_list, 
            columns=['MovieID', 'Title', 'Genres']
        )
        self.movies_df['MovieID'] = self.movies_df['MovieID'].apply(pd.to_numeric)
        
        self.users_df = pd.DataFrame(
            users_list, 
            columns=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
        )
        
        print(f"Loaded {len(self.ratings_df)} ratings, {len(self.movies_df)} movies, {len(self.users_df)} users")
    
    def preprocess_data(self):
        """Preprocess the data for training."""
        print("Preprocessing data...")
        
        # Create movie-genre dataframe
        self.movies_genres_df = self.movies_df[['MovieID', 'Genres']].copy()
        
        # Split and index genres
        def _split_and_index(genre_string):
            genre_list = genre_string.split('|')
            return [self.genres.index(genre) for genre in genre_list]
        
        self.movies_genres_df['Genres'] = self.movies_genres_df['Genres'].map(_split_and_index)
        
        # Create user-movie-rating dataframe
        self.user_movie_rating_df = self.ratings_df[['UserID', 'MovieID', 'Rating']].copy()
    
    def create_movie_genre_pairs(self) -> Tuple[List, List]:
        """
        Create positive and negative movie-genre pairs for training.
        
        Returns:
            Tuple of positive and negative pairs
        """
        print("Creating movie-genre pairs...")
        
        positive_pairs = []
        negative_pairs = []
        
        for _, row in self.movies_genres_df.iterrows():
            movie_id = row['MovieID']
            movie_genres = row['Genres']
            
            # Create positive pairs
            for genre_id in movie_genres:
                positive_pairs.append([movie_id, genre_id, 1])
            
            # Create negative pairs
            negative_genres = [g for g in range(len(self.genres)) if g not in movie_genres]
            for genre_id in negative_genres:
                negative_pairs.append([movie_id, genre_id, 0])
        
        print(f"Created {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs")
        return positive_pairs, negative_pairs
    
    def batch_generator(self, positive_pairs: List, negative_pairs: List, 
                       batch_size: int = 32, negative_ratio: float = 0.5) -> Generator:
        """
        Generate batches of movie-genre pairs for training.
        
        Args:
            positive_pairs: List of positive movie-genre pairs
            negative_pairs: List of negative movie-genre pairs
            batch_size: Size of each batch
            negative_ratio: Ratio of negative pairs in each batch
            
        Yields:
            Tuple of movie IDs, genre IDs, and labels
        """
        batch = np.zeros((batch_size, 3))
        num_positive = batch_size - int(batch_size * negative_ratio)
        
        while True:
            # Sample positive pairs
            idx = np.random.choice(len(positive_pairs), num_positive)
            positive_data = np.array(positive_pairs)[idx]
            batch[:num_positive] = positive_data
            
            # Sample negative pairs
            idx = np.random.choice(len(negative_pairs), int(batch_size * negative_ratio))
            negative_data = np.array(negative_pairs)[idx]
            batch[num_positive:] = negative_data
            
            # Shuffle the batch
            np.random.shuffle(batch)
            yield batch[:, 0], batch[:, 1], batch[:, 2]
    
    def build_movie_genre_model(self, num_movies: int, num_genres: int, embedding_dim: int = 50):
        """
        Build the movie-genre embedding model.
        
        Args:
            num_movies: Number of unique movies
            num_genres: Number of unique genres
            embedding_dim: Dimension of embedding vectors
        """
        print("Building movie-genre model...")
        
        # Input layers
        movie_input = tf.keras.layers.Input(shape=(), name='movie_input')
        genre_input = tf.keras.layers.Input(shape=(), name='genre_input')
        
        # Embedding layers
        movie_embedding = tf.keras.layers.Embedding(
            num_movies + 1, embedding_dim, name='movie_embedding'
        )(movie_input)
        genre_embedding = tf.keras.layers.Embedding(
            num_genres, embedding_dim, name='genre_embedding'
        )(genre_input)
        
        # Flatten embeddings
        movie_vec = tf.keras.layers.Flatten()(movie_embedding)
        genre_vec = tf.keras.layers.Flatten()(genre_embedding)
        
        # Compute dot product
        dot_product = tf.keras.layers.Dot(axes=1)([movie_vec, genre_vec])
        output = tf.keras.layers.Dense(1, activation='sigmoid')(dot_product)
        
        # Create model
        self.m_g_model = tf.keras.Model(inputs=[movie_input, genre_input], outputs=output)
        self.m_g_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        print("Movie-genre model built successfully")
    
    def train_movie_genre_model(self, positive_pairs: List, negative_pairs: List, 
                               epochs: int = 10, batch_size: int = 32):
        """Train the movie-genre embedding model."""
        print(f"Training movie-genre model for {epochs} epochs...")
        
        generator = self.batch_generator(positive_pairs, negative_pairs, batch_size)
        steps_per_epoch = (len(positive_pairs) + len(negative_pairs)) // batch_size
        
        history = self.m_g_model.fit(
            generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=1
        )
        
        return history
    
    def extract_movie_embeddings(self):
        """Extract learned movie embeddings."""
        print("Extracting movie embeddings...")
        
        movie_ids = self.movies_df['MovieID'].values
        movie_embeddings = self.m_g_model.get_layer('movie_embedding')(movie_ids)
        
        # Convert to DataFrame
        columns = [f'dim_{i}' for i in range(movie_embeddings.shape[1])]
        self.embedded_movie_df = pd.DataFrame(movie_embeddings.numpy(), columns=columns)
        
        print(f"Extracted embeddings for {len(self.embedded_movie_df)} movies")
    
    def visualize_embeddings(self, embedding_df: pd.DataFrame, metadata_df: pd.DataFrame, 
                           title: str, color_column: str = None):
        """
        Visualize embeddings using PCA and t-SNE.
        
        Args:
            embedding_df: DataFrame containing embeddings
            metadata_df: DataFrame containing metadata for coloring
            title: Title for the plots
            color_column: Column to use for coloring points
        """
        print(f"Creating visualizations for {title}...")
        
        # PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(embedding_df)
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
        if color_column and color_column in metadata_df.columns:
            pca_df[color_column] = metadata_df[color_column].values
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(scaled_data)
        
        tsne_df = pd.DataFrame(tsne_result, columns=['t-SNE1', 't-SNE2'])
        if color_column and color_column in metadata_df.columns:
            tsne_df[color_column] = metadata_df[color_column].values
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        if color_column:
            scatter1 = ax1.scatter(pca_df['PC1'], pca_df['PC2'], c=pd.Categorical(pca_df[color_column]).codes, alpha=0.6)
            scatter2 = ax2.scatter(tsne_df['t-SNE1'], tsne_df['t-SNE2'], c=pd.Categorical(tsne_df[color_column]).codes, alpha=0.6)
        else:
            ax1.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.6)
            ax2.scatter(tsne_df['t-SNE1'], tsne_df['t-SNE2'], alpha=0.6)
        
        ax1.set_title(f'{title} - PCA')
        ax1.set_xlabel('First Principal Component')
        ax1.set_ylabel('Second Principal Component')
        
        ax2.set_title(f'{title} - t-SNE')
        ax2.set_xlabel('t-SNE 1')
        ax2.set_ylabel('t-SNE 2')
        
        plt.tight_layout()
        plt.savefig(f'visualization/{title.lower().replace(" ", "_")}_embeddings.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, model_path: str = 'models/'):
        """Save the trained models."""
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        if self.m_g_model:
            self.m_g_model.save(os.path.join(model_path, 'movie_genre_model.h5'))
            print("Movie-genre model saved")
        
        if self.u_m_model:
            self.u_m_model.save(os.path.join(model_path, 'user_movie_model.h5'))
            print("User-movie model saved")
    
    def load_model(self, model_path: str = 'models/'):
        """Load pre-trained models."""
        movie_genre_path = os.path.join(model_path, 'movie_genre_model.h5')
        user_movie_path = os.path.join(model_path, 'user_movie_model.h5')
        
        if os.path.exists(movie_genre_path):
            self.m_g_model = tf.keras.models.load_model(movie_genre_path)
            print("Movie-genre model loaded")
        
        if os.path.exists(user_movie_path):
            self.u_m_model = tf.keras.models.load_model(user_movie_path)
            print("User-movie model loaded")


def main():
    """Main function to demonstrate the movie recommender system."""
    # Initialize the recommender
    recommender = MovieRecommender(data_dir='data/ml-1m/')
    
    # Load and preprocess data
    recommender.load_data()
    recommender.preprocess_data()
    
    # Create training pairs
    positive_pairs, negative_pairs = recommender.create_movie_genre_pairs()
    
    # Build and train the model
    num_movies = recommender.movies_df['MovieID'].nunique()
    num_genres = len(recommender.genres)
    
    recommender.build_movie_genre_model(num_movies, num_genres)
    history = recommender.train_movie_genre_model(positive_pairs, negative_pairs, epochs=5)
    
    # Extract embeddings
    recommender.extract_movie_embeddings()
    
    # Save the model
    recommender.save_model()
    
    print("Movie recommender training completed!")


if __name__ == "__main__":
    main()
