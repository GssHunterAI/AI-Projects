"""
Movie Recommender System - Demonstration Script
==============================================

This script demonstrates how to use the MovieRecommender class
with sample data or the full MovieLens dataset.

Run this script to see the movie recommender in action!
"""

import os
import sys
from movie_recommender import MovieRecommender


def check_dataset():
    """Check if the MovieLens dataset is available."""
    data_path = os.path.join('data', 'ml-1m')
    required_files = ['ratings.dat', 'users.dat', 'movies.dat']
    
    if not os.path.exists(data_path):
        return False
    
    for file in required_files:
        if not os.path.exists(os.path.join(data_path, file)):
            return False
    
    return True


def download_instructions():
    """Print instructions for downloading the dataset."""
    print("=" * 60)
    print("DATASET NOT FOUND")
    print("=" * 60)
    print("\nTo run this movie recommender, you need the MovieLens 1M dataset.")
    print("\nDownload Instructions:")
    print("1. Visit: https://grouplens.org/datasets/movielens/")
    print("2. Download 'ml-1m.zip' (MovieLens 1M Dataset)")
    print("3. Extract to the 'data/' directory")
    print("\nExpected structure:")
    print("data/")
    print("└── ml-1m/")
    print("    ├── ratings.dat")
    print("    ├── users.dat")
    print("    └── movies.dat")
    print("\nAfter downloading, run this script again!")
    print("=" * 60)


def run_demo():
    """Run the movie recommender demonstration."""
    print("=" * 60)
    print("MOVIE RECOMMENDER SYSTEM DEMO")
    print("=" * 60)
    
    # Check if dataset is available
    if not check_dataset():
        download_instructions()
        return
    
    try:
        print("\n🎬 Initializing Movie Recommender System...")
        recommender = MovieRecommender(data_dir='data/ml-1m/')
        
        print("📊 Loading MovieLens dataset...")
        recommender.load_data()
        
        print("🔧 Preprocessing data...")
        recommender.preprocess_data()
        
        # Display dataset statistics
        print(f"\n📈 Dataset Statistics:")
        print(f"   • Movies: {len(recommender.movies_df):,}")
        print(f"   • Users: {len(recommender.users_df):,}")
        print(f"   • Ratings: {len(recommender.ratings_df):,}")
        print(f"   • Genres: {len(recommender.genres)}")
        
        # Sample some data
        print(f"\n🎭 Sample Movies:")
        sample_movies = recommender.movies_df.head(3)
        for _, movie in sample_movies.iterrows():
            print(f"   • {movie['Title']} ({movie['Genres']})")
        
        print(f"\n👥 Sample Users:")
        sample_users = recommender.users_df.head(3)
        for _, user in sample_users.iterrows():
            print(f"   • User {user['UserID']}: {user['Gender']}, Age {user['Age']}, Occupation {user['Occupation']}")
        
        print("\n🔄 Creating movie-genre training pairs...")
        positive_pairs, negative_pairs = recommender.create_movie_genre_pairs()
        
        print(f"   • Positive pairs: {len(positive_pairs):,}")
        print(f"   • Negative pairs: {len(negative_pairs):,}")
        
        print("\n🏗️ Building neural network model...")
        num_movies = recommender.movies_df['MovieID'].nunique()
        num_genres = len(recommender.genres)
        
        recommender.build_movie_genre_model(num_movies, num_genres, embedding_dim=50)
        print(f"   • Model built with {num_movies} movies and {num_genres} genres")
        
        print("\n🚀 Training the model (this may take a few minutes)...")
        history = recommender.train_movie_genre_model(
            positive_pairs, 
            negative_pairs, 
            epochs=3,  # Reduced for demo
            batch_size=64
        )
        
        print("\n🎯 Extracting learned embeddings...")
        recommender.extract_movie_embeddings()
        
        print("\n💾 Saving trained model...")
        recommender.save_model()
        
        print("\n📊 Creating visualizations...")
        recommender.visualize_embeddings(
            recommender.embedded_movie_df,
            recommender.movies_df,
            "Movie Embeddings",
            color_column="Genres"
        )
        
        print("\n✅ Demo completed successfully!")
        print("\nThe movie recommender has been trained and saved.")
        print("Check the 'models/' directory for saved models.")
        print("Check the 'visualization/' directory for embedding plots.")
        
        # Show some recommendations (basic similarity)
        print("\n🎬 Sample Movie Similarities:")
        print("(Based on learned embeddings)")
        
        # Simple similarity demo
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        embeddings = recommender.embedded_movie_df.values
        similarities = cosine_similarity(embeddings)
        
        # Find most similar movies to the first few movies
        for i in range(min(3, len(recommender.movies_df))):
            movie_title = recommender.movies_df.iloc[i]['Title']
            similar_indices = similarities[i].argsort()[-4:-1][::-1]  # Top 3 similar (excluding itself)
            
            print(f"\n   Movies similar to '{movie_title}':")
            for idx in similar_indices:
                similar_title = recommender.movies_df.iloc[idx]['Title']
                similarity_score = similarities[i][idx]
                print(f"     • {similar_title} (similarity: {similarity_score:.3f})")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        print("Please check that the dataset is properly downloaded and extracted.")
        
    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_demo()
