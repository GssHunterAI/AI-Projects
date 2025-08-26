# Movie Recommender System

A sophisticated movie recommendation system built using neural network embeddings and the MovieLens dataset. This project implements a two-stage embedding learning approach to understand the relationships between movies, genres, and users.

## Project Overview

This movie recommender system uses deep learning to create meaningful embeddings for movies, genres, and users. The system learns these embeddings through two main stages:

1. **Movie-Genre Embedding Learning**: First learns the correlation between movies and their genres
2. **User-Movie Embedding Learning**: Then learns user preferences using the pre-trained movie embeddings

## Features

- **Neural Network Embeddings**: Uses TensorFlow/Keras to create dense vector representations
- **Two-Stage Learning**: Hierarchical approach for better representation learning
- **Visualization**: PCA and t-SNE visualizations of learned embeddings
- **Modular Design**: Clean, object-oriented Python implementation
- **Comprehensive Analysis**: Jupyter notebook with detailed exploration

## Dataset

The project uses the MovieLens 1M dataset which contains:
- **1,000,209 ratings** from 6,040 users on 3,952 movies
- **User demographics**: Gender, age, occupation, zip code
- **Movie metadata**: Title, genres (18 different genres)
- **Rating scale**: 1-5 stars

### Dataset Structure
```
ml-1m/
├── ratings.dat    # UserID::MovieID::Rating::Timestamp
├── users.dat      # UserID::Gender::Age::Occupation::Zip-code
└── movies.dat     # MovieID::Title::Genres
```

## Installation

1. Clone this repository or download the project files
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download the MovieLens 1M dataset:
   - Visit [MovieLens Datasets](https://grouplens.org/datasets/movielens/)
   - Download the ml-1m.zip file
   - Extract it to the `data/` directory

## Usage

### Quick Start

Run the main script to train the recommender system:

```python
python movie_recommender.py
```

### Detailed Analysis

For a comprehensive exploration, open the Jupyter notebook:

```bash
jupyter notebook Movie_Recommender.ipynb
```

### Using the MovieRecommender Class

```python
from movie_recommender import MovieRecommender

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
history = recommender.train_movie_genre_model(positive_pairs, negative_pairs)

# Extract and visualize embeddings
recommender.extract_movie_embeddings()
recommender.visualize_embeddings(
    recommender.embedded_movie_df, 
    recommender.movies_df, 
    "Movie Embeddings",
    color_column="Genres"
)
```

## Technical Implementation

### Architecture

The system uses a neural network architecture with:
- **Embedding Layers**: Dense vector representations for movies, genres, and users
- **Dot Product Similarity**: Measures compatibility between embeddings
- **Binary Classification**: Predicts movie-genre and user-movie associations

### Model Components

1. **Movie-Genre Model**:
   - Input: Movie ID and Genre ID
   - Output: Probability of association
   - Loss: Binary crossentropy

2. **User-Movie Model** (planned extension):
   - Uses pre-trained movie embeddings
   - Learns user preferences
   - Enables personalized recommendations

### Training Process

1. **Data Preprocessing**: Convert categorical data to numerical indices
2. **Pair Generation**: Create positive and negative training examples
3. **Batch Generation**: Efficient mini-batch training with balanced sampling
4. **Model Training**: SGD optimization with Adam optimizer
5. **Embedding Extraction**: Extract learned representations for analysis

## Visualization

The project includes comprehensive visualization tools:

- **PCA Analysis**: 2D projection of high-dimensional embeddings
- **t-SNE Visualization**: Non-linear dimensionality reduction
- **Interactive Plots**: Plotly-based interactive exploration
- **Genre Clustering**: Visual analysis of movie genre relationships

## Results

The trained model produces:
- **Meaningful Movie Embeddings**: Similar movies cluster together
- **Genre Relationships**: Clear separation and clustering by genre
- **User Patterns**: Identifiable user preference groups

## Project Structure

```
10_Movie_Recommender/
├── README.md                    # This file
├── Movie_Recommender.ipynb      # Jupyter notebook with full analysis
├── movie_recommender.py         # Main Python implementation
├── requirements.txt             # Python dependencies
├── data/                        # Dataset directory
│   └── ml-1m/                  # MovieLens 1M dataset
├── models/                      # Saved model files
│   ├── movie_genre_model.h5    # Trained movie-genre model
│   └── user_movie_model.h5     # Trained user-movie model (future)
└── visualization/               # Generated plots and charts
    └── movie_embeddings.png     # Embedding visualizations
```

## Dependencies

- **TensorFlow**: Deep learning framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib**: Static plotting
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning utilities (PCA, t-SNE)

## Future Enhancements

1. **User-Movie Model**: Complete the second stage of embedding learning
2. **Recommendation Engine**: Implement actual recommendation functionality
3. **Web Interface**: Create a Flask/Django web application
4. **Advanced Architectures**: Experiment with attention mechanisms
5. **Cold Start Problem**: Handle new users and movies
6. **Evaluation Metrics**: Implement RMSE, precision@k, recall@k

## Research Applications

This project demonstrates several important concepts:
- **Representation Learning**: Learning meaningful vector representations
- **Transfer Learning**: Using pre-trained embeddings for downstream tasks
- **Collaborative Filtering**: Learning from user-item interactions
- **Dimensionality Reduction**: Visualizing high-dimensional data

## Contributing

Feel free to contribute to this project by:
1. Implementing the user-movie embedding model
2. Adding evaluation metrics
3. Creating a web interface
4. Improving visualization tools
5. Adding more advanced neural architectures

## License

This project is open source and available under the MIT License.

## Acknowledgments

- **MovieLens Dataset**: GroupLens Research at the University of Minnesota
- **TensorFlow Team**: For the excellent deep learning framework
- **Scientific Python Community**: For the comprehensive ecosystem

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue in this repository.
