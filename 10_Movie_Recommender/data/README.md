# MovieLens Dataset

This directory should contain the MovieLens 1M dataset.

## Download Instructions

1. Visit: https://grouplens.org/datasets/movielens/
2. Download the "ml-1m.zip" file (MovieLens 1M Dataset)
3. Extract the contents to this directory

## Expected Structure

After extraction, this directory should contain:
```
data/
└── ml-1m/
    ├── ratings.dat
    ├── users.dat
    ├── movies.dat
    └── README
```

## Dataset Description

- **ratings.dat**: User ratings (UserID::MovieID::Rating::Timestamp)
- **users.dat**: User demographics (UserID::Gender::Age::Occupation::Zip-code)
- **movies.dat**: Movie information (MovieID::Title::Genres)

The dataset contains 1 million ratings from 6,000 users on 4,000 movies.
