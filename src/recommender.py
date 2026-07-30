import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_segment_top_genre(segment_name, interactions_df, user_segments_df, genre_col="genre_list"):
    
    segment_users = user_segments_df[user_segments_df["segment"] == segment_name]["user_id"]
    segment_interactions = interactions_df[interactions_df["user_id"].isin(segment_users)].copy()

    exploded = segment_interactions.explode(genre_col)

    genre_stats = (
        exploded.groupby(genre_col)["completion_rate"]
        .agg(["mean", "count"])
        .query("count >= 20")  
        .sort_values("mean", ascending=False)
    )

    if genre_stats.empty:
        return None

    return genre_stats.index[0]


def genre_based_recommendations(segment_name, interactions_df, user_segments_df, catalog_df,
                                 genre_col="genre_list", top_n=5):
    
    top_genre = get_segment_top_genre(segment_name, interactions_df, user_segments_df, genre_col)
    if top_genre is None:
        return pd.DataFrame(), None

    catalog_copy = catalog_df.copy()
    matching_mask = catalog_copy[genre_col].apply(
        lambda genres: isinstance(genres, list) and top_genre in genres
    )
    candidate_titles = catalog_copy[matching_mask]

    segment_users = user_segments_df[user_segments_df["segment"] == segment_name]["user_id"]
    segment_interactions = interactions_df[interactions_df["user_id"].isin(segment_users)]

    title_completion = (
        segment_interactions.groupby("show_id")["completion_rate"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "segment_avg_completion", "count": "segment_watch_count"})
    )

    ranked = candidate_titles.merge(title_completion, on="show_id", how="inner")
    ranked = ranked.sort_values("segment_avg_completion", ascending=False)

    return ranked.head(top_n)[["show_id", "title", genre_col, "segment_avg_completion", "segment_watch_count"]], top_genre


def build_tfidf_matrix(catalog_df, description_col="description", genre_col="listed_in"):
   
    combined_text = (
        catalog_df[description_col].fillna("") + " " + catalog_df[genre_col].fillna("")
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",  
        max_features=5000,     
    )
    tfidf_matrix = vectorizer.fit_transform(combined_text)

    return tfidf_matrix, vectorizer


def get_similar_titles(show_id, catalog_df, tfidf_matrix, top_n=5):
    
    idx_lookup = catalog_df.reset_index(drop=True)
    match = idx_lookup[idx_lookup["show_id"] == show_id]
    if match.empty:
        return pd.DataFrame()
    title_idx = match.index[0]

    sim_scores = cosine_similarity(tfidf_matrix[title_idx], tfidf_matrix).flatten()

    similar_indices = sim_scores.argsort()[::-1]
    similar_indices = similar_indices[similar_indices != title_idx][:top_n]

    result = idx_lookup.iloc[similar_indices][["show_id", "title", "listed_in"]].copy()
    result["similarity_score"] = sim_scores[similar_indices]
    return result


def recommend_for_segment_by_similarity(segment_name, interactions_df, user_segments_df,
                                         catalog_df, tfidf_matrix, top_n=5):
    
    segment_users = user_segments_df[user_segments_df["segment"] == segment_name]["user_id"]
    segment_interactions = interactions_df[interactions_df["user_id"].isin(segment_users)]

    title_stats = (
        segment_interactions.groupby("show_id")["completion_rate"]
        .agg(["mean", "count"])
        .query("count >= 20")
        .sort_values("mean", ascending=False)
    )

    if title_stats.empty:
        return pd.DataFrame(), None

    seed_show_id = title_stats.index[0]
    seed_title_row = catalog_df[catalog_df["show_id"] == seed_show_id]
    seed_title_name = seed_title_row["title"].values[0] if not seed_title_row.empty else seed_show_id

    recommendations = get_similar_titles(seed_show_id, catalog_df, tfidf_matrix, top_n=top_n)

    return recommendations, seed_title_name