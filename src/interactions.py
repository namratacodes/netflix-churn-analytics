import numpy as np
import pandas as pd


def build_title_weights(catalog_df, preferred_genres, genre_col="genre_list", boost=5.0):
    
    preferred_set = set(preferred_genres)

    def title_weight(genre_list):
        # genre_list might be NaN/float if listed_in was ever empty --
        # guard defensively even though we confirmed 0 nulls earlier.
        if not isinstance(genre_list, list):
            return 1.0
        return boost if any(g in preferred_set for g in genre_list) else 1.0

    weights = catalog_df[genre_col].apply(title_weight).to_numpy()
    return weights


def sample_titles_for_user(catalog_df, weights, n_titles, rng):
    
    n_available = len(catalog_df)
    n_to_sample = min(n_titles, n_available)

    if n_to_sample == 0:
        return catalog_df.iloc[0:0]  # empty dataframe, same columns

    probabilities = weights / weights.sum()

    sampled_indices = rng.choice(
        n_available,
        size=n_to_sample,
        replace=False,
        p=probabilities,
    )

    return catalog_df.iloc[sampled_indices]


def build_interaction_table(user_df, catalog_df, personas, rng, genre_col="genre_list"):
    
    all_rows = []

    for _, user_row in user_df.iterrows():
        persona_name = user_row["persona"]
        persona_params = personas[persona_name]
        n_sessions = int(user_row["total_sessions"])

        if n_sessions == 0:
            continue  # this user never watched anything in the window -- no rows to add

        weights = build_title_weights(
            catalog_df,
            persona_params["preferred_genres"],
            genre_col=genre_col,
        )

        sampled_titles = sample_titles_for_user(catalog_df, weights, n_sessions, rng)

        mean = persona_params["completion_mean"]
        conc = persona_params["completion_conc"]
        alpha = mean * conc
        beta = (1 - mean) * conc
        completion_rates = rng.beta(alpha, beta, size=len(sampled_titles))

        for (_, title_row), completion_rate in zip(sampled_titles.iterrows(), completion_rates):
            all_rows.append({
                "user_id": user_row["user_id"],
                "persona": persona_name,
                "show_id": title_row["show_id"],
                "title": title_row["title"],
                "genre_list": title_row[genre_col],
                "completion_rate": completion_rate,
            })

    return pd.DataFrame(all_rows)