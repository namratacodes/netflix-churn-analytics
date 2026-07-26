import numpy as np
import pandas as pd

GENRES = [
    "International Movies", "Dramas", "Comedies", "International TV Shows",
    "Documentaries", "Action & Adventure", "TV Dramas", "Independent Movies",
    "Children & Family Movies", "Romantic Movies", "TV Comedies", "Thrillers",
    "Crime TV Shows", "Kids' TV", "Docuseries", "Music & Musicals",
    "Romantic TV Shows", "Horror Movies", "Stand-Up Comedy", "Reality TV",
    "British TV Shows", "Sci-Fi & Fantasy", "Sports Movies"
]

PERSONAS = {
    "power_binger": {
        "weight": 0.20,
        "completion_mean": 0.85, "completion_conc": 10,
        "session_lambda": 20, "decay": 1.0,
        "preferred_genres": ["Crime TV Shows", "TV Dramas", "Thrillers", "Sci-Fi & Fantasy"],
        "tenure_range": (180, 730),  # established users, 6mo-2yr
    },
    "casual_viewer": {
        "weight": 0.30,
        "completion_mean": 0.60, "completion_conc": 6,
        "session_lambda": 8, "decay": 0.98,
        "preferred_genres": ["Comedies", "Romantic Movies", "Documentaries"],
        "tenure_range": (90, 730),
    },
    "churning_user": {
        "weight": 0.20,
        "completion_mean": 0.30, "completion_conc": 5,
        "session_lambda": 10, "decay": 0.85,   # decay is what creates churn signal
        "preferred_genres": ["Crime TV Shows", "TV Dramas", "Thrillers"],  # deliberately overlaps power_binger
        "tenure_range": (60, 500),
    },
    "weekend_watcher": {
        "weight": 0.20,
        "completion_mean": 0.70, "completion_conc": 8,
        "session_lambda": 6, "decay": 1.0,
        "preferred_genres": ["Children & Family Movies", "Kids' TV", "Romantic Movies", "Stand-Up Comedy"],
        "tenure_range": (90, 730),
    },
    "new_user": {
        "weight": 0.10,
        "completion_mean": 0.50, "completion_conc": 2,
        "session_lambda": 5, "decay": 1.0,
        "preferred_genres": GENRES,  # samples broadly, no fixed preference
        "tenure_range": (1, 30),  # this is what makes them "new"
    },
}


def assign_personas(n_users, personas=PERSONAS, seed=42):
    """Assign personas according to configured weights."""
    rng = np.random.default_rng(seed)

    persona_names = list(personas.keys())
    weights = np.array([personas[p]["weight"] for p in persona_names])

    assigned = rng.choice(
        persona_names,
        size=n_users,
        p=weights / weights.sum()
    )

    return assigned


def simulate_session_history(persona_params, rng, window_days=90, tenure_days=None):
    """
    Simulate viewing sessions over the last `window_days`, day by day.

    FIX #1 (off-by-one): the old version looped `range(window_days // 7)`,
    which truncates 90 // 7 = 12 weeks = 84 days -- silently dropping the
    most recent 6 days from ever being sampled. Simulating day-by-day
    removes this entire class of bug; every day in [0, window_days) has a
    chance of a session, including "today".

    FIX #2 (tenure): a user can only have sessions after they joined.
    If tenure_days < window_days, days before the join day are simply not
    simulated -- so a new_user with tenure_days=10 can only ever show
    session activity in the last 10 days of the window, not all 90.
    """
    decay = persona_params["decay"]
    weekly_lambda = persona_params["session_lambda"]
    daily_lambda_base = weekly_lambda / 7.0

    effective_start_day = window_days - tenure_days if tenure_days is not None else 0
    effective_start_day = max(0, effective_start_day)

    session_days = []

    for day in range(effective_start_day, window_days):
        week_index = day // 7
        expected_sessions_today = daily_lambda_base * (decay ** week_index)
        n_sessions_today = rng.poisson(expected_sessions_today)

        # a user can watch more than once a day; record each as a session event
        session_days.extend([day] * n_sessions_today)

    return sorted(session_days)


def derive_features_from_sessions(session_dates, window_days=90):
    """Convert raw session history into modeling features."""
    if len(session_dates) == 0:
        return {
            "days_since_last_watch": window_days,
            "total_sessions": 0,
            "avg_sessions_per_week": 0.0,
        }

    last_session = max(session_dates)
    days_since_last_watch = (window_days - 1) - last_session
    total_sessions = len(session_dates)
    avg_sessions_per_week = total_sessions / (window_days / 7)

    return {
        "days_since_last_watch": days_since_last_watch,
        "total_sessions": total_sessions,
        "avg_sessions_per_week": avg_sessions_per_week,
    }


def generate_completion_rates(persona_params, n_sessions, rng):
    """Generate average completion rate from a Beta distribution."""
    if n_sessions == 0:
        return 0.0

    mean = persona_params["completion_mean"]
    conc = persona_params["completion_conc"]

    alpha = mean * conc
    beta = (1 - mean) * conc

    completions = rng.beta(alpha, beta, size=n_sessions)
    return completions.mean()


def generate_genre_diversity(persona_params, rng):
    """Simulate genre diversity and favorite genre."""
    preferred = persona_params["preferred_genres"]

    if preferred == GENRES:
        favorite = rng.choice(GENRES)
        diversity = rng.integers(8, 16)
    else:
        favorite = rng.choice(preferred)
        diversity = rng.integers(len(preferred), min(len(GENRES), len(preferred) + 6))

    return int(diversity), favorite


def build_user_dataset(n_users=5000, seed=42, window_days=90):
    # single Generator instance, threaded through every call, instead of
    # mixing np.random.seed(...) globally with default_rng in one place.
    # This makes the whole pipeline reproducible from one seed and avoids
    # the inconsistency flagged earlier (legacy global API vs Generator API).
    rng = np.random.default_rng(seed)

    assigned = assign_personas(n_users, personas=PERSONAS, seed=seed)

    rows = []

    for user_id, persona in enumerate(assigned, start=1):
        params = PERSONAS[persona]

        tenure_days = int(rng.integers(params["tenure_range"][0], params["tenure_range"][1] + 1))

        sessions = simulate_session_history(
            params, rng, window_days=window_days, tenure_days=tenure_days
        )

        features = derive_features_from_sessions(sessions, window_days=window_days)
        completion_rate = generate_completion_rates(params, len(sessions), rng)
        diversity, favorite_genre = generate_genre_diversity(params, rng)

        rows.append({
            "user_id": user_id,
            "persona": persona,  # hidden ground truth -- keep for validation, don't feed to models
            "tenure_days": tenure_days,
            "days_since_last_watch": features["days_since_last_watch"],
            "total_sessions": features["total_sessions"],
            "avg_sessions_per_week": features["avg_sessions_per_week"],
            "avg_completion_rate": completion_rate,
            "genre_diversity": diversity,
            "favorite_genre": favorite_genre,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_user_dataset()
    print(df.head())
    print("\nSanity check -- days_since_last_watch range per persona:")
    print(df.groupby("persona")["days_since_last_watch"].agg(["min", "max", "mean"]))

    df.to_csv("data/processed/synthetic_viewing_behavior.csv", index=False)