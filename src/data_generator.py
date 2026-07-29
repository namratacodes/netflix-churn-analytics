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
        "session_lambda": 10, "decay": 0.72,   # steepened from 0.85 -- 0.85 rarely produced 30+ day silence
        "preferred_genres": ["Crime TV Shows", "TV Dramas", "Thrillers"],  # deliberately overlaps power_binger
        "tenure_range": (90, 500),  # widened lower bound so more users have time to decay fully
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


def derive_features_from_sessions(session_dates, window_days=90, observation_days=60):
    """
    Convert raw session history into modeling features, with STRICT
    temporal separation between the observation window (used for features)
    and the outcome window (used only for the churn label).

    LEAKAGE FIX: the previous version compared "last 30 days" vs "prior 30
    days" of the FULL 90-day window -- but the churn label is ALSO defined
    from that same last-30-day period, so the feature leaked the label
    almost exactly (recent_30d_sessions == 0 was structurally ~equivalent
    to churned == 1, giving a suspicious 0.9976 ROC-AUC).

    New design:
      - observation window = days [0, observation_days)  -- e.g. [0, 60)
      - outcome window      = days [observation_days, window_days) -- [60, 90)
      - ALL features come only from the observation window
      - the label (churned_next_30d) comes only from the outcome window
    This mirrors real-world churn modeling: predict a FUTURE outcome from
    PAST behavior only, never let the outcome period leak into features.
    """
    obs_sessions = [d for d in session_dates if d < observation_days]
    outcome_sessions = [d for d in session_dates if d >= observation_days]

    # label: did the user come back AT ALL during the outcome window?
    churned_next_30d = 1 if len(outcome_sessions) == 0 else 0

    if len(obs_sessions) == 0:
        return {
            "obs_total_sessions": 0,
            "obs_avg_sessions_per_week": 0.0,
            "obs_recent_30d_sessions": 0,
            "obs_prior_30d_sessions": 0,
            "obs_session_trend_ratio": 0.0,
            "obs_days_since_last_watch": observation_days,
            "churned_next_30d": churned_next_30d,
        }

    # trend computed ENTIRELY inside the observation window:
    # recent half = days [30, 60), prior half = days [0, 30) -- both safely
    # before the outcome window, so neither can leak the label.
    recent_cutoff = observation_days - 30  # = 30
    obs_recent_30d_sessions = sum(1 for d in obs_sessions if d >= recent_cutoff)
    obs_prior_30d_sessions = sum(1 for d in obs_sessions if d < recent_cutoff)

    if obs_prior_30d_sessions == 0:
        obs_session_trend_ratio = 1.0 if obs_recent_30d_sessions == 0 else 2.0
    else:
        obs_session_trend_ratio = obs_recent_30d_sessions / obs_prior_30d_sessions

    last_obs_session = max(obs_sessions)
    # recency measured relative to the END of the observation window (day 59),
    # NOT relative to "today" (day 89) -- measuring from day 89 would leak,
    # since it would implicitly reveal whether sessions occurred afterward.
    obs_days_since_last_watch = (observation_days - 1) - last_obs_session

    return {
        "obs_total_sessions": len(obs_sessions),
        "obs_avg_sessions_per_week": len(obs_sessions) / (observation_days / 7),
        "obs_recent_30d_sessions": obs_recent_30d_sessions,
        "obs_prior_30d_sessions": obs_prior_30d_sessions,
        "obs_session_trend_ratio": obs_session_trend_ratio,
        "obs_days_since_last_watch": obs_days_since_last_watch,
        "churned_next_30d": churned_next_30d,
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


def build_user_dataset(n_users=5000, seed=42, window_days=90, observation_days=60):
    # single Generator instance, threaded through every call, instead of
    # mixing np.random.seed(...) globally with default_rng in one place.
    rng = np.random.default_rng(seed)

    assigned = assign_personas(n_users, personas=PERSONAS, seed=seed)

    rows = []

    for user_id, persona in enumerate(assigned, start=1):
        params = PERSONAS[persona]

        tenure_days = int(rng.integers(params["tenure_range"][0], params["tenure_range"][1] + 1))

        sessions = simulate_session_history(
            params, rng, window_days=window_days, tenure_days=tenure_days
        )

        features = derive_features_from_sessions(
            sessions, window_days=window_days, observation_days=observation_days
        )

        # IMPORTANT: completion rate is generated from obs_total_sessions only,
        # NOT the full session count -- using sessions from the outcome window
        # here would reintroduce a (smaller) leak, since churned users have
        # zero outcome-window sessions by definition, subtly shifting their
        # completion rate distribution relative to non-churned users for
        # reasons unrelated to genuine viewing behavior.
        completion_rate = generate_completion_rates(params, features["obs_total_sessions"], rng)
        diversity, favorite_genre = generate_genre_diversity(params, rng)

        rows.append({
            "user_id": user_id,
            "persona": persona,  # hidden ground truth -- keep for validation, don't feed to models
            "tenure_days": tenure_days,
            "obs_total_sessions": features["obs_total_sessions"],
            "obs_avg_sessions_per_week": features["obs_avg_sessions_per_week"],
            "obs_recent_30d_sessions": features["obs_recent_30d_sessions"],
            "obs_prior_30d_sessions": features["obs_prior_30d_sessions"],
            "obs_session_trend_ratio": features["obs_session_trend_ratio"],
            "obs_days_since_last_watch": features["obs_days_since_last_watch"],
            "avg_completion_rate": completion_rate,
            "genre_diversity": diversity,
            "favorite_genre": favorite_genre,
            "churned_next_30d": features["churned_next_30d"],  # the label -- from outcome window only
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_user_dataset()
    print(df.head())
    print("\nSanity check -- obs_days_since_last_watch range per persona:")
    print(df.groupby("persona")["obs_days_since_last_watch"].agg(["min", "max", "mean"]))
    print("\nChurn rate overall and by persona:")
    print(df["churned_next_30d"].mean())
    print(df.groupby("persona")["churned_next_30d"].mean())

    df.to_csv("data/processed/synthetic_viewing_behavior.csv", index=False)