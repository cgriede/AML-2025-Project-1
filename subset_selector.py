# --------------------------------------------------------------
#  random_subset_selection.py
# --------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.metrics import check_scoring
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from typing import Tuple, List
from scipy.stats import binomtest

class RandomSubsetSelector:
    """
    Monte-Carlo feature selection:
      1) Sample many random subsets.
      2) Evaluate each with CV.
      3) Keep features that appear most often in the best-scoring subsets.
    """

    def __init__(
        self,
        estimator,
        n_trials: int = 500,
        subset_frac: float = 0.7,
        cv: int = 5,
        scoring: str = "r2",
        top_k: int = 50,
        min_features: int = 1,
        random_state: int | None = None,
        n_jobs: int = -1,
        verbose: bool = False,
        max_samples: int | None = 400,
        alpha: float = 0.05,
    ):
        self.estimator = estimator
        self.n_trials = n_trials
        self.subset_frac = subset_frac
        self.cv = cv
        self.scoring = scoring
        self.top_k = top_k
        self.min_features = min_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.max_samples = max_samples
        self.verbose = verbose
        self.alpha = alpha
        self.rng = np.random.default_rng(random_state)
        self.scores_ = []          # (score, subset) pairs
        self.selected_features_ = None
        self.feature_counts_ = None
        self.X_original_ = pd.DataFrame()

    # ----------------------------------------------------------
    def _evaluate_subset(self, X, y, subset) -> Tuple[float, List[int]]:
        """Train+CV on a single subset (executed in parallel)."""
        model = clone(self.estimator)
        scorer = check_scoring(model, scoring=self.scoring)
        cv_scores = cross_val_score(
            model, X.iloc[:, subset], y, cv=self.cv, scoring=scorer, n_jobs=1
        )
        return cv_scores.mean(), subset

    # ----------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series, feature_names: List[str] | None = None):
        """
        Run the Monte-Carlo search.
        X : array (n_samples, n_features)
        y : array (n_samples,)
        feature_names : optional list for pretty output
        """

        if self.max_samples is not None and X.shape[0] > self.max_samples:
            if self.verbose:
                print(f"Subsampling to {self.max_samples} samples for speed.")
            sample_idx = self.rng.choice(X.shape[0], size=self.max_samples, replace=False)
            X = X.iloc[sample_idx, :].reset_index(drop=True)
            y = y.iloc[sample_idx].reset_index(drop=True)

        n_features = X.shape[1]
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(n_features)]

        subset_size = max(self.min_features, int(n_features * self.subset_frac))

        self.scaler = StandardScaler()
        X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        y = pd.Series(self.scaler.fit_transform(y.values.reshape(-1, 1)).flatten())

        # ---- 1. generate random masks --------------------------------
        masks = [
            sorted(self.rng.choice(n_features, size=subset_size, replace=False))
            for _ in range(self.n_trials)
        ]
        if self.verbose:
            print(f"Generated {len(masks)} random subsets of size {subset_size}.")
        # ---- 2. evaluate in parallel ---------------------------------
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._evaluate_subset)(X, y, mask) for mask in masks
        )
        if self.verbose:
            print("Completed evaluations.")

        # ---- 3. store (score, feature_list) --------------------------
        self.scores_ = [(score, feats) for score, feats in results]

        # ---- 4. keep top-k performing subsets -------------------------
        self.scores_.sort(key=lambda x: x[0], reverse=True)

        freq = np.zeros(n_features, dtype=int)
        score_sum = np.zeros(n_features, dtype=float)
        for score, feats in self.scores_[: self.top_k]:
        # ---- 5. count feature frequency in top subsets ----------------
            for feat in feats:
                freq[feat] += 1
                score_sum[feat] += score

        avg_score = np.divide(score_sum, freq, out=np.zeros_like(score_sum), where=freq != 0)

        # Store as DataFrame for easier access
        self.feature_counts_ = pd.DataFrame({
            'times_in_top_k': freq,
            'avg_score': avg_score
        }, index=feature_names).sort_values('times_in_top_k', ascending=False)


        # ---- 6. decide final feature set: keep statistically significant features -------------------------------
        alpha = self.alpha  # Significance threshold; adjust as needed
        p_values = []

        for count in self.feature_counts_['times_in_top_k']:
            p_val = binomtest(count, n=self.top_k, p=self.subset_frac, alternative='greater').pvalue
            p_values.append(p_val)

        # Add p_values to feature_counts_
        self.feature_counts_['p_value'] = p_values

        # Select features with p_value < alpha
        self.selected_features_ = self.feature_counts_[self.feature_counts_['p_value'] < alpha].index.tolist()

        if self.verbose:
            print(f"Selected {len(self.selected_features_)} features (p < {alpha}):")
            print(self.selected_features_)

        return self


    # ----------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return X restricted to the selected columns."""
        if self.verbose:
            print("Transforming dataset to selected features...")
        if self.selected_features_ is None:
            raise RuntimeError("fit() must be called first")
        return self.X_original_[self.selected_features_]

    # ----------------------------------------------------------
    def fit_transform(self, X, y, **kwargs):
        self.X_original_ = X.copy()
        if self.verbose:
            print("Fitting and transforming in one step...")
        return self.fit(X, y, **kwargs).transform(X)

    # ----------------------------------------------------------
    @property
    def summary(self) -> pd.DataFrame:
        """Nice table: feature | times_in_top_k | selected?"""
        df = self.feature_counts_.copy()
        df["selected"] = df.index.isin(self.selected_features_)
        return df.sort_values("times_in_top_k", ascending=False)
    

"""

#use like this
from subset_selector import RandomSubsetSelector
from xgboost import XGBRegressor

alpha = 0.15
selector = RandomSubsetSelector(
    estimator=XGBRegressor(
        n_estimators=100,      # Number of trees; start low for speed in subset selection
        max_depth=4,           # Tree depth; controls complexity
        learning_rate=0.05,     # Step size; balances speed and accuracy
        subsample=0.6,         # Fraction of samples per tree; adds randomness
        colsample_bytree=0.6,  # Fraction of features per tree; helps with high-dimensional data
        random_state=42,       # For reproducibility
        n_jobs=-1              # Use all cores
    ),
    n_trials=1200,          # more trials → better coverage
    subset_frac=0.02,
    cv=4,
    top_k=400,
    n_jobs=-1,
    verbose=True,
    random_state=42,
    max_samples=300,
    alpha=alpha
)

def load_cached_features(file: Path) -> pd.DataFrame:
    with open(filename, "r") as f:
        cols = [line.strip() for line in f.readlines()]
    return X_train_cleaned[cols]

load_cached = False
filename = None
X_selected = None
if not load_cached:
    X_selected = selector.fit_transform(X_train_cleaned, y_train_cleaned, feature_names=X_train_cleaned.columns)
    filename = Path("feature_selection") / f"sc_alpha{alpha*100:.0f}_{len(selector.selected_features_)}f.csv"
    with open(filename, "w") as f:
        for feature in selector.selected_features_:
            f.write(f"{feature}\n")
    print("\n=== Top-10 most frequent features in best subsets ===")
    print(selector.summary.head(20))
    print(f"Selected feature set shape: {X_selected.shape}")
else:
    filename = Path("feature_selection").iterdir().filter(lambda f: f.name.startswith(f"alpha")).__next__()
    X_selected = load_cached_features(filename)
    
"""
