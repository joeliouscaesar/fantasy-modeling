import altair as alt
import nflreadpy as nfl # type: ignore
import polars as pl
from scipy.special import gamma
import numpy as np

# makes it so plots show up in browser
alt.renderers.enable("browser")
alt.data_transformers.enable("default")

# pull jordy details
jordy = nfl.load_players().filter((pl.col("first_name") == "Jordy") & (pl.col("last_name") == "Nelson"))

# download jordy's stats by season
first_season = jordy["rookie_season"].first() # type:ignore
last_season = jordy["last_season"].first()  # type:ignore
jordy_id = jordy["gsis_id"].first()
season_range = list(range(first_season, last_season+1)) # type:ignore

id_predicate = pl.col("player_id") == jordy_id
jordy_stats = nfl.load_player_stats(season_range).filter(id_predicate)

jordy_stats["fantasy_points"].plot.hist().show()

# this looks kind of exponential, so we'll practice with that 
jordy_stats = jordy_stats.with_columns(
    pl.col("fantasy_points").clip(0).alias("fantasy_points_clipped")
)
jordy_stats["fantasy_points_clipped"].plot.hist().show()
ys = jordy_stats["fantasy_points_clipped"]

# so first thing is we need is the (conjugate) prior distribution for lambda, which will be a gamma distribution

# update rule for a given observation y is 

# lambda ~ Gamma (alpha, beta), lambda|y ~ Gamma (alpha + 1, beta + y)

# appears alpha/beta is the mean in this specification, so we'll start with beta = 1 / fantasy_points_clipped.mean() and alpha = 1
alpha = 1
beta = 1 / ys.mean()

# bayesian updates!
for y in ys:
    alpha += 1
    beta += y


# sample from the posterior Gamma(alpha, beta) and compare it against the observed data
def sample_gamma(alpha: float, beta: float, n: int) -> np.ndarray:
    # numpy parameterizes gamma by (shape, scale); we have (shape=alpha, rate=beta)
    rng = np.random.default_rng()
    return rng.gamma(shape=alpha, scale=1 / beta, size=n)

gamma_samples = sample_gamma(alpha, beta, 100)

# for each of these gammas, sample exponential
def sample_exp(lamb : float, n: int) -> np.ndarray:
    # numpy parameterizes gamma by (shape, scale); we have (shape=alpha, rate=beta)
    rng = np.random.default_rng()
    return rng.exponential(scale=1/lamb, size=n)

# for each gamma in our gamma samples, sample lots of exponential values
exp_samples = [sample_exp(gamma, 1000) for gamma in gamma_samples]
exp_samples_flat = np.concatenate(exp_samples)

# layered histogram comparing observed (clipped) fantasy points against the posterior
# predictive exponential samples, normalized to % of observations within each group
# since the sample has many more values than the observed data
hist_df = pl.concat([
    ys.to_frame().rename({"fantasy_points_clipped": "value"}).with_columns(pl.lit("observed").alias("source")),
    pl.DataFrame({"value": exp_samples_flat}).with_columns(pl.lit("posterior predictive").alias("source")),
])

observed = hist_df.filter(pl.col("source") == "observed").select(pl.col("value")).to_series().hist(bins=list(range(0,100,4)))
observed = observed.with_columns(
    (pl.col("count") / pl.col("count").sum()).alias("frac"),
    pl.lit("observed").alias("source")
)

sampled = hist_df.filter(pl.col("source") == "posterior predictive").select(pl.col("value")).to_series().hist(bins=list(range(0,100,4)))
sampled = sampled.with_columns(
    (pl.col("count") / pl.col("count").sum()).alias("frac"),
    pl.lit("posterior predictive").alias("source")
)

to_plot = pl.concat([observed, sampled])

overlay_hist = (
    alt.Chart(to_plot)
    .mark_bar(opacity=0.6, tooltip=True)
    .encode(
        x=alt.X("category:N", title="Fantasy Points", sort=alt.SortField(field="breakpoint", order="ascending")),
        y=alt.Y("frac:Q", stack=None, title="% of observations", axis=alt.Axis(format="%")),
        color=alt.Color("source:N", title=None),
    )
)
overlay_hist.show()

alpha / beta
