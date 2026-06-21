import time

from footy.tournament.results import load_results
from footy.tournament.simulator import simulate_tournaments
from footy.tournament.aggregate import aggregate
from footy.live.ingest import ingest
from footy.live.scoreboard import scoreboard


class TournamentRunner:
    """Provider-agnostic orchestrator: ingest -> simulate -> scoreboard."""

    def __init__(self, structure, name_map, stage_map, results_path, sampler, predictor, n, seed):
        self.structure = structure
        self.name_map = name_map
        self.stage_map = stage_map
        self.results_path = results_path
        self.sampler = sampler
        self.predictor = predictor
        self.n = n
        self.seed = seed

    def cycle(self, provider) -> dict:
        n_written = ingest(provider, self.structure, self.name_map, self.stage_map, self.results_path)
        results = load_results(self.results_path, self.structure.groups)
        sims = simulate_tournaments(self.structure, results, self.sampler, self.n, self.seed)
        agg = aggregate(self.structure, sims)
        played_dicts = [
            {"team_a": pm.team_a, "team_b": pm.team_b,
             "goals_a": pm.goals_a, "goals_b": pm.goals_b}
            for pm in results.played
        ]
        board = scoreboard(self.predictor, played_dicts)
        return {"played": n_written, "aggregate": agg, "scoreboard": board,
                "meta": {"n_tournaments": self.n, "seed": self.seed,
                         "results_path": str(self.results_path)}}


def watch(runner, provider, interval_minutes, emit):
    """Run cycle -> emit -> sleep, forever, until KeyboardInterrupt."""
    interval_seconds = interval_minutes * 60
    try:
        while True:
            emit(runner.cycle(provider))
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        return
