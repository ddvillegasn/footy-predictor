from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from footy.config import load_config, config_fingerprint
from footy.data.loaders import load_results, load_former_names
from footy.data.clean import clean_results
from footy.data.names import NameCanonicalizer
from footy.predict import Predictor
from footy.tournament.structure import load_structure
from footy.tournament.sampler import MatchSampler
from footy.live.provider import FootballDataProvider
from footy.live.name_map import load_name_map
from footy.live.runner import TournamentRunner, watch


def parse_book_odds(text: str) -> dict:
    """Parse "1x2.home=1.45,over_under.2.5.over=1.67" into a nested dict.

    Each entry is dotted-path=decimal. 2-level paths (market.outcome) and
    3-level paths (market.line.side) are supported. Handles decimal lines
    like "2.5" by recognizing patterns where middle parts are all numeric.
    """
    result: dict = {}
    if not text:
        return result
    for entry in text.split(","):
        entry = entry.strip()
        if not entry:
            continue
        path, _, raw_value = entry.partition("=")
        if not raw_value:
            raise ValueError(f"Malformed --book-odds entry {entry!r}; expected path=odds")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Bad odds in {entry!r}: {raw_value!r} is not a number") from exc

        parts = path.split(".")
        node = result

        # Reconstruct nested dict, recognizing numeric line patterns
        if len(parts) == 2:
            # market.outcome (e.g., "1x2.home")
            market = parts[0]
            outcome = parts[1]
            node = node.setdefault(market, {})
            node[outcome] = value
        elif len(parts) == 3:
            # market.outcome or market.line.side
            # If middle part is numeric+dot-able, it's a line; otherwise market.outcome.sub
            market, middle, last = parts[0], parts[1], parts[2]
            node = node.setdefault(market, {})
            # Try to guess: if middle looks numeric, it might be a line component
            # For now, treat 3-part as market.outcome.detail
            node = node.setdefault(middle, {})
            node[last] = value
        elif len(parts) == 4:
            # market.line_part1.line_part2.side (e.g., over_under.2.5.over)
            market, part2, part3, side = parts[0], parts[1], parts[2], parts[3]
            line = part2 + "." + part3  # Rejoin numeric line
            node = node.setdefault(market, {})
            node = node.setdefault(line, {})
            node[side] = value

    return result


def _build_default_predictor() -> Predictor:
    data_cfg = load_config("data")
    model_cfg = load_config("model")
    mc_cfg = load_config("montecarlo")
    bet_cfg = load_config("betting")

    raw_dir = data_cfg["raw_dir"]
    results = load_results(f"{raw_dir}/{data_cfg['files']['results']}")
    former = load_former_names(f"{raw_dir}/{data_cfg['files']['former_names']}")
    clean = clean_results(results)

    canon = NameCanonicalizer(
        former, data_cfg.get("aliases", {}), data_cfg.get("sensitive_merges", {})
    )
    as_of = clean.df["date"].max() + pd.Timedelta(days=1)
    return Predictor.from_matches(
        clean.df, model_config=model_cfg, mc_config=mc_cfg,
        canonical=canon.canonical, as_of=as_of,
        betting_config=bet_cfg, betting_config_version=config_fingerprint("betting"),
    )


def run(argv: list[str], predictor: Predictor | None = None) -> int:
    parser = argparse.ArgumentParser(prog="predict", description="Predict a national-team match.")
    parser.add_argument("team_a")
    parser.add_argument("team_b")
    parser.add_argument("--neutral", action="store_true", help="Neutral venue (home_advantage=0)")
    parser.add_argument("--tournament", default="Friendly")
    parser.add_argument("--markets", action="store_true", help="Include betting markets")
    parser.add_argument("--book-odds", default=None,
                        help='Bookmaker odds, e.g. "1x2.home=1.45,1x2.draw=4.2"')
    args = parser.parse_args(argv)

    if predictor is None:
        predictor = _build_default_predictor()

    book_odds = parse_book_odds(args.book_odds) if args.book_odds else None
    include_markets = args.markets or book_odds is not None

    try:
        result = predictor.predict(
            args.team_a, args.team_b, neutral=args.neutral, tournament=args.tournament,
            include_markets=include_markets, book_odds=book_odds,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def main() -> None:
    sys.exit(run(sys.argv[1:]))


def _build_runner() -> TournamentRunner:
    data_cfg = load_config("data")
    model_cfg = load_config("model")
    mc_cfg = load_config("montecarlo")
    live_cfg = load_config("live")
    sim_cfg = load_config("tournament_sim")
    if not isinstance(live_cfg.get("stage_map"), dict):
        raise ValueError("configs/live.yaml: 'stage_map' must be a mapping")

    base = "configs/tournaments"
    structure = load_structure(f"{base}/wc2026.yaml")
    predictor = _build_default_predictor()
    canon = predictor.canonical
    for g, teams in structure.groups.items():
        structure.groups[g] = [canon(t) for t in teams]

    sampler = MatchSampler(predictor.model, mc_cfg,
                           model_version=model_cfg["model_version"],
                           config_hash=config_fingerprint("montecarlo"))
    name_map = load_name_map("configs/name_map.yaml")
    return TournamentRunner(
        structure, name_map, live_cfg["stage_map"], f"{base}/wc2026_results.yaml",
        sampler, predictor, n=int(sim_cfg["n_tournaments"]), seed=int(sim_cfg["seed"]))


def _format_summary(result: dict) -> str:
    agg = result["aggregate"]; board = result["scoreboard"]
    champs = sorted(agg["teams"].items(), key=lambda kv: -kv[1]["champion"])[:8]
    lines = [f"Played matches: {result['played']}",
             f"Scoreboard: n={board['n']} accuracy={board['accuracy']} "
             f"log_loss={board['log_loss']} brier={board['brier']} goal_mae={board['goal_mae']}",
             "Top champion probabilities:"]
    for team, d in champs:
        lines.append(f"  {team:<18} {d['champion'] * 100:5.1f}%")
    return "\n".join(lines)


def run_update(argv, runner=None, provider=None, emit=None) -> int:
    parser = argparse.ArgumentParser(prog="update-and-simulate",
                                     description="Fetch official results and re-simulate the tournament.")
    parser.add_argument("--watch", type=int, default=None, metavar="MINUTES",
                        help="Re-run every MINUTES minutes")
    parser.add_argument("--json", action="store_true", help="Emit full JSON")
    args = parser.parse_args(argv)

    if runner is None:
        runner = _build_runner()
    if provider is None:
        provider = FootballDataProvider.from_config(load_config("live"))

    if emit is None:
        if args.json:
            emit = lambda result: print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            emit = lambda result: print(_format_summary(result))

    if args.watch is not None:
        watch(runner, provider, args.watch, emit)
    else:
        emit(runner.cycle(provider))
    return 0


def main_update() -> None:
    sys.exit(run_update(sys.argv[1:]))
