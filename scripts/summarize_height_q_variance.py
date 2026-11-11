"""Audit and summarize saved fixed-state batches without rerunning physics."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import torch

from learning.offline.fixed_state_variance import summarize_returns


def distribution(x):
    x = x.double().flatten()
    return {
        "mean": float(x.mean()),
        "p10": float(x.quantile(0.1)),
        "median": float(x.median()),
        "p90": float(x.quantile(0.9)),
        "max": float(x.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    directory = parser.parse_args().directory
    insertion = torch.load(directory / "insertion.pt", weights_only=True)
    batches = [
        torch.load(directory / f"batch_{b}.pt", weights_only=True) for b in range(8)
    ]
    seeds = [seed for batch in batches for seed in batch["seeds"]]
    assert len(set(seeds)) == 128
    assert [b["batch"] for b in batches] == list(range(8))
    x = torch.stack([b["returns"] for b in batches])
    first = torch.stack([b["first_rewards"] for b in batches])
    survived = torch.stack([b["survived"] for b in batches])
    assert x.shape == (8, 16, 128, 8)
    assert torch.isfinite(x).all() and ((first == 0) | (first == -1)).all()
    assert (x >= -100.000001).all() and (x <= 0).all()
    tail = 0.99**500 / (1 - 0.99)
    lower = x - tail * survived.to(x.dtype)
    raw = {
        "axes": ["batch", "repeat", "state", "candidate"],
        "returns": x,
        "return_lower": lower,
        "return_upper": x.clone(),
        "first_rewards": first,
        "survived": survived,
        "seeds": torch.tensor(seeds).reshape(8, 16),
    }
    torch.save(raw, directory / "raw_returns.pt")
    stats = summarize_returns(x)
    torch.save(stats, directory / "statistics.pt")
    mean = x.mean((0, 1))
    mean_lower = lower.mean((0, 1))
    paired = x - x[..., :1]
    paired_lower = lower - x[..., :1]
    paired_upper = x - lower[..., :1]
    paired_lower[..., 0] = 0
    paired_upper[..., 0] = 0
    report = {
        "protocol": {
            "axes": raw["axes"],
            "shape": list(x.shape),
            "k": 100,
            "H": 500,
            "gamma": 0.99,
            "height_band": [0.27, 0.37],
            "candidate0": "insertion observation policy mean",
            "candidate1_to_7": (
                "mean plus independent learned-std Gaussian perturbations"
            ),
        },
        "batch_seconds": [b["seconds"] for b in batches],
        "total_batch_seconds": sum(b["seconds"] for b in batches),
        "batch_estimator_variance": distribution(stats["batch_estimator_variance"]),
        "batch_estimator_sd": distribution(stats["batch_estimator_variance"].sqrt()),
        "within_batch_action_variance": distribution(
            stats["within_batch_action_variance"]
        ),
        "within_batch_action_sd": distribution(
            stats["within_batch_action_variance"].sqrt()
        ),
        "within_batch_action_range": distribution(stats["within_batch_action_range"]),
        "paired_non_nominal_mean": distribution(stats["paired_mean"][:, 1:]),
        "paired_non_nominal_absolute_mean": distribution(
            stats["paired_mean"][:, 1:].abs()
        ),
        "paired_non_nominal_return_variance": distribution(
            stats["paired_return_variance"][:, 1:]
        ),
        "paired_non_nominal_batch_estimator_variance": distribution(
            stats["paired_batch_estimator_variance"][:, 1:]
        ),
        "batch_ranking_pair_agreement": distribution(
            stats["batch_ranking_agreement_by_state_pair"]
        ),
        "batch_winner_agreement": distribution(stats["batch_winner_agreement"]),
        "batch_ranking_tie_fraction": float(stats["batch_ranking_tie_fraction"]),
        "first_reward_negative_rate": float((first == -1).double().mean()),
        "first_reward_negative_rate_by_candidate": (first == -1)
        .double()
        .mean((0, 1, 2))
        .tolist(),
        "first_step_candidate_disagreement_rate": float(
            (first.amax(-1) != first.amin(-1)).double().mean()
        ),
        "terminal_fraction": float((~survived).double().mean()),
        "surviving_tail_bound": [-tail, 0],
        "mean_return_interval": [float(mean_lower.mean()), float(mean.mean())],
        "sampling_uncertainty_note": (
            "Eight batches estimate variance with seven degrees of freedom; "
            "tail bounds do not include sampling uncertainty."
        ),
        "ranking_note": (
            "Batch agreement measures repeatability, not correctness against true Q; "
            "candidate ties count 1/2. Winner ties use the first index."
        ),
        "validation": insertion["validation"],
        "critic_paths": insertion["artifact_paths"],
        "critics": [],
    }
    # Compare covariance identity rather than subtracting independent variances.
    cov = stats["batch_candidate_covariance"]
    reconstructed = cov.diagonal(dim1=-2, dim2=-1) + cov[:, :1, 0] - 2 * cov[:, :, 0]
    assert torch.allclose(
        reconstructed, stats["paired_batch_estimator_variance"], atol=1e-10
    )
    pairs = list(itertools.combinations(range(8), 2))
    target_sign = torch.stack([torch.sign(mean[:, a] - mean[:, b]) for a, b in pairs])
    for prediction in insertion["predictions"]:
        pred_sign = torch.stack(
            [torch.sign(prediction[:, a] - prediction[:, b]) for a, b in pairs]
        )
        batch_accuracy = []
        for batch_mean in stats["batch_means"]:
            signs = torch.stack(
                [torch.sign(batch_mean[:, a] - batch_mean[:, b]) for a, b in pairs]
            )
            batch_accuracy.append(float(((pred_sign * signs + 1) / 2).mean()))
        report["critics"].append(
            {
                "mae_to_noisy_finite_mean": float((prediction - mean).abs().mean()),
                "bias_to_noisy_finite_mean": float((prediction - mean).mean()),
                "pair_accuracy_to_noisy_128_return_mean": float(
                    ((pred_sign * target_sign + 1) / 2).mean()
                ),
                "pair_accuracy_to_each_16_return_batch": batch_accuracy,
                "prediction_action_range": distribution(
                    prediction.amax(-1) - prediction.amin(-1)
                ),
            }
        )
    report["paired_mean_tail_interval_width"] = distribution(
        (paired_upper - paired_lower).mean((0, 1))[:, 1:]
    )
    report["paired_mean"] = float(paired[..., 1:].mean())
    with (directory / "per_state_action.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "state",
                "candidate",
                "mean_return",
                "mean_tail_lower",
                "mean_tail_upper",
                "variance_of_16_return_estimator",
                "paired_mean",
                "paired_return_variance",
                "paired_batch_estimator_variance",
                "paired_mean_tail_lower",
                "paired_mean_tail_upper",
                "original_q",
                "evaldist_q",
            ]
        )
        for state, candidate in itertools.product(range(128), range(8)):
            writer.writerow(
                [
                    state,
                    candidate,
                    float(mean[state, candidate]),
                    float(mean_lower[state, candidate]),
                    float(mean[state, candidate]),
                    float(stats["batch_estimator_variance"][state, candidate]),
                    float(stats["paired_mean"][state, candidate]),
                    float(stats["paired_return_variance"][state, candidate]),
                    float(stats["paired_batch_estimator_variance"][state, candidate]),
                    float(paired_lower.mean((0, 1))[state, candidate]),
                    float(paired_upper.mean((0, 1))[state, candidate]),
                    *insertion["predictions"][:, state, candidate].tolist(),
                ]
            )
    (directory / "detailed_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
