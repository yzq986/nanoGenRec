from public_benchmarks.movielens_cpu import (
    build_semantic_ids,
    filter_sequences,
    split_train_eval,
    synthetic_movielens,
)
from public_benchmarks.baselines import build_eval_examples, recall_metrics


def test_synthetic_public_movielens_path_builds_examples():
    data = synthetic_movielens(n_users=32, n_items=40, seed=7)
    seqs = filter_sequences(data, min_rating=3.0, min_user_items=5, max_users=20)
    used_items = {mid for items in seqs.values() for mid in items}
    movies = {mid: data.movies[mid] for mid in used_items}

    sid, clusters = build_semantic_ids(
        movies,
        seqs,
        n_clusters=(8, 8, 8),
        feature_dim=32,
        feature_source="hybrid",
        collab_window=3,
        kmeans_iters=2,
        kmeans_sample_size=0,
        seed=7,
    )
    train_examples, alignment_examples, eval_examples = split_train_eval(
        seqs, sid, max_items=20, train_mode="sliding", min_context_items=2)

    assert clusters == [8, 8, 8]
    assert sid
    assert train_examples
    assert alignment_examples
    assert eval_examples
    assert len(train_examples) > len(eval_examples)
    assert all(len(example) >= 6 for example in train_examples)
    assert all(len(example["target_sid"]) == 3 for example in eval_examples)


def test_public_baseline_helpers_compute_recall():
    data = synthetic_movielens(n_users=24, n_items=18, seed=3)
    seqs = filter_sequences(data, min_rating=3.0, min_user_items=5, max_users=0)
    examples, train_histories = build_eval_examples(seqs, max_items=20)
    popularity = [mid for items in train_histories for mid in items]

    metrics = recall_metrics("toy", examples[:5], [1, 5], lambda _ex: popularity)

    assert set(metrics) == {"toy_recall@1", "toy_recall@5"}
    assert 0.0 <= metrics["toy_recall@1"] <= 1.0
    assert metrics["toy_recall@1"] <= metrics["toy_recall@5"]
