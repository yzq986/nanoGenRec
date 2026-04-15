"""统一 CLI — subcommand 分发。

Usage:
    python -m gr_demo train           --model qwen3-0.6b --input_path ...
    python -m gr_demo eval            --results_path ... --model_path ...
    python -m gr_demo eval-all        --models qwen3-0.6b qwen3-4b --quick
    python -m gr_demo compare         --eval_dir eval_results
    python -m gr_demo hyperparam      --model qwen3-0.6b --skip_embedding
    python -m gr_demo preprocess-sid  --model qwen3-0.6b --behavior_path auto
    python -m gr_demo pack            --rkmeans_s3_path ... --upload
"""

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m gr_demo <command> [options]")
        print()
        print("Commands:")
        print("  train       Train RKMeans model with Qwen3 embeddings")
        print("  eval        Evaluate a single model")
        print("  eval-all    Batch evaluate all models")
        print("  compare     Compare multiple model evaluations")
        print("  hyperparam      Hyperparameter grid search")
        print("  preprocess-sid  Train tokenizer + cache SID assignments")
        print("  pack            Pack model.tar.gz for deployment")
        sys.exit(1)

    command = sys.argv[1]
    # Remove the command from argv so submodule parsers work correctly
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == 'train':
        from gr_demo.model.train import main as train_main
        train_main()
    elif command == 'eval':
        from gr_demo.eval.behavior import main as eval_main
        eval_main()
    elif command == 'eval-all':
        from gr_demo.eval.batch import main as batch_main
        batch_main()
    elif command == 'compare':
        from gr_demo.eval.compare import main as compare_main
        compare_main()
    elif command == 'hyperparam':
        from gr_demo.eval.hyperparam import main as hyperparam_main
        hyperparam_main()
    elif command == 'preprocess-sid':
        from gr_demo.eval.preprocess_sid import main as preprocess_main
        preprocess_main()
    elif command == 'pack':
        from gr_demo.model.pack import main as pack_main
        pack_main()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, eval, eval-all, compare, hyperparam, preprocess-sid, pack")
        sys.exit(1)


if __name__ == '__main__':
    main()
