import argparse


def benchmark_args():
    parser = argparse.ArgumentParser(
        description="Set hyperparameters for benchmarking custom NNs."
    )
    parser.add_argument(
        "--model_type",
        action="store",
        type=str,
        nargs="?",
        default="BaselineMLP",
        help="Name of the model type to train",
    )
    parser.add_argument(
        "--test_case",
        action="store",
        type=str,
        nargs="?",
        default="quadratic",
        help="Test case function",
    )
    parser.add_argument(
        "--input_dim",
        action="store",
        type=int,
        nargs="?",
        default=1,
        help="Dimenison of the input variables",
    )
    parser.add_argument(
        "--epochs",
        action="store",
        type=int,
        nargs="?",
        default=1000,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--split_chunk", action="store_true", help="Contiguous chunk test-train split"
    )
    parser.add_argument("--save_model", action="store_true", help="Save the model")
    parser.add_argument(
        "--colormap_diff", action="store_true", help="Save a colormap of the diff in 3D"
    )
    parser.add_argument(
        "--colormap_values",
        action="store_true",
        help="Save a colormap of the pointwise values in 3D",
    )
    try:
        args = parser.parse_args()
    except SystemExit:
        print("Exception occurred! You've likely mispelled a key.")
    return args


def critic_eval_args():
    parser = argparse.ArgumentParser(
        description="Toggle between critic evaluation settings."
    )
    parser.add_argument(
        "--model_type",
        action="store",
        type=str,
        nargs="?",
        default="StandardMLP",
        help="Name of the model type to evaluate",
    )
    parser.add_argument(
        "--load_from",
        action="store",
        type=str,
        nargs="?",
        help="Name of the directory to load the critic from",
    )
    try:
        args = parser.parse_args()
    except SystemExit:
        print("Exception occurred! You've likely mispelled a key.")
    return args
