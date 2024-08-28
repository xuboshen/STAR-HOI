import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--foo", help="foo help")
    parser.add_argument("--door", type=int, choices=range(1, 4))
    parser.add_argument(
        "--test-hoid",
        action="store_true",
        help="Default to be false, i.e. lock the stable diffusion.",
    )
    parser.add_argument(
        "--test-sam2",
        nargs="*",
        help="support more than one argument and merge into a list.",
    )

    args = parser.parse_args()
