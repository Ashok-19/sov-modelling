"""Command line entry point: python -m sov --topics 9"""

import argparse
import logging
import sys

from . import config, data, pipeline


def main():
    p = argparse.ArgumentParser(description="Find sub-themes within each topic of the social listening file.")
    p.add_argument("--topics", type=int, nargs="+",
                   help="topic numbers to run, e.g. --topics 5 9 11 (default: all)")
    p.add_argument("--list-topics", action="store_true", help="show the topics found in the data and exit")
    p.add_argument("--leiden", action="store_true", help="also run the Leiden cross-check")
    p.add_argument("--ctx", type=int, default=config.LLM_CTX, help="LLM context size in tokens")
    p.add_argument("--out", help="output file name (without .xlsx)")
    args = p.parse_args()

    config.CACHE_DIR.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(config.CACHE_DIR / "run.log")])

    if args.list_topics:
        for num, name in data.list_topics(data.load()).items():
            print(f"{num:>3}  {name}")
        return
    pipeline.run(args.topics, leiden=args.leiden, ctx=args.ctx, out_name=args.out)


if __name__ == "__main__":
    main()
