"""Command line options. run.py and `python -m sov` both end up here."""

import argparse
import logging
import sys

from . import config, data, pipeline, progress


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="run.py",
        description="Find the sub-themes inside each topic of a social listening file "
                    "and give each one a name and description.",
        epilog="Examples:\n"
               "  python run.py --list-topics\n"
               "  python run.py -t 9\n"
               "  python run.py -i my_posts.csv -o results -t 5 9 11\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", default=str(config.DATA_FILE),
                   help="Excel (.xlsx) or CSV file with the posts. Needs the columns "
                        f"'{config.ID_COL}', '{config.TEXT_COL}' and '{config.TOPIC_COL}'. "
                        "Default: data/ocrevus_ms_social_listening_classified.xlsx")
    p.add_argument("--sheet", default=config.SHEET, help="sheet to read when the input is Excel (default: %(default)s)")
    p.add_argument("-o", "--output-dir", default=str(config.OUTPUT_DIR), help="folder for the results (default: output/)")
    p.add_argument("-t", "--topics", type=int, nargs="+", metavar="N",
                   help="topic numbers to run, e.g. -t 5 9 11 (default: every topic)")
    p.add_argument("--list-topics", action="store_true", help="show the topics in the input file and exit")
    p.add_argument("--name", help="name for the output files (default: sov_themes_<date>_<time>)")
    p.add_argument("--leiden", action="store_true", help="also run a Leiden clustering as a cross-check")
    p.add_argument("--ctx", type=int, default=config.LLM_CTX, help="LLM context size in tokens (default: %(default)s)")
    p.add_argument("--no-thinking", action="store_true",
                   help="switch off Gemma's thinking step (much faster, answers may be weaker)")
    p.add_argument("--thinking-budget", type=int, default=config.LLM_THINKING_BUDGET,
                   help="most tokens Gemma may spend thinking per answer (default: %(default)s)")
    p.add_argument("--no-ui", action="store_true", help="plain log lines instead of the live display")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config.CACHE_DIR.mkdir(exist_ok=True)

    if args.list_topics:
        for num, name in data.list_topics(data.load(args.input, args.sheet)).items():
            print(f"{num:>3}  {name}")
        return

    config.LLM_CTX = args.ctx
    config.LLM_THINKING = not args.no_thinking
    config.LLM_THINKING_BUDGET = args.thinking_budget

    # Everything is always logged to cache/run.log. On screen you get either
    # the live display or, with --no-ui or when piping, the plain log lines.
    handlers = [logging.FileHandler(config.CACHE_DIR / "run.log")]
    use_ui = not args.no_ui and sys.stdout.isatty()
    if not use_ui:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers)

    if use_ui:
        progress.enable()
    try:
        pipeline.run(args.input, args.sheet, args.output_dir, args.topics,
                     leiden=args.leiden, ctx=args.ctx, out_name=args.name)
    except Exception:
        progress.ui.set_stage("Failed, see the last lines below and cache/run.log")
        logging.getLogger(__name__).exception("Run failed")
        raise
    finally:
        progress.disable()
