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
    p.add_argument("--output-name", metavar="FILENAME",
                   help="base file name for the results, without extension: writes FILENAME.xlsx and "
                        "FILENAME_runlog.json (default: sov_themes_<date>_<time>)")
    p.add_argument("--leiden", action="store_true", help="also run a Leiden clustering as a cross-check")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config.CACHE_DIR.mkdir(exist_ok=True)

    if args.list_topics:
        for num, name in data.list_topics(data.load(args.input, args.sheet)).items():
            print(f"{num:>3}  {name}")
        return

    # Everything is always logged to cache/run.log. On screen you get the live
    # display in a terminal, or plain log lines when the output is piped or
    # redirected to a file.
    handlers = [logging.FileHandler(config.CACHE_DIR / "run.log")]
    use_ui = sys.stdout.isatty()
    if not use_ui:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers)

    if use_ui:
        progress.enable()
    try:
        pipeline.run(args.input, args.sheet, args.output_dir, args.topics,
                     leiden=args.leiden, out_name=args.output_name)
    except Exception:
        progress.ui.set_stage("Failed, see the last lines below and cache/run.log")
        logging.getLogger(__name__).exception("Run failed")
        raise
    finally:
        progress.disable()
