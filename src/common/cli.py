import jsonargparse


def create_parent_parser() -> jsonargparse.ArgumentParser:
    """
    共通の引数セットを生成する。
    """
    parent_parser = jsonargparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--verbose", action="store_true", help="詳細なログを出力します。")

    return parent_parser

