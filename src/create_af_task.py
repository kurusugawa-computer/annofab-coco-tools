import argparse
import json
import subprocess
import sys
import tempfile
import time
from argparse import ArgumentParser
from pathlib import Path

from loguru import logger

from src.common.cli import create_parent_parser
from src.common.utils import configure_loguru, log_exception


def execute_annofabcli_task_put(project_id: str, json_info: dict[str, list[str]], temp_dir: Path) -> None:
    json_file = temp_dir / f"{time.time()}--task_info.json"
    json_file.write_text(json.dumps(json_info, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"{len(json_info)}件のタスクをAnnofabに登録します。 :: project_id='{project_id}'")

    command = ["annofabcli", "task", "put", "--yes", "--project_id", project_id, "--json", f"file://{json_file!s}"]

    subprocess.run(command, check=True)


def create_target_task_info(input_data_ids: list[str]) -> dict[str, list[str]]:
    """
    `annofabcli task put`コマンドの`--json`オプションに渡す情報を生成します。
    """
    return {i: [i] for i in input_data_ids}


def create_input_data_id_list_from_input_data_json(input_data_json: Path) -> list[str]:
    """
    入力データ全件ファイルに記載されている`input_data_id`のリストを生成します。

    """
    input_data_list = json.loads(input_data_json.read_text())
    return [item["input_data_id"] for item in input_data_list]


def create_parser() -> argparse.ArgumentParser:
    parser = ArgumentParser(
        description="Annofabにタスクを作成します。1個のタスクには1個の入力データが含まれています。task_idはinput_data_idと同じ値です。",
        parents=[create_parent_parser()],
    )

    parser.add_argument("--af_project_id", type=str, required=True, help="AnnofabプロジェクトのID")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--af_input_data_json",
        type=Path,
        help="Annofabの入力データ全件ファイルのパス。"
        "`input_data_id`を参照するのに利用します。"
        "`annofabcli input_data download`コマンドでダウンロードできます。"
        "ダウンロードした入力データ全件ファイルに、作成した入力データの情報が含まれていない場合は、`--latest`オプションを付与して、最新の入力データ全件ファイルをダウンロードしてください。",
    )

    group.add_argument(
        "--af_input_data_id",
        nargs="+",
        type=str,
        help="指定した`input_data_id`からタスクを作成します。",
    )

    parser.add_argument("--temp_dir", type=Path, required=False, help="一時ディレクトリのパス")

    return parser


@log_exception()
def main() -> None:
    args = create_parser().parse_args()
    configure_loguru(is_verbose=args.verbose)
    logger.info(f"argv={sys.argv}")

    af_project_id = args.af_project_id

    if args.af_input_data_json is not None:
        af_input_data_id_list = create_input_data_id_list_from_input_data_json(args.af_input_data_json)
    elif args.af_input_data_id is not None:
        af_input_data_id_list = args.af_input_data_id
    else:
        raise ValueError("`--af_input_data_json`か`--af_input_data_id`は必須です。")

    json_info = create_target_task_info(af_input_data_id_list)

    if args.temp_dir is not None:
        temp_dir = args.temp_dir
        temp_dir.mkdir(exist_ok=True, parents=True)
        execute_annofabcli_task_put(af_project_id, json_info, temp_dir)
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            execute_annofabcli_task_put(af_project_id, json_info, temp_dir_path)


if __name__ == "__main__":
    main()
