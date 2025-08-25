import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from jsonargparse import ArgumentParser
from loguru import logger

from src.common.cli import create_parent_parser
from src.common.utils import configure_loguru, log_exception


def execute_annofabcli_input_data_put(project_id: str, json_info: list[dict[str, Any]], temp_dir: Path) -> None:
    json_file = temp_dir / f"{time.time()}--input_data_info.json"
    json_file.write_text(json.dumps(json_info, ensure_ascii=False, indent=2), encoding="utf-8")

    command = ["annofabcli", "input_data", "put", "--yes", "--project_id", project_id, "--json", f"file://{json_file!s}", "--parallelism", "4"]

    subprocess.run(command, check=True)


def create_target_input_data_info(coco_images: list[dict[str, Any]], image_dir: Path) -> list[dict[str, str]]:
    """
    `annofabcli input_data put`コマンドの`--json`オプションに渡す情報を生成します。
    """
    af_results = []
    for coco_image in coco_images:
        file_name = coco_image["file_name"]
        af_results.append(
            {
                "input_data_id": file_name,
                "input_data_name": file_name,
                "input_data_path": f"file://{image_dir / file_name}",
            }
        )
    return af_results


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="COCOデータセットのimagesから、Annofabに入力データを作成します。"
        "Annofabの入力データの`input_data_name`は、COCOデータセットの`image.file_name`を格納します。"
        "`input_data_id`は、`input_data_name`とほとんど同じ値になります。",
        parents=[create_parent_parser()],
    )

    parser.add_argument("--coco_instances_json", type=Path, required=True, help="入力情報であるCOCOデータセット形式アノテーションのJSONファイルのパス。`images`を参照します。")

    parser.add_argument(
        "--image_dir",
        help="COCOデータセットの画像ファイルが存在するディレクトリのパス。",
        type=Path,
        required=True,
    )

    parser.add_argument("--af_project_id", type=str, required=True, help="AnnofabプロジェクトのID")

    parser.add_argument("--coco_image_file_name", type=str, nargs="+", help="作成対象のCOCOのimageのfile_name")
    parser.add_argument("--temp_dir", type=Path, required=False, help="一時ディレクトリのパス")

    return parser


@log_exception()
def main() -> None:
    args = create_parser().parse_args()
    configure_loguru(is_verbose=args.verbose)
    logger.info(f"argv={sys.argv}")

    image_dir = args.image_dir
    af_project_id = args.af_project_id

    coco_instances = json.loads(args.coco_instances_json.read_text())
    coco_images = coco_instances["images"]
    if args.coco_image_file_name is not None:
        coco_images = [img for img in coco_images if img["file_name"] in args.coco_image_file_name]
    json_info = create_target_input_data_info(coco_images, image_dir)

    logger.info(f"COCOデータセットのimage{len(json_info)}件を、Annofabへ入力データとして登録します。 :: af_project_id='{af_project_id}'")

    if args.temp_dir is not None:
        temp_dir = args.temp_dir
        temp_dir.mkdir(exist_ok=True, parents=True)
        execute_annofabcli_input_data_put(af_project_id, json_info, temp_dir)
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            execute_annofabcli_input_data_put(af_project_id, json_info, temp_dir_path)


if __name__ == "__main__":
    main()
