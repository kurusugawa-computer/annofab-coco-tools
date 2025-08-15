import zipfile
import collections
import argparse
import collections
import json
import sys
from jsonargparse import ArgumentParser
from pathlib import Path
from typing import Any

from annofabapi.parser import SimpleAnnotationDirParser
from loguru import logger
from more_itertools import first_true
from pydantic import BaseModel
from yaml import safe_load

from src.common.cli import create_parent_parser
from src.common.utils import configure_loguru, log_exception
from typing import Collections
from annofabapi.parser import lazy_parse_simple_annotation_dir, lazy_parse_simple_annotation_zip


def convert_af_input_data_to_coco_image(af_input_data: dict[str, Any], coco_image_id: int) -> dict[str, Any]:
    """
    Annofabの入力データ情報をCOCO形式のimage情報に変換します。
    """
    original_resolution = af_input_data["system_metadata"]["original_resolution"]
    return {
        "id": coco_image_id,
        "file_name": af_input_data["input_data_name"],
        "width": original_resolution["width"],
        "height": original_resolution["height"],
    }


def convert_af_input_data_list_to_coco_images(af_input_data_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Annofabの入力データ情報のlistをCOCO形式のimageのlistに変換します。
    """
    input_data_name_counter = collections.Counter([af_input_data["input_data_name"] for af_input_data in af_input_data_list])
    duplicated_input_data_names = [name for name, count in input_data_name_counter.items() if count > 1]
    if len(duplicated_input_data_names) > 0:
        raise ValueError(f"Annofabの次のinput_data_nameは重複しています。この変換ツールではinput_data_nameで紐づけるため、input_data_nameは一意である必要があります。 :: {duplicated_input_data_names}")

    coco_images = [convert_af_input_data_to_coco_image(af_input_data, coco_image_id=i) for i, af_input_data in enumerate(af_input_data_list)]
    return coco_images


class AnnotationConverterFromAnnofabToCoco:
    def __init__(
        self, coco_categories: list[dict[str, Any]], coco_images: list[dict[str, Any]], *, target_af_target_labels: Collections[str] | None = None, should_clip_annotation_to_image: bool = False
    ) -> None:
        self.category_ids_by_name: dict[int, str] = {category["name"]: category["id"] for category in coco_categories}
        self.images_by_file_name: dict[int, str] = {image["file_name"]: image for image in coco_images}
        self.should_clip_annotation_to_image = should_clip_annotation_to_image

        self.target_af_target_labels = set(target_af_target_labels) if target_af_target_labels is not None else None

    def convert_af_bounding_box_detail(
        self, af_detail: dict[str, Any], coco_image: dict[str, Any], coco_annotation_id: int, *, task_id: str | None = None, input_data_id: str | None = None
    ) -> dict[str, Any]:
        """
        Bounding BoxのAnnofabのdetail情報をCOCO形式に変換します。
        """
        annotation_id = af_detail["annotation_id"]
        left_top = af_detail["data"]["left_top"].copy()
        right_bottom = af_detail["data"]["right_bottom"].copy()
        image_width = coco_image["width"]
        image_height = coco_image["height"]

        if self.should_clip_annotation_to_image:
            # bboxが画像からはみ出ていないかチェックし、はみ出ていれば修正
            orig_left_top = left_top.copy()
            orig_right_bottom = right_bottom.copy()
            modified = False
            if left_top["x"] < 0:
                left_top["x"] = 0
                modified = True
            if left_top["y"] < 0:
                left_top["y"] = 0
                modified = True
            if right_bottom["x"] > image_width:
                right_bottom["x"] = image_width
                modified = True
            if right_bottom["y"] > image_height:
                right_bottom["y"] = image_height
                modified = True
            if image_width is not None and left_top["x"] > image_width:
                left_top["x"] = image_width
                modified = True
            if image_height is not None and left_top["y"] > image_height:
                left_top["y"] = image_height
                modified = True
            if right_bottom["x"] < left_top["x"]:
                right_bottom["x"] = left_top["x"]
                modified = True
            if right_bottom["y"] < left_top["y"]:
                right_bottom["y"] = left_top["y"]
                modified = True
            if modified:
                logger.debug(
                    f"bboxが画像からはみ出ていたため修正しました。 :: "
                    f"task_id='{task_id}', input_data_id='{input_data_id}', annotation_id='{annotation_id}', coco_image_id='{coco_image['id']}', coco_annotation_id='{coco_annotation_id}' :: "
                    f"original_left_top={orig_left_top}, original_right_bottom={orig_right_bottom}, "
                    f"new_left_top={left_top}, new_right_bottom={right_bottom}"
                )

        bbox = [left_top["x"], left_top["y"], right_bottom["x"] - left_top["x"], right_bottom["y"] - left_top["y"]]
        segmentation = [[left_top["x"], left_top["y"], right_bottom["x"], left_top["y"], right_bottom["x"], right_bottom["y"], left_top["x"], right_bottom["y"]]]
        area = bbox[2] * bbox[3]
        return {
            "id": coco_annotation_id,
            "image_id": coco_image["id"],
            "category_id": self.category_ids_by_name[af_detail["label"]],
            "bbox": bbox,
            "segmentation": segmentation,
            "area": area,
            "iscrowd": 0,
        }

    def convert_af_annotation(self, af_annotation: dict[str, Any], coco_image: dict[str, Any], coco_start_annotation_id: int) -> tuple[list[dict[str, Any]], int]:
        """
        Annofab形式の1個のJSONファイルに格納されているアノテーション情報を、COCO形式の複数個のアノテーションに変換します。

        Args:
            af_annotation: Annofab形式のアノテーション情報
            coco_image: COCO形式のimage
            coco_start_annotation_id: COCO形式のannotation_idの開始番号

        Returns:
            tuple[0]: COCO形式のアノテーションのリスト
            tuple[1]: 次のアノテーションID


        """
        af_details = af_annotation["details"]
        task_id = af_annotation["task_id"]
        input_data_id = af_annotation["input_data_id"]
        coco_annotations = []
        coco_annotation_id = coco_start_annotation_id
        for af_detail in af_details:
            if self.target_af_target_labels is not None and af_detail["label"] not in self.target_af_target_labels:
                continue

            if af_detail["data"]["_type"] != "BoundingBox":
                coco_annotation = self.convert_af_bounding_box_detail(af_detail, coco_image, coco_annotation_id, task_id=task_id, input_data_id=input_data_id)
            else:
                continue

            coco_annotations.append(coco_annotation)
            coco_annotation_id += 1
        return coco_annotations, coco_annotation_id

    def convert_af_annotation_dir(
        self,
        af_annotation_path: Path,
        *,
        target_task_ids: Collections[str] | None = None,
        target_input_data_ids: Collections[str] | None = None,
        target_task_phase: str | None,
        target_task_status: str | None,
    ) -> list[dict[str, Any]]:  # noqa: PLR0912, PLR0915
        """
        AnnofabからダウンロードしたアノテーションZIPまたは展開したディレクトリから、COCO形式のアノテーションを生成します。

        Raises:
            ValueError: アノテーションに問題がある場合

        """
        coco_start_annotation_id = 1

        if zipfile.is_zipfile(af_annotation_path):
            iter_af_annotation_parser = lazy_parse_simple_annotation_zip(af_annotation_path)
        elif af_annotation_path.is_dir():
            iter_af_annotation_parser = lazy_parse_simple_annotation_dir(af_annotation_path)
        else:
            raise ValueError(f"'{af_annotation_path}'はZIPファイルでもディレクトリでもありません。")

        coco_annotations = []

        success_count = 0
        for af_parser in iter_af_annotation_parser:
            if target_task_ids is not None and af_parser.task_id not in target_task_ids:
                continue
            if target_input_data_ids is not None and af_parser.input_data_id not in target_input_data_ids:
                continue

            af_annotation = af_parser.load_json()
            if target_task_phase is not None and af_annotation["task_phase"] != target_task_phase:
                continue
            if target_task_status is not None and af_annotation["task_status"] != target_task_status:
                continue

            try:
                # Annofabのinput_data_nameをCOCOのfile_nameとして変換する
                coco_image = self.images_by_file_name[af_annotation["input_data_name"]]
                sub_coco_annotations, coco_start_annotation_id = self.convert_af_annotation(af_annotation, coco_image, coco_start_annotation_id)
                logger.debug(f"AnnofabのアノテーションJSONファイル'{af_parser.json_file_path}'をCOCO形式のannotations（{len(sub_coco_annotations)}個）に変換しました。 ")
                coco_annotations.extend(sub_coco_annotations)
                success_count += 1
            except Exception:
                logger.opt(exception=True).warning(f"AnnofabのアノテーションJSONファイル'{af_parser.json_file_path}'の変換に失敗しました。")
                continue

        logger.info(f"Annofab形式のアノテーション'{af_annotation_path}'に含まれる{success_count}個のJSONファイルを、COCO形式のannotations（{len(coco_annotations)}個）に変換しました。")
        return coco_annotations


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Annofab形式のアノテーションを、COCOデータセット（Instances）形式に変換します。",
        parents=[create_parent_parser()],
    )

    parser.add_argument(
        "--af_annotation_path",
        type=Path,
        required=True,
        help="Annofab形式のアノテーションZIPファイルのパス。またはZIPファイルを展開したディレクトリのパス。`annofabcli annotation download`コマンドでアノテーションZIPファイルをダウンロードできます。",
    )

    parser.add_argument(
        "--coco_instances_json", type=Path, required=True, help="入力情報であるCOCOデータセット（Instances）形式アノテーションのJSONファイルのパス。`categories`と`images`(オプショナル)を参照します。"
    )

    parser.add_argument("-o", "--output_coco_instances_json", type=Path, required=True, help="変換後のCOCOデータセット（Instances）形式アノテーションの出力先JSONファイルのパス")

    parser.add_argument("--af_task_id", type=str, nargs="+", help="変換対象のタスクのID")
    parser.add_argument("--af_input_data_id", type=str, nargs="+", help="変換対象の入力データのID")
    parser.add_argument("--af_task_phase", type=str, help="変換対象のタスクのフェーズ")
    parser.add_argument("--af_task_status", type=str, help="変換対象のタスクのステータス")

    return parser


@log_exception(logger=logger)
def main() -> None:  # noqa: PLR0915, PLR0912
    args = create_parser().parse_args()
    configure_loguru(is_verbose=args.verbose)
    logger.info(f"argv={sys.argv}")

    af_annotation_dir = Path(args.annotation)
    if not af_annotation_dir.is_dir():
        logger.error(f"'{af_annotation_dir}'はディレクトリではありません。")
        sys.exit(1)

    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    main()
