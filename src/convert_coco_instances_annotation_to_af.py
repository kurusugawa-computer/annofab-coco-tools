import argparse
import collections
import json
import subprocess
import sys
import uuid
from argparse import ArgumentParser
from collections.abc import Collection
from enum import Enum
from pathlib import Path
from typing import Any, assert_never

import numpy
import pycocotools
from annofabapi.segmentation import write_binary_image
from loguru import logger

from src.common.cli import create_parent_parser
from src.common.utils import configure_loguru, log_exception


class CocoAnnotationType(Enum):
    BBOX = "bbox"
    POLYGON_SEGMENTATION = "polygon_segmentation"
    RLE_SEGMENTATION = "rle_segmentation"


def convert_coco_one_segmentation_to_af_format(polygon_segmentation: list[float]) -> dict[str, Any]:
    """ """
    # COCO形式の1個のアノテーションの`segmentation`をAnnofab形式のポリゴンに変換します。
    return {"points": [{"x": polygon_segmentation[i], "y": polygon_segmentation[i + 1]} for i in range(0, len(polygon_segmentation), 2)], "_type": "Polygon"}


class AnnotationConverterFromCocoToAnnofab:
    def __init__(self, coco_instances: dict[str, Any], coco_annotation_type: CocoAnnotationType, *, target_coco_category_names: Collection[str] | None = None) -> None:
        self.coco_annotation_type = coco_annotation_type
        self.coco_images = coco_instances["images"]

        annotations_by_image_id = collections.defaultdict(list)
        for coco_anno in coco_instances["annotations"]:
            annotations_by_image_id[coco_anno["image_id"]].append(coco_anno)
        self.annotations_by_image_id: dict[int, list[dict[str, Any]]] = annotations_by_image_id

        self.target_coco_category_names = set(target_coco_category_names) if target_coco_category_names is not None else None

        self.category_names_by_id: dict[int, str] = {category["id"]: category["name"] for category in coco_instances["categories"]}

    def convert_bbox_annotation_to_af_detail(self, coco_annotation: dict[str, Any]) -> dict[str, Any] | None:
        """
        COCO形式の1個のアノテーションの`bbox`をAnnofab形式の矩形アノテーションに変換します。

        Returns:
            変換したAnnofab形式の矩形アノテーション。変換対象でない場合はNoneを返します。
        """
        coco_category_name = self.category_names_by_id[coco_annotation["category_id"]]
        if self.target_coco_category_names is not None and coco_category_name not in self.target_coco_category_names:
            return None

        attributes = {
            "coco.annotation_id": coco_annotation["id"],
        }
        left_top_x, left_top_y, width, height = coco_annotation["bbox"]
        data = {"left_top": {"x": left_top_x, "y": left_top_y}, "right_bottom": {"x": left_top_x + width, "y": left_top_y + height}, "_type": "BoundingBox"}
        return {"annotation_id": str(uuid.uuid4()), "label": coco_category_name, "attributes": attributes, "data": data}

    def convert_polygon_segmentation_annotation_to_af_detail(self, coco_annotation: dict[str, Any]) -> list[dict[str, Any]]:
        """
        COCO形式の1個のアノテーションの`segmentation`（iscrowd=0のポリゴン）をAnnofab形式のポリゴンに変換します。
        COCOの`segmentation`は複数に分割されている場合があるので、listを返します。
        """
        if coco_annotation["iscrowd"] != 0:
            return []

        coco_category_name = self.category_names_by_id[coco_annotation["category_id"]]
        if self.target_coco_category_names is not None and coco_category_name not in self.target_coco_category_names:
            return []

        attributes = {
            "coco.annotation_id": coco_annotation["id"],
        }
        segmentation = coco_annotation["segmentation"]
        assert isinstance(segmentation, list)
        if isinstance(segmentation[0], list):
            return [
                {
                    "label": coco_category_name,
                    "annotation_id": str(uuid.uuid4()),
                    "attributes": attributes,
                    "data": convert_coco_one_segmentation_to_af_format(polygon),
                }
                for polygon in segmentation
            ]
        else:
            data = convert_coco_one_segmentation_to_af_format(segmentation)
            return [{"label": coco_category_name, "attributes": attributes, "data": data}]

    def convert_rle_segmentation_annotation_to_af_detail(self, coco_annotation: dict[str, Any], coco_image: dict[str, Any]) -> tuple[dict[str, Any] | None, numpy.ndarray | None]:
        """
        COCO形式のRLE形式の`segmentation`（iscrowd=1）をAnnofabの塗りつぶしv1アノテーションに変換します。

        Returns:
            tuple[0]: Annofabの`detail`. iscrowd=0の場合はNone
            tuple[1]: segmentationをboolean arrayに変換したもの。iscrowd=0の場合はNone
        """
        if coco_annotation["iscrowd"] != 1:
            return None, None

        coco_category_name = self.category_names_by_id[coco_annotation["category_id"]]
        if self.target_coco_category_names is not None and coco_category_name not in self.target_coco_category_names:
            return None, None

        attributes = {
            "coco.annotation_id": coco_annotation["id"],
        }
        segmentation = coco_annotation["segmentation"]

        # 以下のコードと同じように、rleを取得した
        # https://github.com/ppwwyyxx/cocoapi/blob/8cbc887b3da6cb76c7cc5b10f8e082dd29d565cb/PythonAPI/pycocotools/coco.py#L266C1-L269C56
        if isinstance(segmentation["counts"], list):
            rle = pycocotools.mask.frPyObjects(segmentation, coco_image["height"], coco_image["width"])
        else:
            rle = segmentation

        segmentation_bool_array = pycocotools.mask.decode(rle).astype(bool)
        annotation_id = str(uuid.uuid4())
        af_detail = {"label": coco_category_name, "annotation_id": annotation_id, "attributes": attributes, "data": {"data_uri": annotation_id, "_type": "Segmentation"}}
        return af_detail, segmentation_bool_array

    def convert_annotations_to_af_details(self, coco_image: dict[str, Any], af_input_data_dir: Path) -> list[dict[str, Any]]:
        """
        COCO形式の`images -> file_name`に対応するアノテーションをAnnofab形式の`details`に変換します。

        Args:
            coco_image: 変換対象のCOCO形式のimage情報
            af_input_data_dir: Annofab形式の入力データに対応するディレクトリ。塗りつぶしアノテーションに変換する場合、このディレクトリに塗りつぶし画像が格納されます。
        """
        coco_annotations = self.annotations_by_image_id[coco_image["id"]]
        match self.coco_annotation_type:
            case CocoAnnotationType.BBOX:
                af_details = []
                for anno in coco_annotations:
                    af_detail = self.convert_bbox_annotation_to_af_detail(anno)
                    if af_detail is not None:
                        af_details.append(af_detail)

            case CocoAnnotationType.POLYGON_SEGMENTATION:
                af_details = [detail for anno in coco_annotations for detail in self.convert_polygon_segmentation_annotation_to_af_detail(anno)]
            case CocoAnnotationType.RLE_SEGMENTATION:
                af_details = []
                for anno in coco_annotations:
                    af_detail, segmentation_bool_array = self.convert_rle_segmentation_annotation_to_af_detail(anno, coco_image)
                    if af_detail is None:
                        continue

                    assert segmentation_bool_array is not None
                    af_input_data_dir.mkdir(exist_ok=True, parents=True)
                    with (af_input_data_dir / af_detail["annotation_id"]).open("wb") as f:
                        write_binary_image(segmentation_bool_array, f)
                    af_details.append(af_detail)
            case _ as unreachable:
                assert_never(unreachable)

        return af_details

    def convert(self, output_dir: Path, input_data_id_to_task_id: dict[str, str], input_data_name_to_input_data_id: dict[str, str]) -> None:
        """
        COCO形式のアノテーション全体をAnnofab形式に変換します。
        """
        output_dir.mkdir(exist_ok=True, parents=True)
        success_count = 0
        logger.info(f"COCOデータセットの{len(self.coco_images)}件のimagesに紐づくアノテーションを、Annofab形式に変換します。")

        for image_index, coco_image in enumerate(self.coco_images):
            if (image_index + 1) % 1000 == 0:
                logger.info(f"{image_index + 1}件目のCOCO imagesに紐づくアノテーションを、Annofabフォーマットに変換中...")

            image_file_name = coco_image["file_name"]
            af_input_data_id = input_data_name_to_input_data_id.get(image_file_name)
            if af_input_data_id is None:
                logger.warning(f"Annofabのinput_data_name='{image_file_name}'に対応するinput_data_idが見つかりません。スキップします。")
                continue

            af_task_id = input_data_id_to_task_id.get(af_input_data_id)
            if af_task_id is None:
                logger.warning(f"Annofabのinput_data_id='{af_input_data_id}'に対応するtask_idが見つかりません。スキップします。")
                continue

            af_annotation_json = output_dir / af_task_id / f"{af_input_data_id}.json"
            try:
                af_details = self.convert_annotations_to_af_details(coco_image, af_input_data_dir=output_dir / af_task_id / af_input_data_id)
                if len(af_details) == 0:
                    logger.debug(f"COCOのimage.file_name='{image_file_name}'に紐づく変換対象のアノテーションは存在しません。")

                af_annotation_json.parent.mkdir(exist_ok=True, parents=True)
                af_annotation_json.write_text(json.dumps({"details": af_details}, ensure_ascii=False, indent=2))
                success_count += 1
                logger.debug(f"COCOのimage.file_name='{image_file_name}'に紐づくアノテーション{len(af_details)}件を、Annofab形式に変換して'{af_annotation_json}'に出力しました。")
            except Exception:
                logger.opt(exception=True).warning(f"COCOのimage.file_name='{image_file_name}'に紐づくアノテーションを、Annofabフォーマットへ変換するのに失敗しました。")

        logger.info(f"{success_count}/{len(self.coco_images)}件のCOCO imagesに紐づくアノテーションを、Annofabフォーマットに変換しました。 :: output_dir='{output_dir}'")


def create_input_data_id_to_task_id_mapping(task_json: Path) -> dict[str, str]:
    """
    Annofabのタスク全件ファイルから、input_data_idとtask_idのマッピングを作成します。

    Returns:
        keyが`input_data_id`、valueが`task_id`の辞書

    Raises:
        ValueError: 1個の入力データが複数のタスクから参照されている
    """
    task_data = json.loads(task_json.read_text())
    result = {}
    for task in task_data:
        for input_data_id in task["input_data_id_list"]:
            task_id = task["task_id"]
            if input_data_id in result:
                raise ValueError(f"input_data_id='{input_data_id}'の入力データは複数のタスクに含まれています。入力データは1個のタスクのみ含まれるように変更してください。")

            result[input_data_id] = task_id
    return result


def create_input_data_name_to_input_data_id_mapping(input_data_json: Path) -> dict[str, str]:
    """
    Annofabの入力データ全件ファイルから、input_data_nameからinput_data_idのマッピングを作成します。

    Returns:
        keyが`input_data_name`、valueが`input_data_id`の辞書

    Raises:
        ValueError: 1個の入力データが複数のタスクから参照されている
    """
    input_data = json.loads(input_data_json.read_text())
    result = {}
    for item in input_data:
        input_data_name = item["input_data_name"]
        input_data_id = item["input_data_id"]
        if input_data_name in result:
            raise ValueError(f"input_data_name='{input_data_name}'の入力データが複数存在します。input_data_nameが重複しないようにしてください。")

        result[input_data_name] = input_data_id
    return result


def execute_annofabcli_task_download(project_id: str, output_json: Path, *, is_latest: bool) -> None:
    command = ["annofabcli", "task", "download", "--project_id", project_id, "--output", str(output_json)]
    if is_latest:
        command.append("--latest")
    subprocess.run(command, check=True)


def create_parser() -> argparse.ArgumentParser:
    parser = ArgumentParser(
        description="COCOデータセット（Instances）に含まれるアノテーションを、Annofab形式に変換します。出力結果は`annofabcli annotation import`コマンドでアノテーションを登録できます。",
        parents=[create_parent_parser()],
    )

    parser.add_argument(
        "--coco_instances_json", type=Path, required=True, help="入力情報であるCOCOデータセット（Instances）形式アノテーションのJSONファイルのパス。`annotations`,`images`,`categories`を参照します。"
    )

    parser.add_argument(
        "--af_task_json",
        type=Path,
        help="Annofabのタスク全件ファイルのパス。"
        "`task_id`と`input_data_id`の関係を参照するのに利用します。"
        "`annofabcli task download`コマンドでダウンロードできます。"
        "ダウンロードしたタスク全件ファイルに、作成したタスクの情報が含まれていない場合は、`--latest`オプションを付与して、最新のタスク全件ファイルをダウンロードしてください。",
    )

    parser.add_argument(
        "--af_input_data_json",
        type=Path,
        help="Annofabの入力データ全件ファイルのパス。`input_data_name`と`input_data_id`の関係を参照するのに利用します。`annofabcli input_data download`コマンドでダウンロードできます。",
    )

    parser.add_argument("--coco_category_name", type=str, nargs="+", help="変換対象のCOCOのcategory_name")
    parser.add_argument(
        "--coco_annotation_type",
        type=str,
        required=True,
        choices=[e.value for e in CocoAnnotationType],
        default=CocoAnnotationType.BBOX.value,
        help="変換対象のアノテーションの種類。`bbox`:バウンディングボックス, `polygon_segmentation`:`iscrowd=0`のポリゴン形式のsegmentation, `rle_segmentation`:`iscrowd=1`のRLE形式のsegmentation",
    )

    parser.add_argument("-o", "--output_dir", type=Path, required=True, help="Annofab形式のアノテーションの出力先ディレクトリのパス")

    parser.add_argument("--temp_dir", type=Path, required=True, help="一時ディレクトリのパス")
    return parser


@log_exception()
def main() -> None:
    args = create_parser().parse_args()
    configure_loguru(is_verbose=args.verbose)
    logger.info(f"argv={sys.argv}")

    coco_instances = json.loads(args.coco_instances_json.read_text())

    input_data_id_to_task_id = create_input_data_id_to_task_id_mapping(args.af_task_json)
    input_data_name_to_input_data_id = create_input_data_name_to_input_data_id_mapping(args.af_input_data_json)
    converter = AnnotationConverterFromCocoToAnnofab(coco_instances, CocoAnnotationType(args.coco_annotation_type), target_coco_category_names=args.coco_category_name)
    converter.convert(args.output_dir, input_data_id_to_task_id=input_data_id_to_task_id, input_data_name_to_input_data_id=input_data_name_to_input_data_id)


if __name__ == "__main__":
    main()
