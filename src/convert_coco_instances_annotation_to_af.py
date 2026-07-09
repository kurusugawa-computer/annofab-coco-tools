import collections
import json
import sys
import uuid
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, assert_never

import numpy
import pycocotools
import pycocotools.mask
from annofabapi.segmentation import write_binary_image
from jsonargparse import ArgumentParser
from loguru import logger

from src.common.cli import create_parent_parser
from src.common.utils import configure_loguru, log_exception


class CocoAnnotationType(Enum):
    BBOX = "bbox"
    POLYGON_SEGMENTATION = "polygon_segmentation"
    RLE_SEGMENTATION = "rle_segmentation"


@dataclass(frozen=True)
class ResizeScale:
    x: float
    y: float
    width: int
    height: int

    @property
    def is_identity(self) -> bool:
        return self.x == 1 and self.y == 1

    def scale_x(self, value: float) -> int:
        return round(value * self.x)

    def scale_y(self, value: float) -> int:
        return round(value * self.y)


IDENTITY_RESIZE_SCALE = ResizeScale(x=1, y=1, width=0, height=0)


def convert_coco_one_segmentation_to_af_format(polygon_segmentation: Sequence[float], resize_scale: ResizeScale = IDENTITY_RESIZE_SCALE) -> dict[str, Any]:
    """
    COCO形式の1個のアノテーションの`segmentation`をAnnofab形式のポリゴンに変換します。
    """
    # Annofabは座標値は整数で格納しているので、round()で整数に変換する。
    return {"points": [{"x": resize_scale.scale_x(polygon_segmentation[i]), "y": resize_scale.scale_y(polygon_segmentation[i + 1])} for i in range(0, len(polygon_segmentation), 2)], "_type": "Points"}


def create_resize_scale_from_af_input_data(af_input_data: dict[str, Any]) -> ResizeScale:
    """
    Annofab入力データ情報から、元画像サイズからAnnofab作業画像サイズへのリサイズ倍率を作成します。
    """
    system_metadata = af_input_data.get("system_metadata") or {}
    original_resolution = system_metadata.get("original_resolution")
    resized_resolution = system_metadata.get("resized_resolution")
    if original_resolution is None or resized_resolution is None:
        return IDENTITY_RESIZE_SCALE

    original_width = original_resolution["width"]
    original_height = original_resolution["height"]
    resized_width = resized_resolution["width"]
    resized_height = resized_resolution["height"]
    if original_width <= 0 or original_height <= 0 or resized_width <= 0 or resized_height <= 0:
        raise ValueError(
            f"Annofab入力データの画像サイズが不正です。 :: "
            f"input_data_id='{af_input_data.get('input_data_id')}', original_resolution={original_resolution}, resized_resolution={resized_resolution}"
        )

    return ResizeScale(x=resized_width / original_width, y=resized_height / original_height, width=resized_width, height=resized_height)


def resize_boolean_array_by_nearest_neighbor(boolean_array: numpy.ndarray, width: int, height: int) -> numpy.ndarray:
    """
    bool配列を最近傍でリサイズします。
    """
    original_height, original_width = boolean_array.shape
    if original_width == width and original_height == height:
        return boolean_array

    x_indices = numpy.minimum((numpy.arange(width) * original_width / width).astype(int), original_width - 1)
    y_indices = numpy.minimum((numpy.arange(height) * original_height / height).astype(int), original_height - 1)
    return boolean_array[y_indices[:, None], x_indices]


class AnnotationConverterFromCocoToAnnofab:
    def __init__(
        self,
        coco_instances: dict[str, Any],
        coco_annotation_type: CocoAnnotationType,
        *,
        target_coco_category_names: Collection[str] | None = None,
        target_coco_image_file_names: Collection[str] | None = None,
    ) -> None:
        self.coco_annotation_type = coco_annotation_type
        coco_images = coco_instances["images"]
        if target_coco_image_file_names is not None:
            coco_images = [img for img in coco_images if img["file_name"] in set(target_coco_image_file_names)]
        self.coco_images = coco_images

        annotations_by_image_id = collections.defaultdict(list)
        for coco_anno in coco_instances["annotations"]:
            annotations_by_image_id[coco_anno["image_id"]].append(coco_anno)
        self.annotations_by_image_id: dict[int, list[dict[str, Any]]] = annotations_by_image_id

        self.target_coco_category_names = set(target_coco_category_names) if target_coco_category_names is not None else None

        self.category_names_by_id: dict[int, str] = {category["id"]: category["name"] for category in coco_instances["categories"]}

    def convert_bbox_annotation_to_af_detail(self, coco_annotation: dict[str, Any], resize_scale: ResizeScale = IDENTITY_RESIZE_SCALE) -> dict[str, Any] | None:
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
            "coco.image_id": coco_annotation["image_id"],
        }
        left_top_x, left_top_y, width, height = coco_annotation["bbox"]
        # Annofabは座標値は整数で格納しているので、round()で整数に変換する。
        data = {
            "left_top": {"x": resize_scale.scale_x(left_top_x), "y": resize_scale.scale_y(left_top_y)},
            "right_bottom": {"x": resize_scale.scale_x(left_top_x + width), "y": resize_scale.scale_y(left_top_y + height)},
            "_type": "BoundingBox",
        }
        return {"annotation_id": str(uuid.uuid4()), "label": coco_category_name, "attributes": attributes, "data": data}

    def convert_polygon_segmentation_annotation_to_af_detail(self, coco_annotation: dict[str, Any], resize_scale: ResizeScale = IDENTITY_RESIZE_SCALE) -> list[dict[str, Any]]:
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
            "coco.image_id": coco_annotation["image_id"],
        }
        segmentation = coco_annotation["segmentation"]
        assert isinstance(segmentation, list)
        return [
            {
                "label": coco_category_name,
                "annotation_id": str(uuid.uuid4()),
                "attributes": attributes,
                "data": convert_coco_one_segmentation_to_af_format(polygon, resize_scale),
            }
            for polygon in segmentation
        ]

    def convert_rle_segmentation_annotation_to_af_detail(
        self, coco_annotation: dict[str, Any], coco_image: dict[str, Any], resize_scale: ResizeScale = IDENTITY_RESIZE_SCALE
    ) -> tuple[dict[str, Any] | None, numpy.ndarray | None]:
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
            "coco.image_id": coco_annotation["image_id"],
        }
        segmentation = coco_annotation["segmentation"]

        # 以下のコードと同じように、rleを取得した
        # https://github.com/ppwwyyxx/cocoapi/blob/8cbc887b3da6cb76c7cc5b10f8e082dd29d565cb/PythonAPI/pycocotools/coco.py#L266C1-L269C56
        if isinstance(segmentation["counts"], list):
            rle = pycocotools.mask.frPyObjects(segmentation, coco_image["height"], coco_image["width"])
        else:
            rle = segmentation

        segmentation_bool_array = pycocotools.mask.decode(rle).astype(bool)
        if not resize_scale.is_identity:
            segmentation_bool_array = resize_boolean_array_by_nearest_neighbor(segmentation_bool_array, resize_scale.width, resize_scale.height)
        annotation_id = str(uuid.uuid4())
        af_detail = {"label": coco_category_name, "annotation_id": annotation_id, "attributes": attributes, "data": {"data_uri": annotation_id, "_type": "Segmentation"}}
        return af_detail, segmentation_bool_array

    def convert_annotations_to_af_details(self, coco_image: dict[str, Any], af_input_data_dir: Path, resize_scale: ResizeScale = IDENTITY_RESIZE_SCALE) -> tuple[list[dict[str, Any]], int]:
        """
        COCO形式の`images -> file_name`に対応するアノテーションをAnnofab形式の`details`に変換します。

        Args:
            coco_image: 変換対象のCOCO形式のimage情報
            af_input_data_dir: Annofab形式の入力データに対応するディレクトリ。塗りつぶしアノテーションに変換する場合、このディレクトリに塗りつぶし画像が格納されます。

        Returns:
            tuple[0]: 変換したAnnofab形式のdetails
            tuple[1]: 変換したCOCOのアノテーションの個数。マルチポリゴンが存在する場合、この値と`len(tuple[0])`の結果は異なります。
        """
        coco_annotations = self.annotations_by_image_id[coco_image["id"]]
        af_details = []
        match self.coco_annotation_type:
            case CocoAnnotationType.BBOX:
                for anno in coco_annotations:
                    af_detail = self.convert_bbox_annotation_to_af_detail(anno, resize_scale)
                    if af_detail is not None:
                        af_details.append(af_detail)
                return af_details, len(af_details)

            case CocoAnnotationType.POLYGON_SEGMENTATION:
                target_coco_annotation_count = 0
                for anno in coco_annotations:
                    sub_details = self.convert_polygon_segmentation_annotation_to_af_detail(anno, resize_scale)
                    if len(sub_details) > 0:
                        target_coco_annotation_count += 1
                    af_details.extend(sub_details)

                return af_details, target_coco_annotation_count

            case CocoAnnotationType.RLE_SEGMENTATION:
                for anno in coco_annotations:
                    af_detail, segmentation_bool_array = self.convert_rle_segmentation_annotation_to_af_detail(anno, coco_image, resize_scale)
                    if af_detail is None:
                        continue

                    assert segmentation_bool_array is not None
                    af_input_data_dir.mkdir(exist_ok=True, parents=True)
                    with (af_input_data_dir / af_detail["annotation_id"]).open("wb") as f:
                        write_binary_image(segmentation_bool_array, f)
                    af_details.append(af_detail)
                return af_details, len(af_details)
            case _ as unreachable:
                assert_never(unreachable)

    def convert(
        self,
        output_dir: Path,
        input_data_id_to_task_id: dict[str, str] | None,
        input_data_name_to_input_data_id: dict[str, str] | None,
        input_data_name_to_input_data: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """
        COCO形式のアノテーション全体をAnnofab形式に変換します。

        Args:
            output_dir: 変換したAnnofab形式のアノテーションを出力するディレクトリ
            input_data_id_to_task_id: keyが`input_data_id`、valueが`task_id`のdict。Noneの場合、`task_id`は`input_data_id`と同じ値だとみなして変換します。
            input_data_name_to_input_data_id: keyが`input_data_name`、valueが`input_data_id`のdict。Noneの場合、`input_data_name`は`input_data_id`と同じ値だとみなして変換します。

        """
        output_dir.mkdir(exist_ok=True, parents=True)
        success_image_count = 0
        skipped_image_count = 0
        total_target_coco_annotation_count = 0
        logger.info(f"COCOデータセットの{len(self.coco_images)}件のimagesに紐づくアノテーションを、Annofab形式に変換します。")

        for image_index, coco_image in enumerate(self.coco_images):
            if (image_index + 1) % 1000 == 0:
                logger.info(f"{image_index + 1}件目のCOCO imagesに紐づくアノテーションを、Annofabフォーマットに変換中...")

            image_file_name = coco_image["file_name"]
            af_input_data_id = input_data_name_to_input_data_id.get(image_file_name) if input_data_name_to_input_data_id is not None else image_file_name
            if af_input_data_id is None:
                logger.warning(f"Annofabのinput_data_name='{image_file_name}'に対応するinput_data_idが見つかりません。スキップします。")
                continue

            af_task_id = input_data_id_to_task_id.get(af_input_data_id) if input_data_id_to_task_id is not None else af_input_data_id
            if af_task_id is None:
                logger.warning(f"Annofabのinput_data_id='{af_input_data_id}'に対応するtask_idが見つかりません。スキップします。")
                continue

            resize_scale = IDENTITY_RESIZE_SCALE
            if input_data_name_to_input_data is not None:
                af_input_data = input_data_name_to_input_data.get(image_file_name)
                if af_input_data is None:
                    logger.warning(f"Annofabのinput_data_name='{image_file_name}'に対応する入力データ情報が見つかりません。スキップします。")
                    continue

                resize_scale = create_resize_scale_from_af_input_data(af_input_data)
                if not resize_scale.is_identity:
                    logger.debug(
                        f"Annofabのリサイズ後画像サイズに合わせてアノテーションを縮小します。 :: "
                        f"input_data_id='{af_input_data_id}', input_data_name='{image_file_name}', scale_x={resize_scale.x}, scale_y={resize_scale.y}, "
                        f"resized_width={resize_scale.width}, resized_height={resize_scale.height}"
                    )

            af_annotation_json = output_dir / af_task_id / f"{af_input_data_id}.json"
            try:
                af_details, target_coco_annotation_count = self.convert_annotations_to_af_details(
                    coco_image, af_input_data_dir=output_dir / af_task_id / af_input_data_id, resize_scale=resize_scale
                )
                if target_coco_annotation_count == 0:
                    skipped_image_count += 1
                    logger.debug(f"COCOのimage.file_name='{image_file_name}'に紐づく変換対象のアノテーションは存在しません。")
                    continue

                af_annotation_json.parent.mkdir(exist_ok=True, parents=True)
                af_annotation_json.write_text(json.dumps({"details": af_details}, ensure_ascii=False, indent=2))
                success_image_count += 1
                total_target_coco_annotation_count += target_coco_annotation_count
                message = (
                    f"COCOのimage.file_name='{image_file_name}'に紐づくアノテーション{target_coco_annotation_count}件を、Annofab形式に変換して、"
                    f"'{af_annotation_json}'に出力しました。 :: "
                    f"変換後のAnnofab形式のアノテーションは{len(af_details)}件です。"
                )
                if len(af_details) != target_coco_annotation_count:
                    message += "（マルチポリゴンが存在するので、COCOのアノテーション数と異なります）。"
                logger.debug(message)
            except Exception:
                logger.opt(exception=True).warning(f"COCOのimage.file_name='{image_file_name}'に紐づくアノテーションを、Annofabフォーマットへ変換するのに失敗しました。")
                continue

        logger.info(
            f"{success_image_count}/{len(self.coco_images)}件のCOCOデータセットimagesに紐づくアノテーション{total_target_coco_annotation_count}件を、Annofabフォーマットに変換しました。"
            f"{skipped_image_count}件のCOCOデータセットのimagesは、アノテーションが存在しなかったためスキップしました。"
            f" :: output_dir='{output_dir}'"
        )


def create_input_data_id_to_task_id_mapping(task_list: list[dict[str, Any]], *, target_input_data_ids: Collection[str] | None = None) -> dict[str, str]:
    """
    Annofabのタスク全件ファイルから、input_data_idとtask_idのマッピングを作成します。

    Args:
        task_list: Annofabのタスク全件情報
        target_input_data_ids: マッピング作成対象のinput_data_id。Noneの場合はすべてのinput_data_idを対象にします。

    Returns:
        keyが`input_data_id`、valueが`task_id`の辞書

    Raises:
        ValueError: 1個の入力データが複数のタスクから参照されている
    """
    target_input_data_id_set = set(target_input_data_ids) if target_input_data_ids is not None else None
    result = {}
    for task in task_list:
        for input_data_id in task["input_data_id_list"]:
            if target_input_data_id_set is not None and input_data_id not in target_input_data_id_set:
                continue

            task_id = task["task_id"]
            if input_data_id in result:
                raise ValueError(f"input_data_id='{input_data_id}'の入力データは複数のタスクに含まれています。入力データは1個のタスクのみ含まれるように変更してください。")

            result[input_data_id] = task_id
    return result


def create_input_data_name_to_input_data_id_mapping(input_data_list: list[dict[str, Any]]) -> dict[str, str]:
    """
    Annofabの入力データ全件ファイルから、input_data_nameからinput_data_idのマッピングを作成します。

    Returns:
        keyが`input_data_name`、valueが`input_data_id`の辞書

    Raises:
        ValueError: 1個の入力データが複数のタスクから参照されている
    """
    result = {}
    for item in input_data_list:
        input_data_name = item["input_data_name"]
        input_data_id = item["input_data_id"]
        if input_data_name in result:
            raise ValueError(f"input_data_name='{input_data_name}'の入力データが複数存在します。input_data_nameが重複しないようにしてください。")

        result[input_data_name] = input_data_id
    return result


def create_input_data_name_to_input_data_mapping(input_data_list: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Annofabの入力データ全件ファイルから、input_data_nameから入力データ情報のマッピングを作成します。
    """
    result = {}
    for item in input_data_list:
        input_data_name = item["input_data_name"]
        if input_data_name in result:
            raise ValueError(f"input_data_name='{input_data_name}'の入力データが複数存在します。input_data_nameが重複しないようにしてください。")

        result[input_data_name] = item
    return result


def get_target_input_data_ids(coco_images: list[dict[str, Any]], input_data_name_to_input_data_id: dict[str, str] | None) -> set[str]:
    """
    COCOのimagesから、Annofab形式への変換で参照するinput_data_idの集合を取得します。

    Args:
        coco_images: COCO形式のimages情報。各要素の`file_name`を参照します。
        input_data_name_to_input_data_id: keyがAnnofabの`input_data_name`、valueが`input_data_id`の辞書。
            Noneの場合、COCOの`image.file_name`を`input_data_id`とみなします。

    Returns:
        変換対象のinput_data_idの集合。`input_data_name_to_input_data_id`に対応する値が存在しないCOCO imageは除外します。
    """
    result = set()
    for coco_image in coco_images:
        image_file_name = coco_image["file_name"]
        af_input_data_id = input_data_name_to_input_data_id.get(image_file_name) if input_data_name_to_input_data_id is not None else image_file_name
        if af_input_data_id is None:
            continue
        result.add(af_input_data_id)
    return result


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="COCOデータセット（Instances）に含まれるアノテーションを、Annofab形式に変換します。"
        "出力結果は`annofabcli annotation import`コマンドでアノテーションを登録できます。"
        "COCOのimage.file_nameはAnnofabのinput_data_name, COCOのcategory.nameはAnnofabのラベル名(英語)として変換します。",
        parents=[create_parent_parser()],
    )

    parser.add_argument(
        "--coco_instances_json", type=Path, required=True, help="入力情報であるCOCOデータセット（Instances）形式アノテーションのJSONファイルのパス。`annotations`,`images`,`categories`を参照します。"
    )

    parser.add_argument(
        "--af_task_json",
        type=Path,
        required=False,
        help="Annofabのタスク全件ファイルのパス。"
        "`task_id`と`input_data_id`の関係を参照するのに利用します。"
        "未指定の場合は、`task_id`は`input_data_id`と同じ値だとみなして変換します。"
        "`annofabcli task download`コマンドでダウンロードできます。"
        "ダウンロードしたタスク全件ファイルに、作成したタスクの情報が含まれていない場合は、`--latest`オプションを付与して、最新のタスク全件ファイルをダウンロードしてください。",
    )

    parser.add_argument(
        "--af_input_data_json",
        type=Path,
        required=False,
        help="Annofabの入力データ全件ファイルのパス。`input_data_name`と`input_data_id`の関係を参照するのに利用します。"
        "未指定の場合は、`input_data_id`は`input_data_name`と同じ値だとみなして変換します。"
        "`annofabcli input_data download`コマンドでダウンロードできます。",
    )

    parser.add_argument(
        "--coco_annotation_type",
        type=str,
        required=True,
        choices=[e.value for e in CocoAnnotationType],
        default=CocoAnnotationType.BBOX.value,
        help="変換対象のアノテーションの種類。`bbox`:バウンディングボックス, `polygon_segmentation`:`iscrowd=0`のポリゴン形式のsegmentation, `rle_segmentation`:`iscrowd=1`のRLE形式のsegmentation",
    )

    parser.add_argument("--coco_image_file_name", type=str, nargs="+", help="変換対象のCOCOのimageのfile_name")
    parser.add_argument("--coco_category_name", type=str, nargs="+", help="変換対象のCOCOのcategory_name")

    parser.add_argument("-o", "--output_dir", type=Path, required=True, help="Annofab形式のアノテーションの出力先ディレクトリのパス")

    return parser


@log_exception()
def main() -> None:
    args = create_parser().parse_args()
    configure_loguru(is_verbose=args.verbose)
    logger.info(f"argv={sys.argv}")

    coco_instances = json.loads(args.coco_instances_json.read_text())

    af_input_data_list = json.loads(args.af_input_data_json.read_text()) if args.af_input_data_json is not None else None
    input_data_name_to_input_data_id = create_input_data_name_to_input_data_id_mapping(af_input_data_list) if af_input_data_list is not None else None
    input_data_name_to_input_data = create_input_data_name_to_input_data_mapping(af_input_data_list) if af_input_data_list is not None else None
    converter = AnnotationConverterFromCocoToAnnofab(
        coco_instances, CocoAnnotationType(args.coco_annotation_type), target_coco_category_names=args.coco_category_name, target_coco_image_file_names=args.coco_image_file_name
    )
    target_input_data_ids = get_target_input_data_ids(converter.coco_images, input_data_name_to_input_data_id)
    input_data_id_to_task_id = (
        create_input_data_id_to_task_id_mapping(json.loads(args.af_task_json.read_text()), target_input_data_ids=target_input_data_ids) if args.af_task_json is not None else None
    )
    converter.convert(
        args.output_dir,
        input_data_id_to_task_id=input_data_id_to_task_id,
        input_data_name_to_input_data_id=input_data_name_to_input_data_id,
        input_data_name_to_input_data=input_data_name_to_input_data,
    )


if __name__ == "__main__":
    main()
