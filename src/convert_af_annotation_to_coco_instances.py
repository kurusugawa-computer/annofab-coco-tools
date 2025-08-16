import collections
import json
import sys
import zipfile
from collections.abc import Collection
from pathlib import Path
from typing import Any

from annofabapi.parser import lazy_parse_simple_annotation_dir, lazy_parse_simple_annotation_zip
from jsonargparse import ArgumentParser
from loguru import logger

from src.common.cli import create_parent_parser
from src.common.utils import configure_loguru, log_exception


def clip_bounding_box_to_image(
    left_top: dict[str, int],
    right_bottom: dict[str, int],
    image_width: int,
    image_height: int,
) -> tuple[dict[str, int], dict[str, int]]:
    """
    境界ボックス（bbox）が画像からはみ出ていないかチェックし、はみ出ていれば修正します。

    Args:
        left_top: 左上座標（Annofab形式）
        right_bottom: 右下座標（Annofab形式）
        image_width: 画像の幅
        image_height: 画像の高さ

    Returns:
        修正後の左上座標、右下座標
    """
    new_left_top = {
        "x": max(left_top["x"], 0),
        "y": max(left_top["y"], 0),
    }
    new_right_bottom = {
        "x": min(right_bottom["x"], image_width),
        "y": min(right_bottom["y"], image_height),
    }
    return new_left_top, new_right_bottom


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

    # coco_image_idは1始まりにする
    coco_images = [convert_af_input_data_to_coco_image(af_input_data, coco_image_id=i + 1) for i, af_input_data in enumerate(af_input_data_list)]
    return coco_images


class AnnotationConverterFromAnnofabToCoco:
    def __init__(
        self, coco_categories: list[dict[str, Any]], coco_images: list[dict[str, Any]], *, target_af_target_labels: Collection[str] | None = None, should_clip_annotation_to_image: bool = False
    ) -> None:
        self.category_ids_by_name: dict[str, int] = {category["name"]: category["id"] for category in coco_categories}
        self.images_by_file_name: dict[str, dict[str, Any]] = {image["file_name"]: image for image in coco_images}
        self.should_clip_annotation_to_image = should_clip_annotation_to_image

        self.target_af_target_labels = set(target_af_target_labels) if target_af_target_labels is not None else None

    def convert_af_bounding_box_detail(
        self, af_detail: dict[str, Any], coco_image: dict[str, Any], coco_annotation_id: int, *, task_id: str | None = None, input_data_id: str | None = None
    ) -> dict[str, Any]:
        """
        Bounding BoxのAnnofabのdetail情報をCOCO形式に変換します。
        """
        annotation_id = af_detail["annotation_id"]
        label = af_detail["label"]
        left_top = af_detail["data"]["left_top"].copy()
        right_bottom = af_detail["data"]["right_bottom"].copy()
        image_width = coco_image["width"]
        image_height = coco_image["height"]
        coco_image_id = coco_image["id"]

        if self.should_clip_annotation_to_image:
            # bboxが画像からはみ出ていないかチェックし、はみ出ていれば修正
            original_left_top = left_top
            original_right_bottom = right_bottom
            new_left_top, new_right_bottom = clip_bounding_box_to_image(
                original_left_top,
                original_right_bottom,
                image_width,
                image_height,
            )
            if new_left_top != left_top or new_right_bottom != right_bottom:
                logger.debug(
                    f"bboxが画像からはみ出ていたため修正しました。 :: "
                    f"task_id='{task_id}', input_data_id='{input_data_id}', annotation_id='{annotation_id}', label='{label}', "
                    f"coco_image_id='{coco_image_id}', coco_annotation_id='{coco_annotation_id}' :: "
                    f"original_left_top={original_left_top}, original_right_bottom={original_right_bottom}, "
                    f"new_left_top={new_left_top}, new_right_bottom={new_right_bottom}"
                )
            left_top = new_left_top
            right_bottom = new_right_bottom

        bbox = [left_top["x"], left_top["y"], right_bottom["x"] - left_top["x"], right_bottom["y"] - left_top["y"]]
        segmentation = [[left_top["x"], left_top["y"], right_bottom["x"], left_top["y"], right_bottom["x"], right_bottom["y"], left_top["x"], right_bottom["y"]]]
        area = bbox[2] * bbox[3]
        return {
            "id": coco_annotation_id,
            "image_id": coco_image["id"],
            "category_id": self.category_ids_by_name[label],
            "bbox": bbox,
            "segmentation": segmentation,
            "area": area,
            # TODO 属性値から算出する
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

            if af_detail["data"]["_type"] == "BoundingBox":
                coco_annotation = self.convert_af_bounding_box_detail(af_detail, coco_image, coco_annotation_id, task_id=task_id, input_data_id=input_data_id)
            else:
                continue

            coco_annotations.append(coco_annotation)
            coco_annotation_id += 1
        return coco_annotations, coco_annotation_id

    def convert_af_annotation_path(
        self,
        af_annotation_zip_or_dir: Path,
        *,
        target_task_ids: Collection[str] | None = None,
        target_input_data_ids: Collection[str] | None = None,
        target_task_phase: str | None,
        target_task_status: str | None,
    ) -> list[dict[str, Any]]:
        """
        AnnofabからダウンロードしたアノテーションZIPまたは展開したディレクトリを、COCO形式のアノテーションに変換します。

        Args:
            af_annotation_zip_or_dir: Annofab形式のアノテーションZIPファイルまたはそれを展開したディレクトリのパス
            target_task_ids: 変換対象のタスクのID
            target_input_data_ids: 変換対象の入力データのID
            target_task_phase: 変換対象のタスクのフェーズ
            target_task_status: 変換対象のタスクのステータス


        """
        coco_start_annotation_id = 1

        if zipfile.is_zipfile(af_annotation_zip_or_dir):
            iter_af_annotation_parser = lazy_parse_simple_annotation_zip(af_annotation_zip_or_dir)
        elif af_annotation_zip_or_dir.is_dir():
            iter_af_annotation_parser = lazy_parse_simple_annotation_dir(af_annotation_zip_or_dir)
        else:
            raise ValueError(f"'{af_annotation_zip_or_dir}'はZIPファイルでもディレクトリでもありません。")

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

        logger.info(f"Annofab形式のアノテーション'{af_annotation_zip_or_dir}'に含まれる{success_count}個のJSONファイルを、COCO形式のannotations（{len(coco_annotations)}個）に変換しました。")
        return coco_annotations


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Annofab形式のアノテーションを、COCOデータセット（Instances）形式に変換します。",
        parents=[create_parent_parser()],
    )

    parser.add_argument(
        "--af_annotation_zip_or_dir",
        type=Path,
        required=True,
        help="Annofab形式のアノテーションZIPファイルのパス。またはZIPファイルを展開したディレクトリのパス。`annofabcli annotation download`コマンドでアノテーションZIPファイルをダウンロードできます。",
    )

    parser.add_argument(
        "--af_input_data_json",
        type=Path,
        required=False,
        help="Annofabの入力データ全件ファイルのパス。COCO形式のimagesを生成するのに利用します。`annofabcli input_data download`コマンドでダウンロードできます。"
        "未指定の場合は、'--coco_instances_json'に指定したJSONファイルの'images'を利用します。",
    )

    parser.add_argument(
        "--coco_instances_json", type=Path, required=True, help="入力情報であるCOCOデータセット（Instances）形式アノテーションのJSONファイルのパス。`categories`と`images`(オプショナル)を参照します。"
    )

    parser.add_argument("-o", "--output_coco_instances_json", type=Path, required=True, help="変換後のCOCOデータセット（Instances）形式アノテーションの出力先JSONファイルのパス")

    parser.add_argument(
        "--clip_annotation_to_image",
        action="store_true",
        help="指定すると、アノテーションが画像からはみ出さないようにクリッピングします。Annofabは矩形やポリゴンは画像外に作図できます。ただし、塗りつぶしアノテーションは画像外に作図できません。",
    )

    parser.add_argument("--af_task_id", type=str, nargs="+", help="変換対象のAnnofabのタスクのID")
    parser.add_argument("--af_input_data_id", type=str, nargs="+", help="変換対象のAnnofabの入力データのID")
    parser.add_argument("--af_label_name", type=str, nargs="+", help="変換対象のAnnofabのラベル名（英語）")
    parser.add_argument("--af_task_phase", type=str, help="変換対象のAnnofabのタスクのフェーズ")
    parser.add_argument("--af_task_status", type=str, help="変換対象のAnnofabのタスクのステータス")

    return parser


@log_exception(logger=logger)
def main() -> None:
    args = create_parser().parse_args()
    configure_loguru(is_verbose=args.verbose)
    logger.info(f"argv={sys.argv}")

    coco_instances = json.loads(args.coco_instances_json.read_text())

    if args.af_input_data_json is not None:
        af_input_data_list = json.loads(args.af_input_data_json.read_text())
        coco_images = convert_af_input_data_list_to_coco_images(af_input_data_list)
        logger.info(f"'{args.af_input_data_json}'に格納されているAnnofabの入力データ {len(af_input_data_list)} 件を、COCO形式のimagesに変換しました。")
    else:
        coco_images = coco_instances["images"]
        logger.info(f"'{args.coco_instances_json}'に格納されているCOCO形式のimages（{len(coco_images)} 件）をそのまま利用します。")

    coco_categories = coco_instances["categories"]
    converter = AnnotationConverterFromAnnofabToCoco(
        coco_categories=coco_categories,
        coco_images=coco_images,
        target_af_target_labels=args.af_label_name,
        should_clip_annotation_to_image=args.clip_annotation_to_image,
    )

    coco_annotations = converter.convert_af_annotation_path(
        args.af_annotation_zip_or_dir, target_input_data_ids=args.af_input_data_id, target_task_ids=args.af_task_id, target_task_phase=args.af_task_phase, target_task_status=args.af_task_status
    )

    result_coco_instances = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories,
    }
    output_coco_instances_json = args.output_coco_instances_json
    output_coco_instances_json.parent.mkdir(exist_ok=True, parents=True)
    output_coco_instances_json.write_text(json.dumps(result_coco_instances, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
