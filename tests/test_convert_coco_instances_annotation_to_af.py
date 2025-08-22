import json
import tempfile
from pathlib import Path
from typing import Any

import numpy
import pytest

from src.convert_coco_instances_annotation_to_af import (
    AnnotationConverterFromCocoToAnnofab,
    CocoAnnotationType,
    convert_coco_one_segmentation_to_af_format,
    create_input_data_id_to_task_id_mapping,
    create_input_data_name_to_input_data_id_mapping,
)


def test_convert_coco_one_segmentation_to_af_format():
    """convert_coco_one_segmentation_to_af_format関数のテスト"""
    # テスト用のポリゴンデータ
    polygon_segmentation = [10, 20, 30, 40, 50, 60]

    # 関数実行
    result = convert_coco_one_segmentation_to_af_format(polygon_segmentation)

    # 結果の検証
    expected = {"points": [{"x": 10, "y": 20}, {"x": 30, "y": 40}, {"x": 50, "y": 60}], "_type": "Points"}
    assert result == expected


def test_create_input_data_id_to_task_id_mapping():
    """create_input_data_id_to_task_id_mapping関数のテスト"""
    # テスト用のデータ
    task_list = [
        {"task_id": "task1", "input_data_id_list": ["data1", "data2"]},
        {"task_id": "task2", "input_data_id_list": ["data3"]},
    ]

    # 関数実行
    result = create_input_data_id_to_task_id_mapping(task_list)

    # 結果の検証
    expected = {"data1": "task1", "data2": "task1", "data3": "task2"}
    assert result == expected


def test_create_input_data_id_to_task_id_mapping_error():
    """create_input_data_id_to_task_id_mapping関数のエラーケースのテスト"""
    # 1つの入力データが複数のタスクから参照されているケース
    task_list = [
        {"task_id": "task1", "input_data_id_list": ["data1", "data2"]},
        {"task_id": "task2", "input_data_id_list": ["data2", "data3"]},  # data2が重複
    ]

    # 例外が発生することを検証
    with pytest.raises(ValueError):
        create_input_data_id_to_task_id_mapping(task_list)


def test_create_input_data_name_to_input_data_id_mapping():
    """create_input_data_name_to_input_data_id_mapping関数のテスト"""
    # テスト用のデータ
    input_data_list = [
        {"input_data_id": "id1", "input_data_name": "name1"},
        {"input_data_id": "id2", "input_data_name": "name2"},
    ]

    # 関数実行
    result = create_input_data_name_to_input_data_id_mapping(input_data_list)

    # 結果の検証
    expected = {"name1": "id1", "name2": "id2"}
    assert result == expected


def test_create_input_data_name_to_input_data_id_mapping_error():
    """create_input_data_name_to_input_data_id_mapping関数のエラーケースのテスト"""
    # input_data_nameが重複するケース
    input_data_list = [
        {"input_data_id": "id1", "input_data_name": "same_name"},
        {"input_data_id": "id2", "input_data_name": "same_name"},  # input_data_nameが重複
    ]

    # 例外が発生することを検証
    with pytest.raises(ValueError):
        create_input_data_name_to_input_data_id_mapping(input_data_list)


class TestAnnotationConverterFromCocoToAnnofab:
    """AnnotationConverterFromCocoToAnnofabクラスのテスト"""

    coco_instances: dict[str, Any]

    @classmethod
    def setup_class(cls):
        """テスト全体の前処理"""

        # テスト用のCOCOデータを読み込む
        cls.coco_instances = json.loads(Path("tests/resources/test_coco_instances.json").read_text())

    def test_init(self):
        """初期化のテスト"""
        # コンバーター初期化
        converter = AnnotationConverterFromCocoToAnnofab(self.coco_instances, CocoAnnotationType.BBOX)

        # 初期化後のプロパティを検証
        assert len(converter.coco_images) == 2
        assert converter.target_coco_category_names is None
        assert len(converter.annotations_by_image_id) == 2
        assert len(converter.annotations_by_image_id[1]) == 2  # image_id=1に紐づくアノテーション数
        assert len(converter.annotations_by_image_id[2]) == 1  # image_id=2に紐づくアノテーション数
        assert converter.category_names_by_id[1] == "person"
        assert converter.category_names_by_id[2] == "car"
        assert converter.category_names_by_id[3] == "cat"

    def test_init_with_filters(self):
        """フィルターを指定した初期化のテスト"""
        # 特定のカテゴリとイメージファイル名でフィルタリング
        converter = AnnotationConverterFromCocoToAnnofab(self.coco_instances, CocoAnnotationType.BBOX, target_coco_category_names=["person"], target_coco_image_file_names=["test_image1.jpg"])

        # フィルタリング後の状態を検証
        assert len(converter.coco_images) == 1
        assert converter.coco_images[0]["file_name"] == "test_image1.jpg"
        assert converter.target_coco_category_names == {"person"}

    def test_convert_bbox_annotation_to_af_detail(self):
        """BBoxアノテーション変換のテスト"""
        converter = AnnotationConverterFromCocoToAnnofab(self.coco_instances, CocoAnnotationType.BBOX)

        # BBOX アノテーションを取得
        coco_annotation = self.coco_instances["annotations"][0]  # person, bbox

        # 変換実行
        result = converter.convert_bbox_annotation_to_af_detail(coco_annotation)

        # 結果検証
        assert result is not None
        assert result["label"] == "person"
        assert result["attributes"]["coco.annotation_id"] == 1
        assert result["attributes"]["coco.image_id"] == 1
        assert result["data"]["_type"] == "BoundingBox"
        assert result["data"]["left_top"] == {"x": 200, "y": 200}
        assert result["data"]["right_bottom"] == {"x": 300, "y": 250}

    def test_convert_bbox_annotation_to_af_detail_filtered(self):
        """BBoxアノテーション変換のフィルターテスト"""
        # 特定のカテゴリでフィルタリング
        converter = AnnotationConverterFromCocoToAnnofab(
            self.coco_instances,
            CocoAnnotationType.BBOX,
            target_coco_category_names=["car"],  # personは対象外
        )

        # personのアノテーション
        coco_annotation = self.coco_instances["annotations"][0]  # person, bbox

        # 変換実行（対象外なのでNoneが返る）
        result = converter.convert_bbox_annotation_to_af_detail(coco_annotation)
        assert result is None

    def test_convert_polygon_segmentation_annotation_to_af_detail(self):
        """ポリゴンセグメンテーションアノテーション変換のテスト"""
        converter = AnnotationConverterFromCocoToAnnofab(self.coco_instances, CocoAnnotationType.POLYGON_SEGMENTATION)

        # ポリゴンアノテーション（iscrowd=0）
        coco_annotation = self.coco_instances["annotations"][0]  # person, polygon

        # 変換実行
        result = converter.convert_polygon_segmentation_annotation_to_af_detail(coco_annotation)

        # 結果検証
        assert len(result) == 1  # 1つのセグメンテーション
        polygon_detail = result[0]
        assert polygon_detail["label"] == "person"
        assert polygon_detail["attributes"]["coco.annotation_id"] == 1
        assert polygon_detail["attributes"]["coco.image_id"] == 1
        assert polygon_detail["data"]["_type"] == "Points"
        assert len(polygon_detail["data"]["points"]) == 3

    def test_convert_annotations_to_af_details_bbox(self):
        """convert_annotations_to_af_detailsメソッドのBBox変換テスト"""
        converter = AnnotationConverterFromCocoToAnnofab(self.coco_instances, CocoAnnotationType.BBOX)

        # テスト用のパス
        temp_dir = Path(tempfile.mkdtemp())

        # 変換実行
        coco_image = self.coco_instances["images"][0]  # test_image1.jpg
        details, count = converter.convert_annotations_to_af_details(coco_image, temp_dir)

        # 結果検証
        assert len(details) == 2  # image_id=1には2つのアノテーションがある
        assert count == 2
        assert details[0]["label"] == "person"
        assert details[1]["label"] == "car"

    def test_convert_annotations_to_af_details_polygon(self):
        """convert_annotations_to_af_detailsメソッドのポリゴン変換テスト"""
        converter = AnnotationConverterFromCocoToAnnofab(self.coco_instances, CocoAnnotationType.POLYGON_SEGMENTATION)

        # テスト用のパス
        temp_dir = Path(tempfile.mkdtemp())

        # 変換実行
        coco_image = self.coco_instances["images"][0]  # test_image1.jpg
        details, count = converter.convert_annotations_to_af_details(coco_image, temp_dir)

        # 結果検証
        assert len(details) == 1  # ポリゴンは1つ（RLEは対象外）
        assert count == 1
        assert details[0]["label"] == "person"


class TestConvertRLESegmentation:
    """RLEセグメンテーションの変換テスト"""

    coco_instances: dict[str, Any]
    converter: AnnotationConverterFromCocoToAnnofab

    @classmethod
    def setup_class(cls):
        """テスト全体の前処理"""
        # テスト用のCOCOデータを読み込む
        cls.coco_instances = json.loads(Path("tests/resources/test_coco_instances.json").read_text())
        # コンバーターを初期化
        cls.converter = AnnotationConverterFromCocoToAnnofab(cls.coco_instances, CocoAnnotationType.RLE_SEGMENTATION)

    def test_convert_rle_segmentation_annotation_to_af_detail(self):
        """convert_rle_segmentation_annotation_to_af_detailメソッドのテスト"""
        # RLE形式のアノテーション (iscrowd=1)
        coco_annotation = self.coco_instances["annotations"][1]  # car, RLE
        coco_image = self.coco_instances["images"][0]  # test_image1.jpg

        # 変換実行
        af_detail, segmentation_bool_array = self.converter.convert_rle_segmentation_annotation_to_af_detail(coco_annotation, coco_image)

        # 結果検証
        assert af_detail is not None
        assert af_detail["label"] == "car"
        assert af_detail["attributes"]["coco.annotation_id"] == 2
        assert af_detail["attributes"]["coco.image_id"] == 1
        assert af_detail["data"]["_type"] == "Segmentation"
        assert af_detail["data"]["data_uri"] == af_detail["annotation_id"]

        # bool配列の検証
        assert segmentation_bool_array is not None
        assert isinstance(segmentation_bool_array, numpy.ndarray)
        assert segmentation_bool_array.shape == (427, 640)  # 画像サイズと同じ
        assert segmentation_bool_array.dtype == bool  # bool型であること

    def test_convert_rle_segmentation_annotation_to_af_detail_not_rle(self):
        """RLEでないアノテーションに対するテスト (iscrowd=0)"""
        # ポリゴン形式のアノテーション (iscrowd=0)
        coco_annotation = self.coco_instances["annotations"][0]  # person, polygon
        coco_image = self.coco_instances["images"][0]  # test_image1.jpg

        # 変換実行 (RLEでないのでNoneが返る)
        af_detail, segmentation_bool_array = self.converter.convert_rle_segmentation_annotation_to_af_detail(coco_annotation, coco_image)

        # 結果検証
        assert af_detail is None
        assert segmentation_bool_array is None

    def test_convert_rle_segmentation_annotation_to_af_detail_filtered(self):
        """フィルタリングされたカテゴリのテスト"""
        # personのみを対象とするフィルタを適用
        filtered_converter = AnnotationConverterFromCocoToAnnofab(
            self.coco_instances,
            CocoAnnotationType.RLE_SEGMENTATION,
            target_coco_category_names=["person"],  # carは対象外
        )

        # RLE形式のアノテーション (car, iscrowd=1)
        coco_annotation = self.coco_instances["annotations"][1]
        coco_image = self.coco_instances["images"][0]

        # 変換実行 (carはフィルタで除外されるのでNoneが返る)
        af_detail, segmentation_bool_array = filtered_converter.convert_rle_segmentation_annotation_to_af_detail(coco_annotation, coco_image)

        # 結果検証
        assert af_detail is None
        assert segmentation_bool_array is None

    def test_convert_annotations_to_af_details_rle(self, tmp_path: Path):
        """convert_annotations_to_af_detailsメソッドのRLE変換テスト"""
        # テスト用の一時ディレクトリ

        # 変換実行
        coco_image = self.coco_instances["images"][0]  # test_image1.jpg
        details, count = self.converter.convert_annotations_to_af_details(coco_image, tmp_path)

        # 結果検証
        assert len(details) == 1  # RLEは1つ
        assert count == 1
        assert details[0]["label"] == "car"

        # 塗りつぶし画像が生成されたか確認
        annotation_id = details[0]["annotation_id"]
        segmentation_file = tmp_path / annotation_id
        assert segmentation_file.exists()  # 塗りつぶし画像が生成されていること
