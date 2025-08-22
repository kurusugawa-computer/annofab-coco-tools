import numpy as np

from src.convert_af_annotation_to_coco_instances import (
    AnnotationConverterFromAnnofabToCoco,
    clip_bounding_box_to_image,
    clip_polygon_to_image,
    get_rle_from_boolean_segmentation_array,
)


class TestClipBoundingBoxToImage:
    def test_no_clipping_needed(self):
        """画像内のバウンディングボックスはクリッピングされない"""
        left_top = {"x": 10, "y": 20}
        right_bottom = {"x": 50, "y": 60}
        image_width = 100
        image_height = 100

        new_left_top, new_right_bottom = clip_bounding_box_to_image(left_top, right_bottom, image_width, image_height)

        # クリッピング不要なので同じ値が返る
        assert new_left_top == left_top
        assert new_right_bottom == right_bottom

    def test_clip_outside_image(self):
        """画像外のバウンディングボックスがクリッピングされる"""
        left_top = {"x": -10, "y": -20}
        right_bottom = {"x": 150, "y": 160}
        image_width = 100
        image_height = 100

        new_left_top, new_right_bottom = clip_bounding_box_to_image(left_top, right_bottom, image_width, image_height)

        # クリッピングされた値が返る
        assert new_left_top == {"x": 0, "y": 0}
        assert new_right_bottom == {"x": 100, "y": 100}

    def test_clip_partially_outside_image(self):
        """一部が画像外のバウンディングボックスが適切にクリッピングされる"""
        left_top = {"x": -10, "y": 20}
        right_bottom = {"x": 50, "y": 160}
        image_width = 100
        image_height = 100

        new_left_top, new_right_bottom = clip_bounding_box_to_image(left_top, right_bottom, image_width, image_height)

        # 一部クリッピングされた値が返る
        assert new_left_top == {"x": 0, "y": 20}
        assert new_right_bottom == {"x": 50, "y": 100}


class TestClipPolygonToImage:
    def test_no_clipping_needed(self):
        """画像内のポリゴンはクリッピングされない"""
        points = [
            {"x": 10, "y": 20},
            {"x": 50, "y": 20},
            {"x": 50, "y": 60},
            {"x": 10, "y": 60},
        ]
        image_width = 100
        image_height = 100

        new_points = clip_polygon_to_image(points, image_width, image_height)

        # クリッピング不要なので同じ値が返る
        assert new_points == points

    def test_clip_outside_image(self):
        """画像外のポイントがクリッピングされる"""
        points = [
            {"x": -10, "y": -20},
            {"x": 150, "y": -20},
            {"x": 150, "y": 160},
            {"x": -10, "y": 160},
        ]
        image_width = 100
        image_height = 100

        new_points = clip_polygon_to_image(points, image_width, image_height)

        # クリッピングされた値が返る
        expected_points = [
            {"x": 0, "y": 0},
            {"x": 100, "y": 0},
            {"x": 100, "y": 100},
            {"x": 0, "y": 100},
        ]
        assert new_points == expected_points

    def test_clip_partially_outside_image(self):
        """一部が画像外のポイントが適切にクリッピングされる"""
        points = [
            {"x": -10, "y": 20},
            {"x": 50, "y": 20},
            {"x": 50, "y": 160},
            {"x": -10, "y": 160},
        ]
        image_width = 100
        image_height = 100

        new_points = clip_polygon_to_image(points, image_width, image_height)

        # 一部クリッピングされた値が返る
        expected_points = [
            {"x": 0, "y": 20},
            {"x": 50, "y": 20},
            {"x": 50, "y": 100},
            {"x": 0, "y": 100},
        ]
        assert new_points == expected_points


class TestAnnotationConverterFromAnnofabToCoco:
    def test_init(self):
        """コンストラクタのテスト"""
        # テスト用のカテゴリとイメージデータ
        coco_categories = [
            {"id": 1, "name": "label1"},
            {"id": 2, "name": "label2"},
        ]
        coco_images = [
            {"id": 1, "file_name": "image1.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "image2.jpg", "width": 200, "height": 200},
        ]

        # インスタンス作成
        converter = AnnotationConverterFromAnnofabToCoco(
            coco_categories=coco_categories,
            coco_images=coco_images,
            target_af_target_labels=["label1"],
            should_clip_annotation_to_image=True,
        )

        # インスタンス変数の検証
        assert converter.category_ids_by_name == {"label1": 1, "label2": 2}
        assert converter.images_by_file_name == {
            "image1.jpg": coco_images[0],
            "image2.jpg": coco_images[1],
        }
        assert converter.target_af_target_labels == {"label1"}
        assert converter.should_clip_annotation_to_image is True

    def test_convert_af_bounding_box_detail(self):
        """バウンディングボックスの変換テスト"""
        # テスト用のカテゴリとイメージデータ
        coco_categories = [
            {"id": 1, "name": "car"},
        ]
        coco_images = [
            {"id": 1, "file_name": "image1.jpg", "width": 100, "height": 100},
        ]

        # インスタンス作成
        converter = AnnotationConverterFromAnnofabToCoco(
            coco_categories=coco_categories,
            coco_images=coco_images,
        )

        # テスト用のAnnofabのbounding boxデータ
        af_detail = {
            "annotation_id": "12345",
            "label": "car",
            "data": {
                "_type": "BoundingBox",
                "left_top": {"x": 10, "y": 20},
                "right_bottom": {"x": 50, "y": 60},
            },
        }

        # 変換実行
        coco_annotation = converter.convert_af_bounding_box_detail(af_detail, coco_images[0], coco_annotation_id=1)

        # 結果の検証
        assert coco_annotation["id"] == 1
        assert coco_annotation["image_id"] == 1
        assert coco_annotation["category_id"] == 1
        assert coco_annotation["bbox"] == [10, 20, 40, 40]  # [x, y, width, height]
        assert coco_annotation["area"] == 1600  # width * height = 40 * 40 = 1600
        assert coco_annotation["iscrowd"] == 0
        # バウンディングボックスのsegmentationは四角形の頂点座標のリスト
        assert coco_annotation["segmentation"] == [[10, 20, 50, 20, 50, 60, 10, 60]]

    def test_convert_af_polygon_detail(self):
        """ポリゴンの変換テスト"""
        # テスト用のカテゴリとイメージデータ
        coco_categories = [
            {"id": 1, "name": "car"},
        ]
        coco_images = [
            {"id": 1, "file_name": "image1.jpg", "width": 100, "height": 100},
        ]

        # インスタンス作成
        converter = AnnotationConverterFromAnnofabToCoco(
            coco_categories=coco_categories,
            coco_images=coco_images,
        )

        # テスト用のAnnofabのpolygonデータ
        af_detail = {
            "annotation_id": "12345",
            "label": "car",
            "data": {
                "_type": "Points",
                "points": [
                    {"x": 10, "y": 20},
                    {"x": 50, "y": 20},
                    {"x": 50, "y": 60},
                    {"x": 10, "y": 60},
                ],
            },
        }

        # 変換実行
        coco_annotation = converter.convert_af_polygon_detail(af_detail, coco_images[0], coco_annotation_id=1)

        # 結果の検証
        assert coco_annotation["id"] == 1
        assert coco_annotation["image_id"] == 1
        assert coco_annotation["category_id"] == 1
        assert coco_annotation["bbox"] == [10, 20, 40, 40]  # [x, y, width, height]
        assert coco_annotation["iscrowd"] == 0
        # ポリゴンのsegmentationは頂点座標を一次元配列にしたもの
        assert coco_annotation["segmentation"] == [[10, 20, 50, 20, 50, 60, 10, 60]]
        # 面積は計算された値と一致すること
        assert coco_annotation["area"] == 1600  # 40 * 40 = 1600


class TestGetRleFromBooleanSegmentationArray:
    def test_get_rle_uncompressed(self):
        """非圧縮RLE形式に変換するテスト"""
        # 2x3の2次元配列を作成（heightが2, widthが3）
        segmentation_array = np.array(
            [
                [False, True, False],
                [True, True, False],
            ],
            dtype=bool,
        )

        rle = get_rle_from_boolean_segmentation_array(segmentation_array)

        # 結果の検証
        assert rle["size"] == [2, 3]  # [height, width]
        # run-length countingの結果
        # 最初のFalseが1個、次にTrueが3個、Falseが2個
        assert rle["counts"] == [1, 3, 2]

    def test_empty_array(self):
        """空の配列のテスト"""
        segmentation_array = np.zeros((2, 3), dtype=bool)  # heightが2, widthが3

        # 非圧縮RLE
        rle_uncompressed = get_rle_from_boolean_segmentation_array(segmentation_array)
        assert rle_uncompressed["size"] == [2, 3]
        assert rle_uncompressed["counts"][0] == 6  # すべてFalseなので、最初のcountは2x3=6
