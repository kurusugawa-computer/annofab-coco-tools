
import numpy as np

from src.convert_af_annotation_to_coco_instances import (
    get_rle_from_boolean_segmentation_array,
)


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

        rle = get_rle_from_boolean_segmentation_array(segmentation_array, is_compressed=False)

        # 結果の検証
        assert rle["size"] == [2, 3]  # [height, width]
        # run-length countingの結果
        # 最初のFalseが1個、次にTrueが3個、Falseが2個
        assert rle["counts"] == [1, 3, 2]

    def test_get_rle_compressed(self):
        """圧縮RLE形式に変換するテスト"""
        # 2x3の2次元配列を作成（heightが2, widthが3）
        segmentation_array = np.array(
            [
                [False, True, False],
                [False, False, True],
            ],
            dtype=bool,
        )

        rle = get_rle_from_boolean_segmentation_array(segmentation_array, is_compressed=True)

        # 圧縮RLE形式は、pycocotools.mask.encodeの仕様に依存するため、
        # サイズだけを確認する
        assert rle["size"] == [2, 3]  # [height, width]
        print(rle)
        assert isinstance(rle["counts"], str)  # countsはstr型であることを確認

    def test_empty_array(self):
        """空の配列のテスト"""
        segmentation_array = np.zeros((2, 3), dtype=bool)  # heightが2, widthが3

        # 非圧縮RLE
        rle_uncompressed = get_rle_from_boolean_segmentation_array(segmentation_array, is_compressed=False)
        assert rle_uncompressed["size"] == [2, 3]
        assert rle_uncompressed["counts"][0] == 6  # すべてFalseなので、最初のcountは2x3=6

        # 圧縮RLE
        rle_compressed = get_rle_from_boolean_segmentation_array(segmentation_array, is_compressed=True)
        assert rle_compressed["size"] == [2, 3]
        assert isinstance(rle_compressed["counts"], str)
