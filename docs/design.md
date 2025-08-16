
# Annofab形式のアノテーションをCOCOデータセット形式に変換するスクリプト


## 引数
* Annofabのproject_id: required
    * AnnofabのアノテーションZIPと入力データ全件ファイル（画像のサイズが記載されている）をダウンロードする。
    * 毎回ダウンロード処理が発生するのよくないので、アノテーションZIPと入力データ全件ファイルを渡せるようにする
* COCOデータセットのInstances Annotation File
    * categories: 必須
    * images: 入力データ全件ファイルが指定されていなければ必須


    