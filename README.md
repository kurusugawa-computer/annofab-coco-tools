# annofab-coco-tools
Annofab形式のアノテーションとCOCOデータセット形式のアノテーションを相互に変換するツール群

# Requirements
* Python 3.12+

# Install
```
$ uv sync
```

# Usage

## COCOデータセット(Instances)をAnnofabに登録する

### Annofabに画像プロジェクトを作成する
https://annofab.readme.io/docs/create-project を参照して、画像プロジェクトを作成します。


### Annofabに入力データを作成する

COCOデータセットのimagesから、Annofabに入力データを作成します。
Annofabの入力データの`input_data_name`は、COCOデータセットの`image.file_name`を格納します。
`input_data_id`は、`input_data_name`とほとんど同じ値になります（IDに使えない文字は適切に変換されます）。

```
$ uv run python -m src.create_af_input_data --coco_instances_json  resources/coco_instances.json \
 --image_dir resources/images/ \
 --af_project_id ${AF_PROJECT_ID} \
 
```

#### Help

```
$ uv run python -m src.create_af_input_data --help
usage: create_af_input_data.py [-h] [--verbose] --coco_instances_json COCO_INSTANCES_JSON --image_dir IMAGE_DIR --af_project_id AF_PROJECT_ID
                               [--coco_image_file_name COCO_IMAGE_FILE_NAME [COCO_IMAGE_FILE_NAME ...]] [--temp_dir TEMP_DIR]

COCOデータセットのimagesから、Annofabに入力データを作成します。Annofabの入力データの`input_data_name`は、COCOデータセットの`image.file_name`を格納します。`input_data_id`は、`input_data_name`とほとんど
同じ値になります。

options:
  -h, --help            Show this help message and exit.
  --verbose             詳細なログを出力します。 (default: False)
  --coco_instances_json COCO_INSTANCES_JSON
                        入力情報であるCOCOデータセット形式アノテーションのJSONファイルのパス。`images`を参照します。 (required, type: <class 'Path'>)
  --image_dir IMAGE_DIR
                        COCOデータセットの画像ファイルが存在するディレクトリのパス。 (required, type: <class 'Path'>)
  --af_project_id AF_PROJECT_ID
                        AnnofabプロジェクトのID (required, type: str)
  --coco_image_file_name COCO_IMAGE_FILE_NAME [COCO_IMAGE_FILE_NAME ...]
                        作成対象のCOCOのimageのfile_name (type: str, default: null)
  --temp_dir TEMP_DIR   一時ディレクトリのパス (type: <class 'Path'>, default: null)
```

### Annofabにタスクを作成する

Annofabにタスクを作成します。1個のタスクには1個の入力データが含まれています。task_idはinput_data_idと同じ値です。

```
# 入力データ全件ファイルをダウンロードする
# 入力データ作成直後は、作成した入力データがファイルに反映されないので、`--latest`オプションを指定する
$ uv run annofabcli input_data download --project_id ${AF_PROJECT_ID} --output out/af_input_data.json --latest

$ uv run python -m src.create_af_task --af_input_data_json out/af_input_data.json \
 --af_project_id ${AF_PROJECT_ID} 
 
```

#### Help

```
$ uv run python -m src.create_af_task --help
usage: create_af_task.py [-h] [--verbose] --af_project_id AF_PROJECT_ID (--af_input_data_json AF_INPUT_DATA_JSON |
                         --af_input_data_id AF_INPUT_DATA_ID [AF_INPUT_DATA_ID ...]) [--temp_dir TEMP_DIR]

Annofabにタスクを作成します。1個のタスクには1個の入力データが含まれています。task_idはinput_data_idと同じ値です。

options:
  -h, --help            show this help message and exit
  --verbose             詳細なログを出力します。
  --af_project_id AF_PROJECT_ID
                        AnnofabプロジェクトのID
  --af_input_data_json AF_INPUT_DATA_JSON
                        Annofabの入力データ全件ファイルのパス。`input_data_id`を参照するのに利用します。`annofabcli input_datadownload`コマンドでダウンロードできます。ダウンロードした入力データ全件ファイルに、作成した入力データの情報が含まれていない場合は、`--latest`オプションを付与して、最新の入力データ全件ファイルをダウンロードしてください。
  --af_input_data_id AF_INPUT_DATA_ID [AF_INPUT_DATA_ID ...]
                        指定した`input_data_id`からタスクを作成します。
  --temp_dir TEMP_DIR   一時ディレクトリのパス
```

### COCOデータセットのannotationをAnnofab形式に変換する


```
# タスク全件ファイルをダウンロードする
# タスク作成直後は、作成したタスクがファイルに反映されないので、`--latest`オプションを指定する
$ uv run annofabcli task download --project_id ${AF_PROJECT_ID} --output out/af_task.json --latest

$ uv run python -m src.convert_coco_instances_annotation_to_af --coco_instances_json  resources/coco_instances.json \
 --af_task_json out/af_task.json \
 --af_input_data_json out/af_input_data.json \
 --coco_annotation_type bbox \
 --output_dir out/af_annotation/

$ tree out/af_annotation 
out/af_annotation 
└── 000000037777.jpg
    └── 000000037777.jpg.json
```


`--coco_annotation_type`には以下の値を選択できます。

* `bbox`
* `polygon_segmentation`：`iscrowd==0`のポリゴン形式のsegmentation。ただしAnnofabはマルチポリゴンに対応していないので、マルチポリゴンは複数のインスタンスに分かれてAnnofabに登録されます。
* `rle_segmentation`：`iscrowd==1`のRLE形式のsegmentation


#### Help
```
$ uv run python -m src.convert_coco_instances_annotation_to_af --help
usage: convert_coco_instances_annotation_to_af.py [-h] [--verbose] --coco_instances_json COCO_INSTANCES_JSON [--af_task_json AF_TASK_JSON]
                                                  [--af_input_data_json AF_INPUT_DATA_JSON]
                                                  [--coco_annotation_type {bbox,polygon_segmentation,rle_segmentation}]
                                                  [--coco_image_file_name COCO_IMAGE_FILE_NAME [COCO_IMAGE_FILE_NAME ...]]
                                                  [--coco_category_name COCO_CATEGORY_NAME [COCO_CATEGORY_NAME ...]] [-o OUTPUT_DIR]

COCOデータセット（Instances）に含まれるアノテーションを、Annofab形式に変換します。出力結果は`annofabcli annotation
import`コマンドでアノテーションを登録できます。COCOのimage.file_nameはAnnofabのinput_data_name, COCOのcategory.nameはAnnofabのラベル名(英語)として変換します。

options:
  -h, --help            Show this help message and exit.
  --verbose             詳細なログを出力します。 (default: False)
  --coco_instances_json COCO_INSTANCES_JSON
                        入力情報であるCOCOデータセット（Instances）形式アノテーションのJSONファイルのパス。`annotations`,`images`,`categories`を参照します。 (required, type: <class'Path'>)
  --af_task_json AF_TASK_JSON
                        Annofabのタスク全件ファイルのパス。`task_id`と`input_data_id`の関係を参照するのに利用します。未指定の場合は、`task_id`は`input_data_id`と同じ値だとみなして変換します。`annofabcli task download`コマンドでダウンロードできます。ダウンロードしたタスク全件ファイルに、作成したタスクの情報が含まれていない場合は、`--latest`オプションを付与して、最新のタスク全件ファイルをダウンロードしてください。
                        (type: <class 'Path'>, default: null)
  --af_input_data_json AF_INPUT_DATA_JSON
                        Annofabの入力データ全件ファイルのパス。`input_data_name`と`input_data_id`の関係を参照するのに利用します。未指定の場合は、`input_data_id`は`input_data_name`と同じ値だとみなして変換します。`annofabcli input_data download`コマンドでダウンロードできます。 
                        (type: <class 'Path'>, default: null)
  --coco_annotation_type {bbox,polygon_segmentation,rle_segmentation}
                        変換対象のアノテーションの種類。`bbox`:バウンディングボックス, `polygon_segmentation`:`iscrowd=0`のポリゴン形式のsegmentation,
                        `rle_segmentation`:`iscrowd=1`のRLE形式のsegmentation (required, type: str, default: bbox)
  --coco_image_file_name COCO_IMAGE_FILE_NAME [COCO_IMAGE_FILE_NAME ...]
                        変換対象のCOCOのimageのfile_name (type: str, default: null)
  --coco_category_name COCO_CATEGORY_NAME [COCO_CATEGORY_NAME ...]
                        変換対象のCOCOのcategory_name (type: str, default: null)
  -o, --output_dir OUTPUT_DIR
```

### Annofabプロジェクにトアノテーション仕様を作成する

以下の通り、アノテーション仕様を作成します。

* COCOのcategory_nameを、Annofabのラベル名（英語）にする
* アノテーションの種類は、以下のいずれかにする
    * 矩形
    * ポリゴン
    * 塗りつぶし（塗りつぶしV2ではない）
* (任意) 以下の属性を追加する
    * 属性名（英語）が`annotation_id`で、属性の種類は「自由記述(1行)」
    * 属性名（英語）が`image_id`で、属性の種類は「自由記述(1行)」



### Annofab形式のアノテーションをAnnofabプロジェクトにインポートする


```
$ uv run annofabcli annotation import --project_id ${AF_PROJECT_ID} \
 --annotation out/af_annotation 
```



## Annofab形式のアノテーションをCOCO形式(Instances)に変換する

### Annofabからアノテーションをダウンロードする

```
$ uv run annfoabcli annotation download --output out/af_annotation.zip
```


### Annofab形式のアノテーションをCOCO形式（Instances）に変換する


```
$ uv run python -m src.convert_af_annotation_to_coco_instances --af_annotation_zip_or_dir out/af_annotation.zip \
 --coco_instances_json out/coco_instances.json \
 --clip_annotation_to_image

``` 

#### 備考
* `out/coco_instances.json`には、`categories`が記載されている必要があります。
* Annofabの「塗りつぶし」アノテーションは、Uncompressed RLEに変換されます。
* Annofabはマルチポリゴンに対応していないので、マルチポリゴンには変換されません。


#### Help

```
$ uv run python -m src.convert_af_annotation_to_coco_instances -h
usage: convert_af_annotation_to_coco_instances.py [-h] [--verbose] --af_annotation_zip_or_dir AF_ANNOTATION_ZIP_OR_DIR [--af_input_data_json AF_INPUT_DATA_JSON]
                                                  --coco_instances_json COCO_INSTANCES_JSON [-o OUTPUT_COCO_INSTANCES_JSON] [--clip_annotation_to_image]
                                                  [--af_task_id AF_TASK_ID [AF_TASK_ID ...]] [--af_input_data_id AF_INPUT_DATA_ID [AF_INPUT_DATA_ID ...]]
                                                  [--af_label_name AF_LABEL_NAME [AF_LABEL_NAME ...]] [--af_task_phase AF_TASK_PHASE] [--af_task_status AF_TASK_STATUS]

Annofab形式のアノテーションを、COCOデータセット（Instances）形式に変換します。Annofabのinput_data_nameはCOCOのimage.file_nameに、Annofabのラベル名(英語)はCOCOのcategory.nameに変換します。

options:
  -h, --help            Show this help message and exit.
  --verbose             詳細なログを出力します。 (default: False)
  --af_annotation_zip_or_dir AF_ANNOTATION_ZIP_OR_DIR
                        Annofab形式のアノテーションZIPファイルのパス。またはZIPファイルを展開したディレクトリのパス。`annofabcli annotation download`コマンドでアノテーションZIPファイルをダウンロードできます。 (required, type: <class 'Path'>)
  --af_input_data_json AF_INPUT_DATA_JSON
                        Annofabの入力データ全件ファイルのパス。COCO形式のimagesを生成するのに利用します。`annofabcli input_data download`コマンドでダウンロードできます。未指定の場合は、'--coco_instances_json'に指定したJSONファイルの'images'を利用します。 (type: <class
                        'Path'>, default: null)
  --coco_instances_json COCO_INSTANCES_JSON
                        入力情報であるCOCOデータセット（Instances）形式アノテーションのJSONファイルのパス。`categories`と`images`(オプショナル)を参照します。 (required, type: <class 'Path'>)
  -o, --output_coco_instances_json OUTPUT_COCO_INSTANCES_JSON
                        変換後のCOCOデータセット（Instances）形式アノテーションの出力先JSONファイルのパス (required, type: <class 'Path'>)
  --clip_annotation_to_image
                        指定すると、アノテーションが画像からはみ出さないようにクリッピングします。Annofabは矩形やポリゴンは画像外に作図できます。ただし、塗りつぶしアノテーションは画像外に作図できません。 (default: False)
  --af_task_id AF_TASK_ID [AF_TASK_ID ...]
                        変換対象のAnnofabのタスクのID (type: str, default: null)
  --af_input_data_id AF_INPUT_DATA_ID [AF_INPUT_DATA_ID ...]
                        変換対象のAnnofabの入力データのID (type: str, default: null)
  --af_label_name AF_LABEL_NAME [AF_LABEL_NAME ...]
                        変換対象のAnnofabのラベル名（英語） (type: str, default: null)
  --af_task_phase AF_TASK_PHASE
                        変換対象のAnnofabのタスクのフェーズ (type: str, default: null)
  --af_task_status AF_TASK_STATUS
                        変換対象のAnnofabのタスクのステータス (type: str, default: null)
```