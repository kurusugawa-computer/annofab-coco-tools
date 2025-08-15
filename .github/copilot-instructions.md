# copilot-instructions.md

## 開発でよく使うコマンド
* コードのフォーマット: `make format`
* Lintの実行: `make lint`
* テストの実行: `make test`


## 技術スタック
* Python 3.12 以上
* テストフレームワーク: Pytest v8 以上

## ディレクトリ構造概要

* `src/**`: アプリケーションのソースコード
* `tests/**`: テストコード
* `tests/resources/**`: テストコードが参照するリソース
* `resources/**`: アプリケーションが参照するリソースや設定ファイル


## コーディングスタイル

### Python
* できるだけ型ヒントを付ける
* docstringはGoogleスタイル


## レビュー
* PRレビューの際は、日本語でレビューを行う