# VET EXAM LLM
獣医師国家試験問題の「学説試験問題（B）」をLLMを利用して問題を解いた。  
`gpt-4-1106-preview`を利用すると70%前後の正解率となる。  
このリポジトリでは[Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine](https://arxiv.org/abs/2311.16452)で紹介されていた`Med Prompt`という手法を用いて、さらにスコアを改善できるかを検討した。

※ 「実地試験問題（C）」や「実地試験問題（D）」の画像は公開されておらず、検討できてない。しかし、[Capability of GPT-4V(ision) in Japanese National Medical Licensing Examination](https://www.medrxiv.org/content/10.1101/2023.11.07.23298133v1.full)で紹介されているように、現時点では、GPT4-Vを利用しても、画像からの追加情報は良いスコアはもたらすことができないかもしれない。

### Data Source
[獣医師国家試験](https://www.maff.go.jp/j/syouan/tikusui/zyui/shiken/shiken.html)

### PDF Parse Tech
[Multi-Modal on PDF's with tables.](https://github.com/run-llama/llama_index/blob/main/docs/examples/multi_modal/multi_modal_pdf_tables.ipynb)

PDFをパースするために以下の方法を利用した。
- Table TransformerでPDFのテーブルをイメージで抽出する
- 抽出したイメージをGPT4-Vに渡して、指定したフォーマットにする

### Med Prompt
Med Promptは、前処理フェーズとテストケースで最終的な予測を生成する推論ステップの2段階で構成されている。

前処理:
- トレーニングデータセット内の各質問は、軽量な埋め込みモデルを通じて埋め込みベクトルを生成するために渡される(by text-embedding-ada-002)。
- 各質問に対して、GPT-4は思考の連鎖と最終回答の予測を生成するために活用する。生成された回答が正しく、基準ラベルと一致する場合、関連する質問、その埋め込みベクトル、思考の連鎖、および回答を保存する。
- そうでない場合は、最終的な回答が間違っている場合、その推論を信頼することはできないという前提で、質問を完全に破棄する。

推論時:
- テスト質問が与えられ、前処理中に使用した同じ埋め込みモデルでテストサンプルを再埋め込みし、前処理プールから類似の例をkNNを使用して取得する。
- これらの例とそれに対応するGPT-4によって生成された推論チェーンは、GPT-4のコンテキストとして構造化される。
- その後、テスト質問と対応する回答選択肢が最後に追加され、これが最終的なプロンプトとして機能する。モデルは、少数ショットの例に従って、思考の連鎖と候補回答を出力する。
- 最後に、上記のステップを複数回繰り返すことでアンサンブル処理を実行する。テスト質問の回答選択肢の順序をシャッフルすることで多様性を高める。最終的な予測回答を決定するために、最も頻繁な回答を選択する。
