# VET EXAM LLM

### MedPrompt
[Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine](https://arxiv.org/abs/2311.16452)

前処理:
- トレーニングデータセット内の各質問は、軽量な埋め込みモデルを通じて埋め込みベクトルを生成するために渡されます（アルゴリズム1の行4）。我々は、OpenAIのtext-embedding-ada-002を使用して埋め込みを作成しました。
- 各質問に対して、GPT-4は思考の連鎖と最終回答の予測を生成するために活用されます（行5）。生成された回答が正しく、基準ラベルと一致する場合、関連する質問、その埋め込みベクトル、思考の連鎖、および回答を保存します。
- そうでない場合は、最終的な回答が間違っている場合、その推論を信頼することはできないという前提で、質問を完全に破棄します（行6-7）。

推論時:
- テスト質問が与えられ、前処理中に使用した同じ埋め込みモデルでテストサンプルを再埋め込みし、前処理プールから類似の例をkNNを使用して取得します（行12-13）。
- これらの例とそれに対応するGPT-4によって生成された推論チェーンは、GPT-4のコンテキストとして構造化されます（行14）。
- その後、テスト質問と対応する回答選択肢が最後に追加され、これが最終的なプロンプトとして機能します（行17）。モデルは、少数ショットの例に従って、思考の連鎖と候補回答を出力します。
- 最後に、上記のステップを複数回繰り返すことでアンサンブル処理を実行します。テスト質問の回答選択肢の順序をシャッフルすることで多様性を高めます（行15-16）、セクション4.3および図4で詳細に説明されています。最終的な予測回答を決定するために、最も頻繁な回答を選択します（行20）。


COT Prompt

```
## Question: {{question}}
{{answer_choices}}
## Answer
model generated chain of thought explanation
Therefore, the answer is [final model answer (e.g. A,B,C,D)]
```

### REF
[Multi-Modal on PDF's with tables.](https://github.com/run-llama/llama_index/blob/main/docs/examples/multi_modal/multi_modal_pdf_tables.ipynb)
