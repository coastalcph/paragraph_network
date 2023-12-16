# Re-framing Case Law Citation Prediction from a Paragraph Perspective
This is the code and data repository for the paper "Re-framing Case Law Citation Prediction from a Paragraph Perspective" published at [JURIX 2023](https://jurix23.maastrichtlawtech.eu/program/).
It contains;
1. The [paragraph-to-paragraph dataset](https://huggingface.co/datasets/ngarneau/paragraph_to_paragraph).
2. The [metadata of each case](meta_all_20230207.json.gz).
3. The [training files](https://huggingface.co/datasets/ngarneau/link_prediction) for link prediction.
4. The [training files](https://huggingface.co/datasets/ngarneau/link_prediction_contrastive) for the link prediction (contrastive training).
5. The [vectors](https://huggingface.co/datasets/ngarneau/par_to_par_vectors) obtained from training on these files.
6. The [script](link_prediction_embeddings.py) to compute the mean average precision using the dataset, metadata, and vectors file.
