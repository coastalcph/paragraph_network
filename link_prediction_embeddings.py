import pickle
import json
from datetime import datetime as dt
from collections import defaultdict
import numpy as np
import pandas as pd
import click
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from scipy.sparse import vstack


class Paragraph:
    def __init__(self, case_id, number, date, text, citations):
        self.case_id = case_id
        self.number = number
        self.date = dt.strptime(date, "%Y-%m-%d")
        self.text = text
        self.citations = citations

    def get_id(self):
        return f"{self.case_id}-{self.number}"


def retrieve_candidate_paragraphs(paragraphs, date):
    # Given a paragraph's date of publication,
    # all the previous paragraphs can be considered as citation candidates
    candidates = set()
    for p in paragraphs:
        if p.date < date:
            candidates.add(p)
    return candidates


def generate_train_test(metadata):
    # Training set will be paragraph from < 2018, test set >= 2018
    train = list()
    test = list()
    for case_id, m in metadata.items():
        m['case_id'] = case_id
        date = m['meta']['date']
        year = date.split('-')[0]
        if int(year) < 2018:
            train.append(m)
        else:
            test.append(m)
    return train, test


def retrieve_candidates_ids(metadatas, date):
    # Given a paragraph's date of publication,
    # all the previous paragraphs can be considered as citation candidates
    candidates = set()
    for case_id, metadata in metadatas.items():
        p_date = dt.strptime(metadata['meta']['date'], "%Y-%m-%d")
        if p_date < date:
            candidates.add(case_id)
    return candidates


def tf_idf(tf_idf, text1, text2):
    vector1 = tf_idf.transform([text1])
    vector2 = tf_idf.transform([text2])
    similarity = cosine_similarity(vector1, vector2)
    return similarity[0]

def concat_sparse_matrix(vectors_by_par, paragraphs):
    vectors = [vectors_by_par[p].reshape(1, -1) for p in paragraphs]
    matrix = vstack(vectors)
    return matrix

def concat_par_vectors(vectors_by_par, paragraphs):
    vectors = [vectors_by_par[p].reshape(1, -1) for p in paragraphs]
    matrix = np.vstack(vectors)
    return matrix


def compute_precision(concat_func, all_paragraphs_obj, vectors_by_par, paragraph, k=10, verbose=False):
    candidates = retrieve_candidate_paragraphs(all_paragraphs_obj, paragraph.date)
    candidates_texts = list({p.text for p in candidates if p.text in vectors_by_par})
    citations_to_find = {c for c in paragraph.citations if c in candidates_texts}
    num_citations = len(citations_to_find)
    if num_citations:
        candidates_vectors = concat_func(vectors_by_par, candidates_texts)
        source_vector = vectors_by_par[paragraph.text]
        sims = cosine_similarity(candidates_vectors, source_vector.reshape(1, -1)).reshape(-1)
        indices = np.argsort(sims)[::-1]
        
        results = defaultdict(lambda: defaultdict(dict))
        num_good = 0
        precisions = list()
        ranks = list()
        for i, candidate_index in enumerate(indices):
            candidate_sim = sims[candidate_index]
            candidate_text = candidates_texts[candidate_index]
            if verbose and i < 11:
                ranks.append((candidate_text, float(candidate_sim)))
            if candidate_text in citations_to_find:
                num_good += 1
                precision = num_good / (i+1)
                precisions.append(precision)
                citations_to_find.remove(candidate_text)
                results[paragraph.text][candidate_text]['rank'] = i
                results[paragraph.text][candidate_text]['precision'] = precision
                results[paragraph.text][candidate_text]['similarity'] = float(candidate_sim)
        results['average_precision'] = sum(precisions) / num_citations
        if verbose:
            results['ranks'] = ranks
        return dict(results)
    else:
        print("Hey!")
        return None
    

def find_paragraph_by_text(paragraphs, text):
    for p in paragraphs:
        if p.text == text:
            return p


def load_vectors(vectors_file):
    vectors = pickle.load(open(vectors_file, "rb"))
    filtered = dict()
    for k, v in vectors.items():
        if np.isnan(v).sum() > 0:
            pass
        else:
            filtered[k] = v
    return filtered


@click.command()
@click.argument("paragraph_file")
@click.argument("metadata_file")
@click.argument("vectors_file")
def main(paragraph_file, metadata_file, vectors_file):
    print("Loading data...")
    paragraphs_df = pd.read_excel(paragraph_file)
    print("Rows", len(paragraphs_df))
    paragraphs_df = paragraphs_df.dropna()
    print("Rows", len(paragraphs_df))
    metadata = json.load(open(metadata_file))
    train, test = generate_train_test(metadata)
    train_celex = {m['case_id'] for m in train}
    test_celex = {m['case_id'] for m in test}
    train_paragraphs_from = list(set(paragraphs_df[paragraphs_df['CELEX_FROM'].isin(train_celex)]["TEXT_FROM"].tolist()))
    train_paragraphs_to = list(set(paragraphs_df[paragraphs_df['CELEX_TO'].isin(train_celex)]["TEXT_TO"].tolist()))
    train_paragraphs = [p for p in train_paragraphs_from + train_paragraphs_to if type(p) is str]

    print("Building objects...")
    train_paragraphs_obj = list()
    test_paragraphs_obj = list()
    texts = set()
    grp_by_celex_df = paragraphs_df.groupby(["CELEX_FROM", "NUMBER_FROM"])
    for (celex_from, number_from), subset_df in tqdm(grp_by_celex_df):
        paragraph = subset_df['TEXT_FROM'].tolist()[0]
        date = subset_df['DATE_FROM'].tolist()[0]
        citations = subset_df['TEXT_TO'].tolist()
        obj = Paragraph(celex_from, number_from, date, paragraph, citations)
        if celex_from in train_celex:
            train_paragraphs_obj.append(obj)
            texts.add(paragraph)
        elif celex_from in test_celex:
            test_paragraphs_obj.append(obj)
            texts.add(paragraph)
        else:
            print("oups")

    paragraphs_to_obj = list()
    for _, row in tqdm(paragraphs_df.iterrows()):
        paragraph = row['TEXT_TO']
        if paragraph not in texts:
            celex_to = row['CELEX_TO']
            number_to = row['NUMBER_TO']
            date = row['DATE_TO']
            citations = None
            obj = Paragraph(celex_to, number_to, date, paragraph, citations)
            paragraphs_to_obj.append(obj)
            texts.add(paragraph)

    print(len(train_paragraphs_obj), len(test_paragraphs_obj), len(paragraphs_to_obj))

    p_from = set(paragraphs_df['TEXT_FROM'].tolist())
    p_to = set(paragraphs_df['TEXT_TO'].tolist())
    all_paragraphs = list(p_from.union(p_to))
    all_paragraphs = [p for p in all_paragraphs if type(p) is str]
    print(len(p_from), len(p_to), len(all_paragraphs))

    if vectors_file == 'tfidf':  # train and compute vectors on the fly
        print("Fitting tf-idf...")
        vectorizer = TfidfVectorizer(
            stop_words='english',
            strip_accents='ascii'
        )
        X = vectorizer.fit_transform(train_paragraphs)
        vectors = vectorizer.transform(all_paragraphs)
        vectors_by_par = dict()
        for p, v in zip(all_paragraphs, vectors):
            vectors_by_par[p] = v
        concat_func = concat_sparse_matrix
    else:
        vectors_by_par = load_vectors(vectors_file)
        concat_func = concat_par_vectors

    test_pars_with_citations = [p for p in test_paragraphs_obj if len(p.citations)]
    all_paragraphs_obj = train_paragraphs_obj + test_paragraphs_obj + paragraphs_to_obj

    print("Computing precisions single thread...")
    results = list()
    logs = list()
    pbar = tqdm(test_pars_with_citations)
    for i, p in enumerate(pbar):
        r = compute_precision(concat_func, all_paragraphs_obj, vectors_by_par, p, verbose=True)
        if r is not None:
            results.append(r['average_precision'])
            logs.append(r)
        if i > 0 and i % 10 == 0:
            pbar.set_description(f"MAP: {np.mean(results)}")
    mean_average_precision = np.mean(results)
    print(mean_average_precision)
    json.dump(results, open(f"./data/processed/link_prediction.json", "w"), indent=True)
    json.dump(logs, open(f"./data/processed/link_logs.json", "w"), indent=True)

if __name__ == '__main__':
    main()
