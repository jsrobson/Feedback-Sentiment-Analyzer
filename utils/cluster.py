# == Third party imports ==
from collections import defaultdict

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import hdbscan
import pandas as pd

ST_MODEL = "all-roberta-large-v1"

def get_sentence_transformer() -> SentenceTransformer:
    model = SentenceTransformer(ST_MODEL)
    return model

class Cluster:
    def __init__(self, sentences: list[str]):
        self.sentences = sentences
        self.st_model = SentenceTransformer(ST_MODEL)
        self.topic_model = self._build_clusters()
        self.hierarchy = self.topic_model.hierarchical_topics(self.sentences)

    def _build_clusters(self) -> BERTopic | None:
        print("Building clusters...")
        if not self.sentences:
            return None

        embeddings = self.st_model.encode(self.sentences, show_progress_bar=True)

        umap_model = UMAP(n_components=5, min_dist=0.0, metric='cosine',
                          random_state=42)
        reduced_embeddings = umap_model.fit_transform(embeddings)

        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=4, metric="euclidean")
        hdbscan_model.fit_predict(reduced_embeddings)

        vectorizer_model = CountVectorizer(
            # set low to reduce likelihood that keywords are removed
            max_df=0.4,
            stop_words="english"
        )
        representation_model = KeyBERTInspired()
        topic_model = BERTopic(embedding_model=self.st_model,
                               umap_model=umap_model,
                               hdbscan_model=hdbscan_model,
                               vectorizer_model=vectorizer_model,
                               representation_model=representation_model)

        topic_model.fit_transform(self.sentences, embeddings)
        return topic_model

    def package_model_data(self) -> dict[int, dict]:
        topic_model = self.topic_model
        topic_info = topic_model.get_topic_info()
        topic_name_lookup = dict(zip(topic_info["Topic"], topic_info["Name"]))
        data = {}

        for topic_id in topic_model.get_topic_info()['Topic']:
            # skip 'uncategorized' items
            if topic_id == -1:
                continue
            name = topic_name_lookup.get(topic_id)
            count = topic_model.get_topic_freq(topic_id)
            tags = [w for w, _ in topic_model.get_topic(topic_id)]
            feedback = topic_model.get_representative_docs(topic_id)
            sampled_feedback = feedback[:4]
            data[topic_id] = {
                "id": topic_id,
                "name": name,
                "count": count,
                "tags": tags,
                "feedback": sampled_feedback
            }
        return data

    def get_subtopic_id(self, ind):
        topics = self.topic_model.topics_
        # return pd.Series(topics, index=self.df.index)
        return pd.Series(topics, index=ind)

    def assign_topic(self, t_id: int):
        # Step 1: select rows where t_id appears in Topics
        condition = self.hierarchy['Topics'].apply(lambda x: t_id in x)
        matching_rows = self.hierarchy.loc[condition]

        if matching_rows.empty:
            return None  # no parent found

        # Step 2: find the minimum list length among matching rows
        min_len = matching_rows['Topics'].apply(len).min()

        # Step 3: get all rows with the minimum length
        smallest_rows = matching_rows[
            matching_rows['Topics'].apply(len) == min_len
            ]

        # Step 4: if multiple rows tie, pick one deterministically
        # (for example, by Parent_ID or hierarchical level)
        if len(smallest_rows) > 1:
            # You could use any tie-breaker here:
            # - smallest parent ID
            # - most specific name
            # - or even return a list of all parent names
            smallest_rows = smallest_rows.sort_values(by='Parent_ID',
                                                      ascending=True)

        parent_name = smallest_rows.iloc[0]['Parent_Name']
        return parent_name




