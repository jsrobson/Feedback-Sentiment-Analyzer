import re
from collections import defaultdict

# == Third party imports ==
import pandas as pd

# == Local ==
from .topic_base import Subtopic, Topic
from utils import Cluster, Sentiment, Summary


pd.set_option('display.max_columns', None)

FB_COL = "Comments"
SMT_LABEL = "smt_label"
SMT_SCORE = "smt_score"
T_ID = "topic_name"
ST_ID = "subtopic_id"


def get_csv(df: pd.DataFrame):
    return df.to_csv()


class Parser:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.smt = Sentiment()
        self.summary = Summary()
        self.cluster = Cluster(self.df[FB_COL].tolist())

        self.topics: list[Topic] = []
        self.subtopics: dict[int, Subtopic] = {}

    def _get_sentimental(self, feedback: list[str]):
        sentiment_counts = defaultdict(int)
        for fb in feedback:
            sentiment = self.smt.get_feedback_sentiment(fb)["label"]
            sentiment_counts[sentiment] += 1
        return dict(sentiment_counts)

    def _build_subtopics(self):
        for tid, dt in self.cluster.package_model_data().items():
            sentiment_count = self._get_sentimental(dt['feedback'])
            cleaned = re.sub(r'^[-\d_]+', '', dt['name'])
            st = Subtopic(
                name=cleaned,
                id=dt['id'],
                count=dt['count'],
                tags=dt.get('tags', []),
                feedback=dt['feedback'],
                sentiment=sentiment_count
            )
            self.subtopics[tid] = st

    def _build_topics(self):
        t_dict = defaultdict(list)
        for st_id in self.subtopics.keys():
            name = self.cluster.assign_topic(st_id)
            t_dict[name].append(st_id)
        for name, collection in t_dict.items():
            t = Topic(
                name=name,
                related_sub_topics=collection
            )
            self.topics.append(t)

    def _build_topic_names(self):
        print("Building topic names...")
        for t in self.topics:
            if t.read_name:
                continue
            t.lookup_sub_topic(self.subtopics)
            read_name = self.summary.get_output(t.name, t.name_prompt())
            t.read_name = read_name


    def _build_subtopic_info(self):
        print("Building subtopic information...")
        for st in self.subtopics.values():
            read_name = self.summary.get_output(st.name, st.name_prompt())
            summary_txt = self.summary.get_output(st.name, st.summary_prompt())
            st.read_name = read_name
            st.summary = summary_txt

    def get_summary(self):
        records = []
        d_map = {t.read_name: t.related_sub_topics for t in self.topics}
        for st in self.subtopics.values():
            st_data = st.get_data_dict()
            parent_topic = next(
                (t_name for t_name, sub_ids in d_map.items() if st.id in
                 sub_ids),
                "None"
            )
            st_data["General Topic"] = parent_topic
            st_data = {'General Topic': st_data.pop('General Topic'), **st_data}
            records.append(st_data)
        return pd.DataFrame(records)

    def run(self):
        self._build_subtopics()
        self._build_topics()
        self._build_topic_names()
        self._build_subtopic_info()
        df = self.get_summary().head(25)
        df.to_csv(
            "data/output.csv",
            index=False,
            encoding="utf-8",  # good default for text data
            sep=",",  # can change to '\t' for TSV
            quoting=1,  # 1 == csv.QUOTE_NONNUMERIC
        )

