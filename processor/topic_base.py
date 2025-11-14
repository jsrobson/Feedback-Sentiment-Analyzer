"""
Class defines abstract base class TopicBase, and establishes Subtopic
and Topic classes inheriting TopicBase.
"""
# == Standard Library imports ==
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TopicBase(ABC):
    """
    Dataclass for abstract base class TopicBase; establishes baseline rules
    for child classes Subtopic and Topic.
    """
    name: str                           # raw text name

    @abstractmethod
    def name_prompt(self) -> str:
        """
        Abstract method, inherited by Subtopic and Topic. Establishes a
        prompt to request name string generation by LLM.
        :return: Generated name string.
        """
        pass

@dataclass
class Subtopic(TopicBase):
    """
    Dataclass for Subtopic object, inherited basic rules from TopicBase.
    """
    id: int
    count: int
    tags: list[str]
    feedback: list[str] = field(default_factory=list)
    sentiment: dict[str, int] = field(default_factory=dict)
    read_name: Optional[str] = None
    summary: Optional[str] = None

    def get_data_dict(self) -> dict[str, str | int]:
        """
        Method returns data dictionary comprising Subtopic fields; used to
        build tabular data output.
        :return: Data dictionary, containing readable name, recurrent
        sentiment values, count of responses, and generated summary.
        """
        return {
            "Subtopic": self.read_name,
            "Sentiment": max(self.sentiment, key=self.sentiment.get),
            "Number of Responses": self.count,
            "Summary": self.summary
        }

    def get_str_data(self) -> str:
        """
        Method returns string comprising Subtopic fields, used to inform
        prompt input into LLM.
        :return: String, containing unique identifier, machine-generated
        name, keywords, feedback strings, and derived sentiment.
        """
        # build sentiment list from dict
        smt = [f"{label}: {count}" for label, count in self.sentiment.items()]
        # generate data as string for prompting
        str_data = f"""
        id: {self.id}\n\n,
        name: {self.name}\n\n,
        tags: {", ".join(self.tags)}\n\n,
        feedback: {", ".join(self.feedback)}\n\n,
        sentiment: {", ".join(smt) if self.sentiment else ""}
        """
        return str_data

    def name_prompt(self) -> str:
        """
        Method returns generative AI prompt for subtopic name generation.
        :return: Generative AI prompt, string.
        """
        # generate prompt for subtopic name development given subtopic str data
        prompt = f"""
        You are an analyst helping to label clusters of user feedback with 
        concise and descriptive names. We are labelling subtopics, which are 
        defined as a smaller, more specific topic that is part of a larger, 
        broader subject.

        The subtopic has the following short description:
        "{self.name}"

        The associated keywords for the subtopic are:
        {", ".join(self.tags)}
        
        The associated feedback for the subtopic is:
        {", ".join(self.feedback) if self.feedback else ""}

        Please generate a **brief, human-readable name** for this subtopic.
        - Keep it 2–5 words.
        - Make it clear and intuitive.
        - Avoid generic terms.
        - Use title case.

        Only return the name, without explanation.
        """
        return prompt.strip()

    def summary_prompt(self) -> str:
        """
        Method returns generative AI prompt for subtopic summary development.
        :return: Generative AI prompt, string.
        """
        # build sentiment list from dict
        smt = [f"{label}: {count}" for label, count in self.sentiment.items()]
        # generate prompt for subtopic summary development
        prompt_st = f"""
        You are analyzing user feedback data.

        The following keywords represent a cluster of related feedback:
        {", ".join(self.tags)}

        Here are example feedback statements from this cluster:
        {", ".join(self.feedback)}
        
        Sentiment distribution for this cluster:
        {", ".join(smt) if self.sentiment else ""}

        Write a short, cohesive paragraph summarizing the main topic or 
        theme of these keywords, feedback statements, and sentiment describe. 
        The summary should:
            - Be factual and objective
            - Do not start with "Here's a summary of the cluster" or similar
            - Capture the key issue or focus of discussion across feedback 
            statements
            - Avoid repetition or quoting text directly
            - Be roughly 3–5 sentences in length
            - Use plain and neutral language
        """
        return prompt_st.strip()

@dataclass
class Topic(TopicBase):
    """
    Dataclass for Topic object, inherited basic rules from TopicBase.
    """
    # list of ints corresponding to subtopic unique identifiers
    related_sub_topics: list[int] = field(default_factory=list)
    # list of subtopic data strings
    subtopic_data: list[str] = field(default_factory=list)
    # human-readable topic name
    read_name: Optional[str] = ""

    def name_prompt(self) -> str:
        """
        Method returns generative AI prompt for topic name generation.
        :return: Generative AI prompt, string.
        """
        # generate prompt for topic name development given subtopic str data
        subtopic_info_str = "\n\n".join(self.subtopic_data)
        prompt = f"""
        You are generating a clear, human-readable name for a topic based on 
        grouped subtopics. A topic is defined as the overarching theme or main 
        idea, relative to the subtopic which is a smaller, more specific topic.

        The current topic identifier is:
        "{self.name.replace("_", " ")}"

        Below are descriptions of the subtopics that belong to this topic.
        Each includes details like its id, name, tags, feedback, 
        and sentiment distribution:

        {subtopic_info_str}

        Generate a concise and intuitive name for the overall topic that uses
        the topic identifier and descriptions of the subtopic.
        
        Guidelines:
        - Use 2–5 words in Title Case.
        - Reflect the main unifying idea or focus across the subtopics.
        - Avoid jargon, underscores, or overly generic labels.
        - Do not include the word “Topic” or “Subtopic” in the name.
        - Return only the final name, nothing else.
        """

        return prompt.strip()


    def lookup_sub_topic(self, sub_topics: dict[int, Subtopic]) -> None:
        """
        Method to lookup and store subtopic data associated with the given
        topic using a corresponding unique identifier for subtopics.
        :param sub_topics: Dictionary, with unique identifier as key and
        corresponding subtopic object as value.
        """
        # create simple data structure for subtopic data
        data_st = []
        # filter subtopics into list by association with topic
        topic_subs = [sub_topics[st_id] for st_id in self.related_sub_topics
                      if st_id in sub_topics]
        # for each associated subtopic, get its string data and place into list
        for sub in topic_subs:
            data_st.append(sub.get_str_data())
        # set the topic's subtopic data to list output so it can be fed into
        # prompt generation
        self.subtopic_data = data_st