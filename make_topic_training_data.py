from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass
import random
from glob import glob
from tqdm import tqdm
from permutation_task import PermutationTask
import os
import json
import argparse


@dataclass
class TopicModel:
    """A simple topic model implementation.
    
    This model assigns topic distributions to documents and word distributions to topics
    in a way that mimics Latent Dirichlet Allocation but with simplified assumptions.
    """
    num_topics: int
    vocabulary: List[str]
    topic_word_dist: np.ndarray = None
    alpha: float = 0.3  # Dirichlet prior for document-topic distribution
    beta: float = 0.3   # Dirichlet prior for topic-word distribution
    
    def __post_init__(self):
        # Generate topic-word distributions
        if self.topic_word_dist is None:
            self.topic_word_dist = np.random.dirichlet(
                [self.beta] * len(self.vocabulary), 
                size=self.num_topics
            )

    @classmethod
    def init_from_file(cls, file_path: str, vocabulary: List[str]):
        with open(file_path, 'r') as f:
            data = json.load(f)
            topic_word_dist = np.array(data["topic_mapping_0"])
            num_topics = len(topic_word_dist)
            return cls(num_topics=num_topics, vocabulary=vocabulary, topic_word_dist=topic_word_dist)
    
    @classmethod
    def init_from_json(cls, json_path: str, vocabulary: List[str]):
        topic_word_dist = np.array(json.load(open(json_path)))
        num_topics = len(topic_word_dist)
        return cls(num_topics=num_topics, vocabulary=vocabulary, topic_word_dist=topic_word_dist)

    def generate_document(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a document from the topic model.

        Returns:
            np.ndarray: An array of words sampled according to the topic model's distribution.
            Each word is selected by sampling a topic from the document-topic distribution and then
            sampling a word from that topic's word distribution.
            np.ndarray: The topic for each word in the document.
            np.ndarray: The document-topic distribution.
        """
        doc_topic_dist = np.random.dirichlet([self.alpha] * self.num_topics)

        # First sample topics for each word position
        topics = np.random.choice(
            np.arange(self.num_topics), 
            size=100, 
            p=doc_topic_dist
        )
        
        # Then sample words for each topic
        words = []
        for topic in topics:
            word_idx = np.random.choice(
                len(self.vocabulary),
                size=1,
                p=self.topic_word_dist[topic]
            )[0]
            words.append(self.vocabulary[word_idx])
        
        return np.array(words), topics, doc_topic_dist


def main(args):
    # Initialize the PermutationMDP with the specified number of items
    permutation_task = PermutationTask(num_items=args.num_items)
    
    all_splits = args.splits.split(",")
    # Use the action_to_nl dictionary values as vocabulary
    vocabulary = list(permutation_task.action_to_nl.values())
    
    for split in all_splits:
        os.makedirs(os.path.join(args.data_dir, split), exist_ok=True)

    if args.init_from_json:
        topic_model = TopicModel.init_from_json(args.init_from_json, vocabulary)
    else:
        topic_model = TopicModel(num_topics=args.num_topics, vocabulary=vocabulary)
    for i in tqdm(range(args.num_stories)):
        if i > args.num_stories * args.train_split_ratio:
            split = "test"
        else:
            split = "train"
        if os.path.exists(os.path.join(args.data_dir, split, f"story_{i}.json")):
            continue
        words, topics, doc_topic_dist = topic_model.generate_document()
        story = " ".join(words.tolist())
        with open(os.path.join(args.data_dir, split, f"story_{i}.json"), 'w') as f:
            json.dump({
                "story": story,
                "per_word_topic_labels_0": topics.tolist(),
                "topic_mapping_0": topic_model.topic_word_dist.tolist(),
                "topic_dist_0": doc_topic_dist.tolist()
            }, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="S3_NTP_data")
    parser.add_argument("--splits", type=str, default="train,test")
    parser.add_argument("--data", type=str, default="permutation")
    parser.add_argument("--num_topics", type=int, default=4)
    parser.add_argument("--train_split_ratio", type=float, default=0.999)
    parser.add_argument("--num_stories", type=int, default=1000000)
    parser.add_argument("--init_from_json", type=str, default=None)
    parser.add_argument("--num_items", type=int, default=3)
    args = parser.parse_args()
    main(args)
