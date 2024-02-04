from itertools import product, chain
from typing import List
import json
from datasets import load_dataset
import spacy
import random
from random import shuffle
import os
import pickle
from copy import deepcopy
# from sentence_transformers import SentenceTransformer

from germandpr import remove_titles_and_headings_from_contexts


class DynamicRAGDataset:
    """
    A RAG dataset with varying and adjustable number of (positive and negative) contexts and length of contexts.
    Finetuning your model on this allows you, to 'meet your RAG retriever in the middle'.
    Adjust the parameters to match the settings of your RAG chunker and retriever.
    All parameter combinations will be equally distributed in the dataset.
    """

    def __init__(
        self,
        germandpr_path="deepset/germandpr",
        instructions_path="instructions.jsonl", # load your Airoboros generations here
        max_num_positive_ctxs=1,
        max_num_negative_ctxs=3,
        max_context_length=-1,
    ):
        self.germandpr = load_dataset(germandpr_path)["train"]
        self.instructions = self.load_instructions(instructions_path)
        self.inst_to_dpr = self.process_and_map_instructions_to_dpr()
        self.param_combis = self.create_parameter_combinations(
            max_num_positive_ctxs, max_num_negative_ctxs, max_context_length
        )

        # all combinations without a relevant context
        self.no_answer_combis = [param for param in self.param_combis if param[0] == 0]

        # all combinations with at least one relevant context
        self.param_combis = [param for param in self.param_combis if param[0] != 0]

        self.nlp = spacy.load("de_core_news_lg")
        # self.embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")

    def load_instructions(self, file_path):
        data = []
        with open(file_path, "r") as file:
            for line in file:
                item = json.loads(line)
                instruction = item.get("instruction", "")
                context, question = instruction.split("\nQUESTION: ")
                sentence_context = context.replace("CONTEXT: ", "")
                entry = {
                    "question": question.strip(),
                    "sentence_context": sentence_context.strip(),
                    "answer": item["response"].strip(),
                }
                data.append(entry)
        return data

    def process_and_map_instructions_to_dpr(self):
        """Creates a dictionary for mapping airoboros data to GermanDPR once"""

        # TODO: use index/ID instead
        instruction_to_dpr = {}
        for i, item in enumerate(self.germandpr):
            print(f"Processing GermanDPR item {i} of {len(self.germandpr)}")
            positive_ctxs = item["positive_ctxs"]["text"][0]
            question_dpr = item["question"]
            for item_inst in self.instructions:
                sentence_context = item_inst["sentence_context"]
                question = item_inst["question"]
                if sentence_context in positive_ctxs and question in question_dpr:
                    key = (sentence_context, question)
                    # item = remove_titles_and_headings_from_contexts(item)
                    instruction_to_dpr[key] = item
                    # instruction_to_dpr[key] = i

        return instruction_to_dpr

    def create_parameter_combinations(
        self, max_num_positive_ctxs, max_num_negative_ctxs, max_context_length
    ):
        """Creates and filters a cartesian product of all sensible parameter combinations."""

        # keep -1 as argument for readability/understandability but set to 0 because of some weird filter behavior below
        max_context_length = (
            0 if max_context_length == -1 else max_context_length
        )

        # Create ranges for each parameter
        positive_ctxs_nums = range(max_num_positive_ctxs + 1)
        negative_ctxs_nums = range(max_num_negative_ctxs + 1)
        positive_ctxs_lengths = range(
            -1, max_context_length + 1
        )  # -1 is for the full context
        negative_ctxs_lengths = range(-1, max_context_length + 1)

        # cartesian product of all parameters
        param_combis = list(
            product(
                positive_ctxs_nums,
                negative_ctxs_nums,
                positive_ctxs_lengths,
                negative_ctxs_lengths,
            )
        )

        # remove all (0, 0, x, y) combinations
        param_combis = [param for param in param_combis if param[0] + param[1] > 0]

        # remove combinations where the context length is 0 but the number of contexts is greater than 0
        param_combis = [
            param for param in param_combis if not (param[0] > 0 and param[2] == 0)
        ]
        param_combis = [
            param for param in param_combis if not (param[1] > 0 and param[3] == 0)
        ]

        # if one of the contexts is absent, filter out all context lenghts except 0
        param_combis = [
            param for param in param_combis if not (param[0] == 0 and param[2] != 0)
        ]
        param_combis = [
            param for param in param_combis if not (param[1] == 0 and param[3] != 0)
        ]

        # return only as many full contexts as GermanDPR actually provides, one positive and three negatives
        param_combis = [
            param for param in param_combis if not (param[0] > 1 and param[2] == -1)
        ]
        param_combis = [
            param for param in param_combis if not (param[1] > 3 and param[3] == -1)
        ]

        # sometimes a positive context of one sentence contains not enough information
        # param_combis = [
        #     param for param in param_combis if not (param[0] == 1 and param[2] == 1)
        # ]

        return param_combis

    def chunk(self, contexts: List[str], chunk_size: int) -> List[str]:
        """
        Takes in a list of contexts and returns a list of chunks of the contexts with the given chunk size.
        You can base this on tokens or your RAG chunker instead of sentences.
        """

        chunks = []
        for context in contexts:
            sentences = list(self.nlp(context).sents)

            if chunk_size == -1:
                chunks.extend([" ".join(sentence.text for sentence in sentences)])
            else:
                for i in range(0, len(sentences), chunk_size):
                    chunk = " ".join(
                        sentence.text for sentence in sentences[i : i + chunk_size]
                    )
                    chunks.append(chunk)
        return chunks

    def compile_contexts(self, item, param_tuple):
        """Chop and mix the one positive and two negative contexts of GermanDPR based on the parameters."""

        positive_ctxs = item["positive_ctxs"]["text"]  # is a list of one element
        negative_ctxs = item["hard_negative_ctxs"][
            "text"
        ]  # three negatives in GermanDPR
        answer_span = item["answers"][0]

        # In the final dataset all context lenghts were set to -1, i.e. only full contexts were used,
        # because handling edge cases of varying context lenghts/sentence counts got too anoying.
        (
            num_positive_ctxs,
            num_negative_ctxs,
            positive_ctx_length,
            negative_ctx_length,
        ) = param_tuple

        positive_ctxs_out = []
        if num_positive_ctxs > 0:
            if positive_ctx_length == -1 and num_positive_ctxs == 1:
                positive_ctxs_out.append(positive_ctxs[0])

            else:
                positive_chunks = self.chunk(positive_ctxs, positive_ctx_length)
                # Find the chunk that contains the answer and add it first
                for i, chunk in enumerate(positive_chunks):
                    if answer_span in chunk:
                        positive_ctxs_out.append(chunk)
                        positive_chunks.pop(i)
                        break

                if num_positive_ctxs > 1:
                    shuffle(positive_chunks)  # avoid always adding titles/beginnings first
                    # Add the remaining chunks
                    for chunk in positive_chunks[: num_positive_ctxs - 1]:
                        positive_ctxs_out.append(chunk)

        negative_ctxs_out = []
        if num_negative_ctxs > 0:
            if negative_ctx_length == -1 and num_negative_ctxs in [1, 2, 3]:
                # full contexts
                if num_negative_ctxs == 1:
                    negative_ctxs_out.append(negative_ctxs[0])
                elif num_negative_ctxs == 2:
                    negative_ctxs_out.extend(
                        negative_ctxs[:2]
                    )  # first two negatives from GermanDPR
                else:
                    negative_ctxs_out.extend(
                        negative_ctxs
                    )  # all three full negatives from GermanDPR
            else:  # chunking
                negative_chunks = self.chunk(negative_ctxs, negative_ctx_length)
                shuffle(negative_chunks)  # avoid always adding titles/beginnings first
                for chunk in negative_chunks[:num_negative_ctxs]:
                    negative_ctxs_out.append(chunk)

        # TODO: handle cases where there are less sentences in the context than the number of requested chunks
        # maybe sort and zip dataset and parameter combinations by number of sentences/chunks

        ### DEBUGGING
        # # check requested and actual number of chunks
        # failed = (len(positive_ctxs_out) != num_positive_ctxs) or (
        #     len(negative_ctxs_out) != num_negative_ctxs
        # )
        # # if positive context is present, check if answer is in it
        # failed = (
        #     failed or (answer_span not in positive_ctxs_out[0])
        #     if positive_ctxs_out
        #     else failed
        # )
        # if failed:
        #     print(f"{param_tuple=}")
        #     print(f"{len(positive_ctxs_out)=}")
        #     print(f"{num_positive_ctxs=}")
        #     print(f"{len(negative_ctxs_out)=}")
        #     print(f"{num_negative_ctxs=}")
        #     # write offending item to file
        #     with open("failed_items.jsonl", "a") as file:
        #         json.dump(item, file)
        #         file.write("\n")
        #     return None
        # assert len(positive_ctxs_out) == num_positive_ctxs
        # assert len(negative_ctxs_out) == num_negative_ctxs
        # assert answer_span in positive_ctxs_out[0]

        return positive_ctxs_out, negative_ctxs_out

    # TODO: pair with dissimilar contexts specifically
    # you can use built-in FAISS in huggingface `datasets`
    # def embedd(self, text):
    #     return self.embedding_model.encode(text)

    def randomize_instructions(self):
        """Pair contexts with question by sampling without replacement."""
        #TODO: only return questions?

        instructions = deepcopy(self.instructions)
        questions = [item["question"] for item in instructions]

        randomized_instructions = []
        for item in instructions:
            original_question = item["question"]
            sampled_question = original_question
            while sampled_question == original_question:
                sampled_question = random.choice(questions)
            item["question"] = sampled_question
            randomized_instructions.append(item)
            questions.remove(sampled_question)
        return randomized_instructions

if __name__ == "__main__":
    dat = DynamicRAGDataset()

    output_path = "germanrag.jsonl"
    with open(output_path, "w") as f:
        pass  # clear file

    # extend parameter combinations beyond the number of instructions
    factor = len(dat.instructions) // len(dat.param_combis) + 1
    no_answer_factor = (len(dat.instructions) // 2) // len(dat.no_answer_combis) + 1

    dat.param_combis = factor * dat.param_combis
    dat.no_answer_combis = no_answer_factor * dat.no_answer_combis

    first_pass = zip(dat.instructions, dat.param_combis)

    # only irrelevant contexts for one further half of the instructions
    randomized_instructions = dat.randomize_instructions()
    no_answer_pass = zip(
        dat.instructions[: len(dat.instructions) // 2], dat.no_answer_combis
        # randomized_instructions[: len(dat.instructions) // 2], dat.no_answer_combis
    )

    full_pass = chain(first_pass, no_answer_pass)

    for i, (item, param_tuple) in enumerate(full_pass):
        print(
            f"Processing Dynamic RAG item {i} of {len(dat.instructions) + len(dat.instructions)//2}"
        )
        print(f"{param_tuple=}")

        sentence_context, question = item["sentence_context"], item["question"]
        item_dpr = dat.inst_to_dpr[(sentence_context, question)]
        # idx_dpr = dat.inst_to_dpr[(sentence_context, question)]
        # item_dpr = dat.germandpr[idx_dpr]

        contexts = dat.compile_contexts(item_dpr, param_tuple)

        if contexts is not None:
            positive_ctxs_out, negative_ctxs_out = contexts
            contexts = positive_ctxs_out + negative_ctxs_out
            shuffle(contexts)  # so context containing answer is not always first
            positive_ctx_idx = contexts.index(positive_ctxs_out[0]) if positive_ctxs_out else -1
            print(contexts)

            if i > len(dat.instructions) - 1: # no answer pass
                assert not positive_ctxs_out and positive_ctx_idx == -1
                answer = "Mit den gegebenen Informationen ist diese Frage nicht zu beantworten."
                question = randomized_instructions[i - len(dat.instructions)]["question"]
            else:
                answer = item["answer"]

            entry = {
                "contexts": contexts,
                "question": question,
                "answer": answer,
                "positive_ctx_idx": positive_ctx_idx,
            }

            with open(output_path, "a") as file:
                json.dump(entry, file)
                file.write("\n")
