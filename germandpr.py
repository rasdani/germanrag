### python -m spacy download de_core_news_lg
from datasets import load_dataset
import spacy
from collections import defaultdict
import json
import tiktoken
from typing import List, Dict

nlp = spacy.load("de_core_news_lg")


def collapse_germandpr(germandpr):
    """Among multiple identical positive contexts, keep only the one with the longest/shortest answer."""

    def get_item_based_on_answer_length(items: List[Dict]) -> Dict:
        # this was set to longest and was buggy during generation, the final dataset probably just picked the first occurence
        # TODO: make shortest, longest, first/random an option
        # Rationale: 
        # shortest for succinct factoids which don't span multiple sentences --> less halluzinations, lower generation cost
        # longest for more material/hand holding for augmenting answer spans with LLM

        # Is a shorter answer span more likely to be answerable with a single/two sentence context?
        # e.g. short facts like years 1982, and single names
        ret = max(items, key=lambda item: len(item["answers"][0]))
        # ret = min(items, key=lambda item: len(item["answers"][0]))
        return ret

    positive_ctxs_dict = defaultdict(list)
    for item in germandpr:
        positive_ctxs_dict[item["positive_ctxs"]["text"][0]].append(item)

    germandpr_subset = [
        get_item_based_on_answer_length(items) for items in positive_ctxs_dict.values()
    ]
    return germandpr_subset


def pair_with_answer_starts(germandpr):
    """GermanDPR is missing the answer_start indices from GermanQuAD."""
    germanquad = load_dataset("deepset/germanquad")["train"]
    # match on question, context and answer to be sure
    # TODO: use indices/IDs instead, e.g. from the original datasets
    quad_dict = {
        (item["question"], item["context"], item["answers"]["text"][0]): item[
            "answers"
        ]["answer_start"][0]
        for item in germanquad
    }

    for item in germandpr:
        key = (item["question"], item["positive_ctxs"]["text"][0], item["answers"][0])
        if key in quad_dict:
            item["answer_start"] = quad_dict[key]

    return germandpr


# TODO: there are cases, where the answer span occurs multiple times in the context, handle these like below
# def get_three_sentence_context(context, answer_start):
#     doc = nlp(context)
#     sentences = list(doc.sents)

#     for i, sentence in enumerate(sentences):
#         # breakpoint()
#         sentence_start = sentence.start_char
#         sentence_end = sentence.end_char

#         if sentence_start <= answer_start < sentence_end:
#             # Start index of the previous sentence or the current sentence if no previous
#             prev_sentence_start = sentences[i - 1].start_char if i > 0 else sentence_start
#             # End index of the next sentence or the current sentence if no next
#             next_sentence_end = sentences[i + 1].end_char if i < len(sentences) - 1 else sentence_end

#             assert prev_sentence_start <= answer_start < next_sentence_end
#             return context[prev_sentence_start:next_sentence_end]


def get_single_sentence_context(context, answer_start, answer_text):
    doc = nlp(context)
    sentences = list(doc.sents)

    # Get all start indices of answer_text in context
    answer_starts = [
        i for i in range(len(context)) if context.startswith(answer_text, i)
    ]
    assert answer_start in answer_starts

    for sentence in sentences:
        sentence_start = sentence.start_char
        sentence_end = sentence.end_char

        if sentence_start <= answer_start < sentence_end:
            return context[sentence_start:sentence_end]


### these were only used while experimenting in the Jupyter notebook
# def is_first_element_single_word(lst):
#     first = lst[0].strip()
#     if lst and isinstance(first, str):
#         return len(first.split()) == 1
#     return False


# def nlp_is_first_element_single_word(lst):
#     first = lst[0].strip()
#     if lst and isinstance(first, str):
#         doc = nlp(first)
#         return len(doc) == 1
#     return False


# def remove_headings(split_text):
#     if split_text[0] == "":
#         split_text.pop(0)
#     if split_text[0].count("=") >= 2:
#         split_text.pop(0)
#     return split_text


# def remove_titles_and_headings_from_contexts(item):
#     """
#     Removes everything from e.g.
#     ['Glühlampe\n\n==== Nichtelektrische Lichtquellen ====\nNichtelektrische Lichtquellen sind...]
#     such that it starts with a proper sentence.
#     Uses the individual strenghts of conventional string splitting and spacy's tokenization.
#     """
#     # inpsected these special cases, it's ok to pop one more of their elements after processing
#     bad_guys_answers = [
#         "Knicklichter",
#         "Goodluck Jonathan",
#         "Beim Danner-Verfahren fließt eine Glasschmelze als Band auf einen schräg nach unten geneigten, rotierenden keramischen Hohlzylinder ",
#     ]
#     positive_ctxs = item["positive_ctxs"]["text"]
#     negative_ctxs = item["hard_negative_ctxs"]["text"]
#     contexts = positive_ctxs + negative_ctxs
#     for ctx in contexts:
#         ctx_split = ctx.split("\n")
#         if not is_first_element_single_word(ctx_split):
#             print(ctx_split)
#             raise ValueError("First element is not a single word")
#         ctx_split.pop(0)  # remove title
#         ctx_split = remove_headings(ctx_split)
#         if nlp_is_first_element_single_word(ctx_split):
#             print(ctx)
#             print(ctx_split)
#             print(item["question"])
#             print(item["positive_ctxs"]["text"])
#             print(item["answers"])
#             if item["answers"][0] in bad_guys_answers:
#                 continue
#             raise ValueError("First sentence is a single word")


# def process_context(ctx, answer_text):
#     """Process a single context: remove title and headings, and check if the first sentence is a single word.
#     Uses the individual strengths of conventional string splitting and spacy's tokenization.
#     """
#     # Special cases, it's ok to pop one more of their elements after processing
#     bad_guys_answers = [
#         "Knicklichter",
#         "Goodluck Jonathan",
#         "Beim Danner-Verfahren fließt eine Glasschmelze als Band auf einen schräg nach unten geneigten, rotierenden keramischen Hohlzylinder ",
#     ]
#     ctx_split = ctx.split("\n")

#     if not is_first_element_single_word(ctx_split):
#         print(ctx_split)
#         raise ValueError("First element is not a single word")

#     # Remove the title
#     ctx_split.pop(0)

#     # Remove any headings
#     ctx_split = remove_headings(ctx_split)

#     if nlp_is_first_element_single_word(ctx_split):
#         if answer_text in bad_guys_answers:
#             ctx_split.pop(0)

#         else:
#             raise ValueError("First sentence is a single word")

#     return "\n".join(ctx_split)


# def remove_titles_and_headings_from_contexts(item):
#     """
#     Removes everything from e.g.
#     ['Glühlampe\n\n==== Nichtelektrische Lichtquellen ====\nNichtelektrische Lichtquellen sind...]
#     such that it starts with a proper sentence.
#     """
#     answer_text = item["answers"][0]

#     # Process positive contexts
#     for idx, ctx in enumerate(item["positive_ctxs"]["text"]):
#         item["positive_ctxs"]["text"][idx] = process_context(ctx, answer_text)

#     # Process negative contexts
#     for idx, ctx in enumerate(item["hard_negative_ctxs"]["text"]):
#         item["hard_negative_ctxs"]["text"][idx] = process_context(ctx, answer_text)

#     return item


if __name__ == "__main__":
    # large spacy model, better sentence segmentation, still fast
    # python -m spacy download de_core_news_lg
    nlp = spacy.load("de_core_news_lg")
    enc = tiktoken.encoding_for_model("gpt-4") # estimate API cost

    # dataset = load_dataset("deepset/germanquad")
    dataset = load_dataset("deepset/germandpr")
    dataset = dataset["train"]

    # typo in Portugal context
    dataset = dataset.filter(
        lambda example: "Portugal" not in example["positive_ctxs"]["title"][0]
    )
    # Utrecht duplicate
    datset = dataset.filter(
        lambda example: "Bahnhof Utrecht Centraal" not in example["answers"][0]
    )
    # Oklahoma near duplicate
    datset = dataset.filter(
        lambda example: "Wie heißt das Stadion der Oklahoma City Thunder?"
        not in example["question"]
    )

    dataset = collapse_germandpr(dataset)
    dataset = pair_with_answer_starts(dataset)

    context_lengths = []
    token_counts = []
    none_count = 0  # Nones are answer spans going beyond a sentence
    with open("german_llm_data/germandpr_subset.jsonl", "w") as f:
        pass  # clear file
    for i in range(0, len(dataset), 1):  # Test
        question = dataset[i]["question"]
        # full_context = dataset[i]['context']
        full_context = dataset[i]["positive_ctxs"]["text"][0]
        answer_start = dataset[i]["answer_start"]
        answer_text = dataset[i]["answers"][0]
        print("EXAMPLE", i)
        print("FULL CONTEXT:", full_context, "\n")
        # sometimes necessary information spans multiple sentences
        # experimental: use LLM validation in `validate_generations.py`
        # relevant_context = get_three_sentence_context(full_context, answer_start)
        relevant_context = get_single_sentence_context(
            full_context, answer_start, answer_text
        )
        if relevant_context is None:
            none_count += 1
            continue
        print("RELEVANT CONTEXT:", relevant_context, "\n")
        print("QUESTION:", question, "\n")
        print("ANSWER:", answer_text, "\n")
        print("=" * 50)
        context_lengths.append(len(relevant_context))
        context_tokens = enc.encode(relevant_context)
        question_tokens = enc.encode(question)
        token_counts.append(len(context_tokens) + len(question_tokens))

        entry = {
            "question": question,
            "answer_span": answer_text,
            "sentence_context": relevant_context,
        }
        with open("germandpr_subset.jsonl", "a") as f:
            json.dump(entry, f)
            f.write("\n")

    avg_context_length = (
        sum(context_lengths) / len(context_lengths) if context_lengths else 0
    )
    print("Average length of relevant context:", avg_context_length)
    print("Number of None's returned:", none_count)
    print("Sum of all tokens:", sum(token_counts))  # 138594
