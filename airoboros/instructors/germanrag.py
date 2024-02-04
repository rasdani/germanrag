### Adapted from https://github.com/jondurbin/airoboros
import os
import re
import datasets


async def generate(
    instructor,
    start_key="BEGINANSWER",
    end_key="ENDANSWER",
    filter_response=True,
    template_kwargs={},
    **kwargs,
):
    """Generator for generic inline question answer pair training data."""
    dataset = datasets.load_dataset("json", data_files="germandpr_subset.jsonl")["train"]

    category = "germanrag"
    config = instructor.instructors.get(category)
    if not config:
        return
    target_count = config.get("count")
    if target_count is None:
        target_count = instructor.default_count
    target_count = int(target_count)
    if not target_count:
        return

    # Load the prompt template.
    path = config.get("prompt_path", f"{category}.txt")
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", path)
    with open(path) as infile:
        template = infile.read()

    # API params, overriding defaults with this instructor's config.
    api_params = {**instructor.api_params, **config.get("api_params", {})}

    # Generate the instruction/response pairs until we reach the target count.
    # batch_size = config.get("batch_size")
    # if batch_size is None:
    #     batch_size = instructor.default_batch_size
    # batch_size = int(batch_size)
    # flesch = config.get("flesch") or instructor.default_flesch
    if category not in instructor.instructor_counts:
        instructor.instructor_counts[category] = 0
    # language = config.get("language") or instructor.language
    while instructor.instructor_counts[category] < target_count:
        item = dataset[instructor.instructor_counts[category]]
        context = item["sentence_context"]
        question = item["question"]
        answer_span = item["answer_span"]
        # prompt_args = {"flesch": flesch, "question": question, "context": context, "answer_span": answer_span}
        prompt_args = {"question": question, "context": context, "answer_span": answer_span}
        prompt = template.format(**prompt_args)
        response = await instructor.generate_response(
            prompt,
            messages=kwargs.get("messages", []),
            filter_response=filter_response,
            **api_params,
        )
        if not response:
            continue

        instruction = f"CONTEXT: {context}\nQUESTION: {question}\n"
        for response in re.findall(
            f"{start_key}\n(.*?)\n{end_key}", response, re.DOTALL
        ):
            yield {
                "instruction": instruction.strip(),
                "response": response.strip(),
                "category": category,
            }
            if instructor.instructor_counts[category] >= target_count:
                break
