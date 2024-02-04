# GermanRAG 🇩🇪📜🦜

This is the code used for creating [GermanRAG](https://huggingface.co/datasets/DiscoResearch/germanrag), a German dataset for finetuning LLMs on Retrieval Augmented Generation tasks (RAG).

## How to use
- Install the requirements with `pip install -r requirements.txt`.
- Generate `germandpr_subset.jsonl` with `python germandpr.py`
- Clone [Airoboros](https://github.com/jondurbin/airoboros), `pip install -e .` there and copy `germandpr_subset.jsonl` aswell as `config_germanrag.yaml` into the root directory.
- Copy `airoboros/instructors/germanrag.py` and `airoboros/instructors/prompts/germanrag.txt` from this repo to the respective directories in Airoboros.
- Add `from airoboros.instructors.germanrag import generate as germanrag_generator` [here](https://github.com/jondurbin/airoboros/blob/29aa8c7864371b0603c33f5ad303887670f62b78/airoboros/self_instruct.py#L1000).
- Add `"germanrag": germanrag_generator` [here](https://github.com/jondurbin/airoboros/blob/29aa8c7864371b0603c33f5ad303887670f62b78/airoboros/self_instruct.py#L1024).
- Run `airoboros generate-instructions --config-path config_germanrag.yaml`
- Copy your generated `instructions.jsonl` back into this repo's root directory.
- Optional: Validate generations with `python validate_generations.py`.
- Run `python germanrag.py` to generate the final dataset.

## Room for improvement
- Choose how to deduplicate/collapse the contexts in GermanDPR, i.e. on shortest, longest, first/random answer span.
- Fix function for three sentence context window.
- Experimental/Optional: Finish choping and mixing of contexts on chunk level.
- Add (true) negatives beyond hard negatives, by pairing with random/dissimilar contexts.
- Generalize to more datasets in SQuAD format.

## Acknowledgments
- The GermanRAG dataset is derived from [GermanDPR](https://www.deepset.ai/germanquad), see 'Acknowledgments' in the [dataset card](https://huggingface.co/datasets/DiscoResearch/germanrag#acknowledgements).
- Airoboros by [Jon Durbin](https://github.com/jondurbin), consider giving a [tip](https://github.com/jondurbin/airoboros/tree/29aa8c7864371b0603c33f5ad303887670f62b78?tab=readme-ov-file#support-the-work) ;)

## Collaborate
Feel free to open issues/PRs and come join us in our [Discord](https://discord.gg/A575uNaEfu)! 😊


Check out our models at [DiscoResearch](https://huggingface.co/DiscoResearch) 🪩🧪.
