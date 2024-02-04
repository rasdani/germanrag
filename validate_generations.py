### EXPERIMENTAL ###
# Use an LLM to validate the correctness of the generated answers.
# In this case, we validated if the single sentence containing the answer span was sufficient to answer the question.
# We initially planed to chop and mix the contexts more fine-grained,i.e. on sentence level.
import time
from dotenv import load_dotenv
import json
from typing import List, Optional
from openai import OpenAI
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
import instructor

load_dotenv()

base_url = "YOUR_FAVORITE_COMPATIBLE_ENDPOINT"
model = "YOUR_BELOVED_MODEL"
api_key = "YOUR_API_KEY" # you can set this in the .env file
client = instructor.patch(OpenAI(base_url=base_url))


# https://jxnl.github.io/instructor/blog/2023/11/18/validate-citations/
class Validation(BaseModel):
    is_valid: bool
    error_messages: Optional[str] = Field(None, description="Error messages if any")


class Statements(BaseModel):
    generated_answer: str

    @model_validator(mode="after")
    def substring_quote_exists(self, info: ValidationInfo):
        context = info.context.get("text_chunks", None)
        context = context[1]

        resp: Validation = client.chat.completions.create(
            response_model=Validation,
            messages=[
                {
                    "role": "user",
                    "content": f"Enthält der folgende Kontext alle Informationen, auf die die Antwort Bezug nimmt?\n\nAntwort: {self.generated_answer}\n\nKontext: {context}",
                }
            ],
            model=model,
        )

        if resp.is_valid:
            return self

        raise ValueError(resp.error_messages)


class AnswerWithContext(BaseModel):
    answer: List[Statements]


file_path = "instructions.jsonl"
dataset = []
with open(file_path, "r") as file:
    lines = file.readlines()
    start = time.time()
    for i, line in enumerate(lines):
        item = json.loads(line)
        instruction = item["instruction"]
        context, _ = instruction.split("\nQUESTION: ")
        sentence_context = context.replace("CONTEXT: ", "")
        answer = item["response"].strip()
        print(f"{'=' * 50} Item {i} of {len(lines)} {'=' * 50}")
        print("CONTEXT:", sentence_context)
        print("ANSWER:", answer, "\n")
        try:
            AnswerWithContext.model_validate(
                {
                    "answer": [
                        {"generated_answer": answer},
                    ],
                },
                context={
                    "text_chunks": {
                        1: sentence_context,
                    }
                },
            )
        except ValidationError as e:
            print(e)
            with open("invalid_items.jsonl", "a") as f:
                f.write(json.dumps(item) + "\n")
            continue

        with open("valid_items.jsonl", "a") as f:
            f.write(json.dumps(item) + "\n")

    end = time.time()
    print(f"Time: {end - start}")
