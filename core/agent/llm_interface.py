from openai import OpenAI
from pydantic import BaseModel


class LLMInterface:

    def __init__(self, api_key, model="gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, system_prompt, user_prompt):

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0
            )

            if not response.choices:
                return ""

            message = response.choices[0].message

            if not message or not message.content:
                return ""

            return message.content

        except Exception as e:
            return f'{{"error": "LLM call failed", "details": "{str(e)}"}}'

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ) -> BaseModel:
        """
        Generate a structured response validated by a Pydantic model.
        """

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                response_format=response_model,
            )

            if not response.choices:
                raise RuntimeError("LLM returned no choices.")

            parsed = response.choices[0].message.parsed

            if parsed is None:
                raise RuntimeError("LLM returned no structured response.")

            return parsed

        except Exception as e:
            raise RuntimeError(
                f"Structured LLM call failed: {str(e)}"
            ) from e
