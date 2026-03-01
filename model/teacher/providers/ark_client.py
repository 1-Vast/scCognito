from __future__ import annotations

from typing import Optional
from openai import OpenAI


class ArkChatClient:
    """
    OpenAI-compatible ChatCompletions client for Ark (Volcengine).
    NOTE:
      - cli.py may pass enable_web_search/web_search_max_keyword.
      - To keep compatibility, we accept these args even if we don't enable tools here.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_id: str,
        enable_web_search: bool = True,
        web_search_max_keyword: int = 2,
        timeout: Optional[float] = None,
    ) -> None:
        self.model_id = model_id
        self.enable_web_search = bool(enable_web_search)
        self.web_search_max_keyword = int(web_search_max_keyword)

        # Keep OpenAI SDK usage to match your existing teacher pipeline
        # (Ark uses OpenAI-compatible base_url)
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def chat(self, system_prompt: str, user_content: str) -> str:
        """
        Teacher currently expects JSON-only output; we keep it simple.

        If later you confirm Ark supports tool calling/web_search in chat.completions,
        you can add tools/tool_choice here when self.enable_web_search is True.
        """
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        return (resp.choices[0].message.content or "").strip()