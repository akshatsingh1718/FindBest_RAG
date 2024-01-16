from pydantic import BaseModel
from typing import List, Sequence
from llama_index.langchain_helpers.agents import (
    IndexToolConfig,
    LlamaIndexTool,
)


class ToolsProvider(BaseModel):
    tools: List

    @staticmethod
    def from_tools(query_engine):
        tools_config = [
            IndexToolConfig(
                query_engine=query_engine,
                name=f"SearchMenu",
                description=f"use this tool when you need to search things related to the menu of the restraunt.",
                tool_kwargs={"return_direct": True},
            )
        ]

        tools = [
            LlamaIndexTool.from_tool_config(tool_config) for tool_config in tools_config
        ]
        return tools
