from typing import Callable
from langchain.prompts.base import StringPromptTemplate, StringPromptValue
import inspect
from utils.constant import Constant as C

class CustomPromptTemplateForTools(StringPromptTemplate):
    # The template to use
    template: str
    
    # The list of tools available
    tools_getter: Callable

    def format(self, **kwargs) -> str:
        # Get the intermediate steps 
        intermediate_steps= kwargs.pop(C.INTERMEDIATE_STEPS)
        thoughts= ""

        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\n{C.OBSERVATION}: {observation}\n{C.THOUGHT}: "

        # Set the agent_scratchpad variable to that value
        kwargs[C.AGENT_SCRATCHPAD]= thoughts

        tools= self.tools_getter(kwargs['input'])

        kwargs[C.TOOLS] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools]
        )

        kwargs[C.TOOL_NAMES]= ", ".join([tool.name for tool in tools])

        return self.template.format(**kwargs)