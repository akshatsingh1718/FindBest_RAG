import re
from typing import Union
from langchain.agents.agent import AgentOutputParser
from langchain.agents.conversational.prompt import FORMAT_INSTRUCTIONS
from langchain.schema import AgentAction, AgentFinish

class SalesConvoOutputParser(AgentOutputParser):
    ai_prefix: str= "AI"
    verbose: bool= False

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS
    
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if self.verbose:
        # if True:
            print("Text")
            print(text)
            print("-------------")

        if f"{self.ai_prefix}:" in text: # Ai thinks it dosent need to use tools
            # print("Output AgentFinish:" + "=" * 40)
            # print({"output" : text.split(f"{self.ai_prefix}:")[-1].strip()})
            # print(f"{text=}")

            return AgentFinish(
                {"output" : text.split(f"{self.ai_prefix}:")[-1].strip()},
                text
            )

        regex= r"Action: (.*?)[\n]*Action Input: (.*)"
        match= re.search(regex, text)

        if not match:
            print("Agent Error: " + "=" * 40)
            return AgentFinish(
                {
                    "output": "I apologize, I was unable to find the answer to your question. Is there anything else I can help with?"
                },
                text
            )
        
        action= match.group(1)
        action_input= match.group(2)

        # print("Output parser" + "=" * 40)
        # print(action)
        # print(action_input)
        # print("------------")
        # print(text)
        # print("------------")

        return AgentAction(action.strip(), action_input.strip(" ").strip('"'), text)
    
    @property
    def _type(self)->str:
        return "sales-agent"