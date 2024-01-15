## Q. Different types of chat mode

**Ans.** There are several types of chat modes available in the llama_index.chat_engine.types.ChatMode class. Here are the different types:

BEST: This mode selects the best chat engine based on the current Language Learning Model (LLM). If you're using an OpenAI model that supports the function calling API, it corresponds to the OpenAIAgent. Otherwise, it corresponds to the ReActAgent.

CONDENSE_QUESTION: This mode corresponds to the CondenseQuestionChatEngine. It first generates a standalone question from the conversation context and the last message, then queries the query engine for a response.

CONTEXT: This mode corresponds to the ContextChatEngine. It first retrieves text from the index using the user's message, then uses the context in the system prompt to generate a response.

OPENAI: This mode corresponds to the OpenAIAgent. It uses an OpenAI function calling agent loop. Note that this only works with OpenAI models that support the function calling API.

REACT: This mode corresponds to the ReActAgent. It uses a ReAct agent loop with query engine tools.

SIMPLE: This mode corresponds to the SimpleChatEngine. It allows you to chat with the LLM without making use of a knowledge base.

Each of these modes has its own use cases and can be chosen based on the specific requirements of your chat application.

# Q. how to add message history

**Ans.** To add a message to the chat history, you can use the put method of the ChatMemoryBuffer class. This method takes a ChatMessage as an argument. Here's an example of how you can use it:

```python
from llama_index.memory import ChatMemoryBuffer, ChatMessage

# Initialize the ChatMemoryBuffer
chat_memory_buffer = ChatMemoryBuffer()

# Create a ChatMessage
message = ChatMessage(role='user', content='Hello, world!')

# Add the message to the chat history
chat_memory_buffer.put(message)
```

In this example, a ChatMessage is created with the role of 'user' and the content 'Hello, world!'. This message is then added to the chat history using the put method of the ChatMemoryBuffer instance.

Please note that the ChatMessage class has two properties: role and content. The role can be one of the following: 'system', 'user', 'assistant', or 'function'. The content is the actual text of the message.


# Q. how to add system message

**Ans** To add a system message to the chat history, you can use the put method of the ChatMemoryBuffer class. This method takes a ChatMessage as an argument. Here's an example of how you can use it:

```python
from llama_index.memory import ChatMemoryBuffer, ChatMessage

# Initialize the ChatMemoryBuffer
chat_memory_buffer = ChatMemoryBuffer()

# Create a ChatMessage with role 'system'
message = ChatMessage(role='system', content='System message content')

# Add the message to the chat history
chat_memory_buffer.put(message)
```

In this example, a ChatMessage is created with the role of 'system' and the content 'System message content'. This message is then added to the chat history using the put method of the ChatMemoryBuffer instance.