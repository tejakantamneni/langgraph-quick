from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_anthropic import ChatAnthropic

from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_core.messages import BaseMessage

load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool = TavilySearchResults(max_results=2)
tools = [tool]

# tool_node = BasicToolNode(tools=[tool])
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

llm = ChatAnthropic(model="claude-3-haiku-20240307")
llm_with_tools = llm.bind_tools(tools)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {"tools": "tools", "__end__": "__end__"},
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

if __name__ == "__main__":
    print("begin...")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        for event in graph.stream({"messages": [("user", user_input)]}):
            for value in event.values():
                if isinstance(value["messages"][-1], BaseMessage):
                    print("Assistant:", value["messages"][-1].content)
