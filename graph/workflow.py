from typing import Literal, TypedDict
from langgraph.graph import END, StateGraph
from agents.planner import Plan, create_planner_agent
from agents.retriever import retrieve_context
from agents.writer import  create_writer_agent

class WorkFlowState(TypedDict):
    user_request: str
    plan: Plan | None
    final_answer: str | None
    retrieved_context: str | None

def planner_node(state: WorkFlowState) -> WorkFlowState:
    planner = create_planner_agent()
    plan: Plan = planner.invoke(state['user_request'])

    return {
        'user_request': state['user_request'],
        'plan': plan,
        'retrieved_context': state['retrieved_context'],
        'final_answer' : state['final_answer']
    }

def route_after_planner(state: WorkFlowState) -> Literal['retriever', 'writer']:
    plan = state['plan']
    if plan is None:
        raise ValueError('Plan is missing. Cannot work without a plan')
    if plan.needs_retrieval:
        return 'retriever'
    return 'writer'

def retrival_node(state: WorkFlowState) -> WorkFlowState:
    context = retrieve_context(state["user_request"])
    return {
        "user_request": state["user_request"],
        "plan": state["plan"],
        "retrieved_context": context,
        "final_answer": state["final_answer"]
    }

def write_node(state: WorkFlowState) -> WorkFlowState:
    writer = create_writer_agent()
    plan = state['plan']

    if plan is None:
        raise ValueError('Plan is missing. Writer cannot work without plan.')
    
    steps_text = "\n".join(f"- {step}" for step in plan.steps)

    retrieved_context = state['retrieved_context']
    if retrieve_context is None:
        retrieved_context = "No retrieved context was used."

    prompt = f"""
        You are a writer agent. Use the user's request, planner's steps  any retrieved context  to write a helpful final answer.

        User request:
        {state["user_request"]}

        Planner steps:
        {steps_text}

        Retrieved context:
        {retrieved_context}

        Final answer:
    """
    response = writer.invoke(prompt)
    return {
        "user_request": state["user_request"],
        "plan": plan,
        "retrieved_context": retrieved_context,
        "final_answer": response.content
    }

def create_workflow():
    graph = StateGraph(WorkFlowState)
    graph.add_node('planner', planner_node)
    graph.add_node('writer', write_node)
    graph.add_node('retriever', retrival_node)
    graph.set_entry_point('planner')
    graph.add_conditional_edges(
        'planner',
        route_after_planner,
        {
            "retriever": "retriever",
            "writer": "writer",
        }
    )
    graph.add_edge('retriever', 'writer')
    graph.add_edge('writer', END)
    return graph.compile()