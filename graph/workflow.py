from typing import TypedDict
from langgraph.graph import END, StateGraph
from agents.planner import Plan, create_planner_agent
from agents.writer import  create_writer_agent

class WorkFlowState(TypedDict):
    user_request: str
    plan: Plan | None
    final_answer: str | None

def planner_node(state: WorkFlowState) -> WorkFlowState:
    planner = create_planner_agent()
    plan: Plan = planner.invoke(state['user_request'])

    return {
        'user_request': state['user_request'],
        'plan': plan,
        'final_answer' : state['final_answer']
    }

def write_node(state: WorkFlowState) -> WorkFlowState:
    writer = create_writer_agent()
    plan = state['plan']

    if plan is None:
        raise ValueError('Plan is missing. Writer cannot work without plan.')
    
    steps_text = "\n".join(f"- {step}" for step in plan.steps)

    prompt = f"""
        You are a writer agent. Use the user's request and the planner's steps to write a helpful final answer.

        User request:
        {state["user_request"]}

        Planner steps:
        {steps_text}

        Final answer:
    """
    response = writer.invoke(prompt)
    return {
        "user_request": state["user_request"],
        "plan": plan,
        "final_answer": response.content
    }

def create_workflow():
    graph = StateGraph(WorkFlowState)
    graph.add_node('planner', planner_node)
    graph.add_node('writer', write_node)
    graph.set_entry_point('planner')
    graph.add_edge('planner', 'writer')
    graph.add_edge('writer', END)
    return graph.compile()