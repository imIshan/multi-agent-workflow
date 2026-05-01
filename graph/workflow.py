from typing import TypedDict
from langgraph.graph import END, StateGraph
from agents.planner import Plan, create_planner_agent

class WorkFlowState(TypedDict):
    user_request: str
    plan: Plan | None

def planner_node(state: WorkFlowState) -> WorkFlowState:
    planner = create_planner_agent()
    plan: Plan = planner.invoke(state['user_request'])

    return {
        'user_request': state['user_request'],
        'plan': plan
    }

def create_workflow():
    graph = StateGraph(WorkFlowState)
    graph.add_node('planner', planner_node)
    graph.set_entry_point('planner')
    graph.add_edge('planner', END)
    return graph.compile()