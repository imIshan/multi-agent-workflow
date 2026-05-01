from agents.planner import Plan
from graph.workflow import create_workflow
workflow = create_workflow()

initial_state = {
    'user_request': 'Explain langgraph and compare it with langchain',
    'plan': None
}
final_state = workflow.invoke(initial_state)
plan: Plan  = final_state['plan']

print("Task type:", plan.task_type)
print("Needs retrieval:", plan.needs_retrieval)

print("\nSteps:")
for step in plan.steps:
    print("-", step)