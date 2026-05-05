from agents.planner import Plan
from graph.workflow import create_workflow
workflow = create_workflow()

initial_state = {
    'user_request': 'Explain langgraph and compare it with langchain',
    'plan': None,
    'retrieved_context': None,
    'final_answer': None
}
final_state = workflow.invoke(initial_state)
plan: Plan  = final_state['plan']
retrieved_context = final_state['retrieved_context']
final_answer = final_state['final_answer']

print("Task type:", plan.task_type)
print("Needs retrieval:", plan.needs_retrieval)

print("\nSteps:")
for step in plan.steps:
    print("-", step)

print("\nRetrieved context:")
print(retrieved_context)

print("\nFinal answer:")
print(final_answer)