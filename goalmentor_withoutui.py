import os
from dotenv import load_dotenv
from typing import Annotated, Literal
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

load_dotenv()

# LLM Configuration
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
    model="llama3-70b-8192",
    temperature=0.3
)

# State Definition
class GoalState(TypedDict):
    goal: str | None
    steps: list[str]
    current_step_index: int
    progress: Literal["asking_goal", "tracking", "done"]
    messages: Annotated[list[BaseMessage], add_messages]

def get_helpful_resources(goal: str, step: str) -> str:
    """Generate helpful tips and resources using LLM"""
    
    prompt = f"""
    The user is working on this goal: "{goal}"
    They are stuck on this specific step: "{step}"
    
    Provide practical help in this exact format:

    💡 PRACTICAL TIPS:
    • [Tip 1 - specific, actionable advice]
    • [Tip 2 - specific, actionable advice] 
    • [Tip 3 - specific, actionable advice]

    📺 RECOMMENDED YOUTUBE SEARCHES:
    • "[search term 1]"
    • "[search term 2]"
    • "[search term 3]"

    🔗 HELPFUL RESOURCES:
    • [Website/platform 1 - specific recommendation]
    • [Website/platform 2 - specific recommendation]
    • [Website/platform 3 - specific recommendation]

    📝 ACTION PLAN:
    Break this step into 3 smaller mini-steps:
    1. [Mini-step 1]
    2. [Mini-step 2] 
    3. [Mini-step 3]

    Keep responses practical and specific to their goal and step.
    """
    
    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        return f"""
💡 PRACTICAL TIPS:
• Break this step into smaller, manageable tasks
• Set a specific deadline for completion
• Find an accountability partner to check your progress

📺 RECOMMENDED YOUTUBE SEARCHES:
• "how to {step.lower()}"
• "{goal.lower()} beginner guide" 
• "step by step {step.lower()}"

🔗 HELPFUL RESOURCES:
• Reddit communities related to your goal
• Online forums and discussion groups
• Free online courses on relevant topics

📝 ACTION PLAN:
1. Identify the main obstacle preventing completion
2. Research solutions using the resources above
3. Take one small action today toward this step
"""

def run_goal_mentor():
    print("Goal Mentor Initialized.")
    
    # Initial state
    goal = None
    steps = []
    current_step_index = 0
    
    # Ask for goal
    print("\nMentor: What is one goal you're working toward? (e.g., get a job, learn Python, lose weight)")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == "exit":
            print("Session ended. Stay consistent and keep progressing!")
            break
        
        # If we don't have a goal yet, this input should be the goal
        if not goal:
            goal = user_input
            print(f"\nGreat! Your goal is: {goal}")
            print("Let me create a 4-week plan for you...")
            
            # Generate plan using LLM
            messages = [
                SystemMessage(content="Break the user's goal into 4 clear, simple weekly steps. Format as a numbered list (1. step one\\n2. step two\\n etc.)."),
                HumanMessage(content=f"My goal is: {goal}")
            ]
            
            try:
                response = llm.invoke(messages)
                text = response.content.strip() if response else ""
                
                # Parse steps from response
                steps = []
                for line in text.split("\n"):
                    line = line.strip()
                    if line and len(line) > 2 and line[0].isdigit() and ". " in line:
                        step = line.split(". ", 1)[-1].strip()
                        steps.append(step)
                
                if not steps:
                    print("\nMentor: I couldn't create a plan. Please describe your goal differently.")
                    goal = None  # Reset to ask again
                    continue
                
                # Show the plan
                print("\nMentor: Here is your 4-week plan:")
                for i, step in enumerate(steps):
                    print(f"{i+1}. {step}")
                
                # Ask about first step
                current_step_index = 0
                print(f"\nLet's start with week 1. Have you completed this step?")
                print(f"➡️ {steps[0]} (yes/no)")
                
            except Exception as e:
                print(f"\nMentor: Sorry, I had trouble creating your plan. Error: {e}")
                print("Please try describing your goal again.")
                goal = None
                continue
        
        # If we have a goal and steps, handle progress tracking
        elif steps:
            user_response = user_input.lower()
            
            if "yes" in user_response:
                # User completed current step
                current_step_index += 1
                
                if current_step_index >= len(steps):
                    # All steps completed!
                    print("\nMentor: 🎉 Congratulations! You've completed all steps for your goal. Well done!")
                    break
                else:
                    # Move to next step
                    next_step = steps[current_step_index]
                    print(f"\nMentor: ✅ Great job! Next step (Week {current_step_index + 1}):")
                    print(f"➡️ {next_step}")
                    print("\nHave you completed this step? (yes/no)")
            
            elif "no" in user_response:
                # User hasn't completed current step - provide help!
                current_step = steps[current_step_index]
                print(f"\nMentor: No worries! Let me help you with:")
                print(f"➡️ {current_step}")
                
                print("\n" + "="*60)
                print("🚀 HERE'S HOW I CAN HELP YOU:")
                print("="*60)
                
                # Get personalized resources
                print("Generating personalized tips and resources...")
                resources = get_helpful_resources(goal, current_step)
                print(resources)
                
                print("\n" + "="*60)
                print("💪 YOU'VE GOT THIS!")
                print("="*60)
                
                print(f"\nTake your time with these resources. When you're ready, let me know:")
                print(f"Have you completed this step? (yes/no)")
                print("Or type 'help' for more resources, or 'skip' to move to the next step")
            
            elif "help" in user_response:
                # User wants more help
                current_step = steps[current_step_index]
                print(f"\nMentor: Let me give you more detailed guidance for:")
                print(f"➡️ {current_step}")
                
                # Get additional resources
                additional_help = f"""
🎯 ALTERNATIVE APPROACHES:
Try a different method if the current one isn't working:

📊 TRACK YOUR PROGRESS:
• Create a simple checklist
• Set daily mini-goals
• Use apps like Habitica or Todoist
• Track progress visually (calendar, chart)

🤝 GET SUPPORT:
• Join online communities (Reddit, Discord, Facebook groups)
• Find a study/workout/accountability buddy
• Consider hiring a coach or mentor
• Ask friends/family for encouragement

⚡ MOTIVATION BOOSTERS:
• Remind yourself WHY this goal matters to you
• Visualize how you'll feel when you complete it
• Celebrate small wins along the way
• Create a reward system for progress

🔄 COMMON OBSTACLES & SOLUTIONS:
• Lack of time → Schedule specific time blocks
• Lack of motivation → Start with just 5 minutes daily
• Don't know where to start → Break it into micro-steps
• Feel overwhelmed → Focus on just today's task
                """
                
                print(additional_help)
                print(f"\nReady to try again? Have you completed this step? (yes/no)")
            
            elif "skip" in user_response:
                # User wants to skip current step
                current_step_index += 1
                
                if current_step_index >= len(steps):
                    print("\nMentor: You've reached the end of your plan!")
                    print("🎉 Even if you skipped some steps, you've made progress. Keep going!")
                    break
                else:
                    next_step = steps[current_step_index]
                    print(f"\nMentor: Okay, let's move to the next step (Week {current_step_index + 1}):")
                    print(f"➡️ {next_step}")
                    print("\nHave you completed this step? (yes/no)")
            
            else:
                # Invalid response
                current_step = steps[current_step_index]
                print(f"\nMentor: Please respond with:")
                print("• 'yes' - if you completed the step")
                print("• 'no' - if you need help with the step") 
                print("• 'help' - for additional resources")
                print("• 'skip' - to move to the next step")
                print(f"\nCurrent step: ➡️ {current_step}")

if __name__ == "__main__":
    print("Welcome to Goal Mentor.")
    print("Type 'exit' to end the session.")
    run_goal_mentor()