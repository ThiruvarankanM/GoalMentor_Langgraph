import os
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Annotated, Literal
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import json

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Goal Mentor AI",
    page_icon="🎯",
    layout="wide"
)

# Define the state structure
class GoalMentorState(TypedDict):
    goal: str
    goal_refined: bool
    steps: List[Dict[str, str]]  # Each step has title, description, tips
    current_step_index: int
    user_message: str
    assistant_response: str
    conversation_history: List[Dict[str, str]]
    mode: Literal["goal_setting", "planning", "execution", "help", "completed"]
    user_feedback: Dict[str, str]  # feedback on steps
    stuck_count: int  # how many times user asked for help
    completion_status: Dict[int, Dict]  # detailed completion info
    next_action: str

# Initialize LLM
@st.cache_resource
def init_llm():
    return ChatOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
        model="llama3-70b-8192",
        temperature=0.7
    )

# Check API key
if not os.environ.get("GROQ_API_KEY"):
    st.error("🔑 API key not found! Please add your Groq API key to the .env file")
    st.stop()

try:
    llm = init_llm()
except Exception as e:
    st.error(f"❌ Error connecting to AI: {e}")
    st.stop()

def create_conversation_context(state: GoalMentorState) -> List:
    """Create context from conversation history"""
    messages = []
    
    # Add system context
    system_msg = f"""You are a professional goal mentor. The user's goal is: "{state.get('goal', 'Not set yet')}"
    
Current situation:
- Mode: {state.get('mode', 'goal_setting')}
- Current step: {state.get('current_step_index', 0) + 1} of {len(state.get('steps', []))}
- Times asked for help: {state.get('stuck_count', 0)}

Be encouraging, practical, and conversational. Don't be robotic."""
    
    messages.append(SystemMessage(content=system_msg))
    
    # Add recent conversation
    for msg in state.get('conversation_history', [])[-6:]:  # Last 6 messages
        if msg['role'] == 'user':
            messages.append(HumanMessage(content=msg['content']))
        else:
            messages.append(AIMessage(content=msg['content']))
    
    return messages

def goal_analysis_node(state: GoalMentorState) -> GoalMentorState:
    """Analyze and potentially refine the user's goal"""
    goal = state["goal"]
    user_msg = state.get("user_message", "")
    
    # Check if goal needs refinement
    analysis_prompt = f"""
The user wants to achieve: "{goal}"

Analyze this goal and respond naturally:
1. Is this goal specific and actionable enough?
2. If not, ask 1-2 friendly questions to help refine it
3. If it's good, acknowledge it and show enthusiasm

Recent user message: "{user_msg}"

Be conversational and encouraging, not robotic.
"""
    
    try:
        messages = create_conversation_context(state)
        messages.append(HumanMessage(content=analysis_prompt))
        
        response = llm.invoke(messages)
        assistant_msg = response.content.strip()
        
        # Simple heuristic: if response contains questions, goal needs refinement
        needs_refinement = "?" in assistant_msg and not state.get("goal_refined", False)
        
        # Update conversation history
        new_history = state.get("conversation_history", []).copy()
        if user_msg:
            new_history.append({"role": "user", "content": user_msg})
        new_history.append({"role": "assistant", "content": assistant_msg})
        
        return {
            **state,
            "assistant_response": assistant_msg,
            "conversation_history": new_history,
            "mode": "goal_setting" if needs_refinement else "planning",
            "goal_refined": not needs_refinement
        }
        
    except Exception as e:
        return {
            **state,
            "assistant_response": f"I'm having trouble connecting right now. Let's work with your goal: {goal}",
            "mode": "planning",
            "goal_refined": True
        }

def planning_node(state: GoalMentorState) -> GoalMentorState:
    """Create a smart, detailed plan"""
    goal = state["goal"]
    
    planning_prompt = f"""
Create a 4-week plan for: "{goal}"

Return ONLY a JSON object with this exact structure:
{{
    "steps": [
        {{
            "title": "Week 1: [Short descriptive title]",
            "description": "[What to focus on this week]",
            "tips": "[2-3 practical tips for success]",
            "key_actions": "[Specific things to do]"
        }},
        // ... 3 more weeks
    ],
    "encouragement": "[Personal encouraging message about their goal]"
}}

Make it practical, achievable, and motivating. Each week should build on the previous one.
"""
    
    try:
        messages = create_conversation_context(state)
        messages.append(HumanMessage(content=planning_prompt))
        
        response = llm.invoke(messages)
        content = response.content.strip()
        
        # Try to parse JSON
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        
        plan_data = json.loads(content)
        steps = plan_data.get("steps", [])
        encouragement = plan_data.get("encouragement", "Great goal! Let's make it happen.")
        
        # Update conversation
        new_history = state.get("conversation_history", []).copy()
        new_history.append({"role": "assistant", "content": f"{encouragement}\n\nI've created your 4-week plan! Ready to start with Week 1?"})
        
        return {
            **state,
            "steps": steps,
            "assistant_response": f"{encouragement}\n\nI've created your 4-week plan! Ready to start with Week 1?",
            "conversation_history": new_history,
            "mode": "execution",
            "current_step_index": 0
        }
        
    except:
        # Fallback plan
        fallback_steps = [
            {
                "title": f"Week 1: Foundation Building",
                "description": f"Set up the basics for achieving {goal}",
                "tips": "Start small, be consistent, track your progress",
                "key_actions": "Research, plan, and take the first small step"
            },
            {
                "title": f"Week 2: Skill Development", 
                "description": f"Focus on building key skills needed for {goal}",
                "tips": "Practice daily, seek feedback, don't fear mistakes",
                "key_actions": "Dedicate time each day to skill-building activities"
            },
            {
                "title": f"Week 3: Implementation",
                "description": f"Apply what you've learned toward {goal}",
                "tips": "Take action, measure results, adjust as needed",
                "key_actions": "Execute your plan and track concrete progress"
            },
            {
                "title": f"Week 4: Optimization",
                "description": f"Refine your approach and plan next steps for {goal}",
                "tips": "Reflect on what worked, optimize your methods",
                "key_actions": "Fine-tune your approach and set up for continued success"
            }
        ]
        
        new_history = state.get("conversation_history", []).copy()
        new_history.append({"role": "assistant", "content": "I've created a solid 4-week plan for you! Let's start with Week 1."})
        
        return {
            **state,
            "steps": fallback_steps,
            "assistant_response": "I've created a solid 4-week plan for you! Let's start with Week 1.",
            "conversation_history": new_history,
            "mode": "execution",
            "current_step_index": 0
        }

def execution_node(state: GoalMentorState) -> GoalMentorState:
    """Handle ongoing execution and user interactions"""
    current_idx = state["current_step_index"]
    steps = state["steps"]
    user_msg = state.get("user_message", "").lower()
    
    # Check if all steps completed
    if current_idx >= len(steps):
        return {
            **state,
            "mode": "completed"
        }
    
    current_step = steps[current_idx]
    
    # Determine user intent
    if any(word in user_msg for word in ["done", "complete", "finished", "next"]):
        # User completed current step
        completion_status = state.get("completion_status", {})
        completion_status[current_idx] = {
            "completed_at": "now",  # In real app, use datetime
            "user_feedback": user_msg
        }
        
        new_idx = current_idx + 1
        
        if new_idx >= len(steps):
            # All done!
            congrats_msg = f"🎉 Congratulations! You've completed all 4 weeks of your goal plan for '{state['goal']}'! That's amazing dedication."
        else:
            next_step = steps[new_idx]
            congrats_msg = f"✅ Awesome! You've completed {current_step['title']}!\n\nReady for {next_step['title']}?\n\n{next_step['description']}"
        
        new_history = state.get("conversation_history", []).copy()
        new_history.append({"role": "user", "content": state.get("user_message", "")})
        new_history.append({"role": "assistant", "content": congrats_msg})
        
        return {
            **state,
            "current_step_index": new_idx,
            "completion_status": completion_status,
            "assistant_response": congrats_msg,
            "conversation_history": new_history,
            "mode": "completed" if new_idx >= len(steps) else "execution"
        }
    
    elif any(word in user_msg for word in ["help", "stuck", "difficult", "hard", "don't know"]):
        # User needs help
        return {
            **state,
            "mode": "help",
            "stuck_count": state.get("stuck_count", 0) + 1
        }
    
    else:
        # General conversation about current step
        step_context = f"""
The user is working on: {current_step['title']}
Description: {current_step['description']}
Tips: {current_step['tips']}

User said: "{state.get('user_message', '')}"

Respond conversationally. Be encouraging and provide relevant advice for their current step.
"""
        
        try:
            messages = create_conversation_context(state)
            messages.append(HumanMessage(content=step_context))
            
            response = llm.invoke(messages)
            assistant_msg = response.content.strip()
            
            new_history = state.get("conversation_history", []).copy()
            new_history.append({"role": "user", "content": state.get("user_message", "")})
            new_history.append({"role": "assistant", "content": assistant_msg})
            
            return {
                **state,
                "assistant_response": assistant_msg,
                "conversation_history": new_history
            }
            
        except:
            return {
                **state,
                "assistant_response": f"I understand you're working on {current_step['title']}. {current_step['tips']} What specific part would you like to focus on?",
            }

def help_node(state: GoalMentorState) -> GoalMentorState:
    """Provide targeted help when user is stuck"""
    current_idx = state["current_step_index"]
    current_step = state["steps"][current_idx]
    stuck_count = state.get("stuck_count", 0)
    user_msg = state.get("user_message", "")
    
    # Customize help based on how many times they've been stuck
    if stuck_count == 1:
        help_level = "gentle encouragement and tips"
    elif stuck_count == 2:
        help_level = "more specific, actionable advice"
    else:
        help_level = "very detailed, step-by-step breakdown"
    
    help_prompt = f"""
The user is stuck on: {current_step['title']}
Description: {current_step['description']}
They said: "{user_msg}"
Times they've asked for help: {stuck_count}

Provide {help_level}. Be understanding and practical. 

If they've been stuck multiple times, consider suggesting they might want to break this down into smaller parts or try a different approach.
"""
    
    try:
        messages = create_conversation_context(state)
        messages.append(HumanMessage(content=help_prompt))
        
        response = llm.invoke(messages)
        assistant_msg = response.content.strip()
        
        new_history = state.get("conversation_history", []).copy()
        new_history.append({"role": "user", "content": user_msg})
        new_history.append({"role": "assistant", "content": assistant_msg})
        
        return {
            **state,
            "assistant_response": assistant_msg,
            "conversation_history": new_history,
            "mode": "execution"  # Return to normal execution
        }
        
    except:
        fallback_help = f"""I can see you're having trouble with {current_step['title']}. That's totally normal!
        
Here are some ways to approach it:
1. Break it into smaller 15-minute tasks
2. Focus on just one small part today
3. Ask yourself: what's the smallest step I could take right now?

{current_step['tips']}

What feels most manageable to you?"""
        
        return {
            **state,
            "assistant_response": fallback_help,
            "mode": "execution"
        }

def completion_node(state: GoalMentorState) -> GoalMentorState:
    """Handle goal completion celebration"""
    goal = state["goal"]
    completion_status = state.get("completion_status", {})
    
    celebration_msg = f"""🎉 CONGRATULATIONS! 🎉

You've successfully completed your 4-week journey toward "{goal}"!

Here's what you accomplished:
"""
    
    for i, step in enumerate(state["steps"]):
        if i in completion_status:
            celebration_msg += f"✅ {step['title']}\n"
        else:
            celebration_msg += f"📝 {step['title']}\n"
    
    celebration_msg += f"""
That's incredible dedication! You should be proud of yourself.

What's next? Ready to set a new goal, or do you want to keep building on this one?"""
    
    new_history = state.get("conversation_history", []).copy()
    new_history.append({"role": "assistant", "content": celebration_msg})
    
    return {
        **state,
        "assistant_response": celebration_msg,
        "conversation_history": new_history,
    }

def route_conversation(state: GoalMentorState) -> str:
    """Smart routing based on state and user intent"""
    mode = state.get("mode", "goal_setting")
    user_msg = state.get("user_message", "").lower()
    
    # Check for new goal request
    if any(phrase in user_msg for phrase in ["new goal", "start over", "different goal"]):
        return "goal_analysis"
    
    # Route based on current mode
    if mode == "goal_setting":
        return "goal_analysis"
    elif mode == "planning":
        return "planning"  
    elif mode == "help":
        return "help"
    elif mode == "completed":
        return "completion"
    else:  # execution mode
        return "execution"

# Create the LangGraph
def create_goal_mentor_graph():
    workflow = StateGraph(GoalMentorState)
    
    # Add all the nodes
    workflow.add_node("goal_analysis", goal_analysis_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("execution", execution_node)
    workflow.add_node("help", help_node)
    workflow.add_node("completion", completion_node)
    
    # Set entry point
    workflow.set_entry_point("goal_analysis")
    
    # Add conditional routing
    workflow.add_conditional_edges(
        "goal_analysis",
        lambda state: "planning" if state.get("mode") == "planning" else "goal_analysis"
    )
    
    workflow.add_edge("planning", "execution")
    
    workflow.add_conditional_edges(
        "execution", 
        route_conversation,
        {
            "execution": "execution",
            "help": "help", 
            "completion": "completion",
            "goal_analysis": "goal_analysis"
        }
    )
    
    workflow.add_edge("help", "execution")
    workflow.add_edge("completion", END)
    
    return workflow.compile(checkpointer=MemorySaver())

# Initialize the app
def initialize_session_state():
    if 'graph' not in st.session_state:
        st.session_state.graph = create_goal_mentor_graph()
    
    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = "goal_mentor_session"
    
    if 'current_state' not in st.session_state:
        st.session_state.current_state = {
            "goal": "",
            "goal_refined": False,
            "steps": [],
            "current_step_index": 0,
            "conversation_history": [],
            "mode": "goal_setting",
            "user_feedback": {},
            "stuck_count": 0,
            "completion_status": {},
            "assistant_response": "Hi! I'm your Goal Mentor. What goal would you like to work on together?",
            "user_message": "",
            "next_action": ""
        }

# Main Streamlit App
def main():
    st.title("🎯 Goal Mentor AI")
    st.markdown("*Your intelligent companion for achieving meaningful goals*")
    
    initialize_session_state()
    
    # Sidebar for progress and controls
    with st.sidebar:
        st.header("📊 Progress Dashboard")
        
        current_state = st.session_state.current_state
        
        if current_state.get("goal"):
            st.success(f"**Goal:** {current_state['goal']}")
            
            if current_state.get("steps"):
                st.write("**4-Week Plan:**")
                steps = current_state['steps']
                current_idx = current_state.get('current_step_index', 0)
                
                for i, step in enumerate(steps):
                    if i < current_idx:
                        st.write(f"✅ {step['title']}")
                    elif i == current_idx:
                        st.write(f"🔄 {step['title']}")
                    else:
                        st.write(f"⏳ {step['title']}")
                
                # Progress bar
                progress = min(current_idx / len(steps), 1.0)
                st.progress(progress)
                st.write(f"Progress: {int(progress * 100)}%")
        
        st.divider()
        
        # Quick actions
        st.write("**Quick Actions:**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🆘 Need Help", use_container_width=True):
                st.session_state.pending_message = "I'm stuck and need help"
        
        with col2:
            if st.button("✅ Step Done", use_container_width=True):
                st.session_state.pending_message = "I completed this step!"
        
        if st.button("🎯 New Goal", type="secondary", use_container_width=True):
            # Reset everything
            st.session_state.current_state = {
                "goal": "",
                "goal_refined": False,
                "steps": [],
                "current_step_index": 0,
                "conversation_history": [],
                "mode": "goal_setting",
                "user_feedback": {},
                "stuck_count": 0,
                "completion_status": {},
                "assistant_response": "What new goal would you like to work on?",
                "user_message": "",
                "next_action": ""
            }
            st.rerun()
    
    # Main chat interface
    st.header("💬 Chat with Your Mentor")
    
    # Display conversation
    current_state = st.session_state.current_state
    
    # Show assistant's latest response
    if current_state.get("assistant_response"):
        with st.chat_message("assistant"):
            st.write(current_state["assistant_response"])
    
    # Show current step details if in execution mode
    if current_state.get("mode") == "execution" and current_state.get("steps"):
        current_idx = current_state.get("current_step_index", 0)
        if current_idx < len(current_state["steps"]):
            current_step = current_state["steps"][current_idx]
            
            st.info(f"""
**{current_step['title']}**

{current_step['description']}

💡 **Tips:** {current_step['tips']}

🎯 **Key Actions:** {current_step.get('key_actions', 'Focus on the description above')}
""")
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    # Handle pending messages from sidebar buttons
    if 'pending_message' in st.session_state:
        user_input = st.session_state.pending_message
        del st.session_state.pending_message
    
    if user_input:
        # Show user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Process through LangGraph
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        new_state = {
            **current_state,
            "user_message": user_input
        }
        
        try:
            with st.spinner("🤔 Thinking..."):
                result = st.session_state.graph.invoke(new_state, config)
                st.session_state.current_state = result
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Oops! Something went wrong: {e}")
            # Fallback response
            st.session_state.current_state["assistant_response"] = "I'm having a technical hiccup. Could you try rephrasing that?"
            st.rerun()

if __name__ == "__main__":
    main()
