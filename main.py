import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Goal Mentor",
    page_icon="📋",
    layout="wide"
)

# Initialize LLM
@st.cache_resource
def init_llm():
    return ChatOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
        model="llama3-70b-8192",
        temperature=0.3
    )

# Initialize session state
if 'goal' not in st.session_state:
    st.session_state.goal = None
if 'steps' not in st.session_state:
    st.session_state.steps = []
if 'current_step_index' not in st.session_state:
    st.session_state.current_step_index = 0
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Check if API key exists
if not os.environ.get("GROQ_API_KEY"):
    st.error("API key not found in environment variables.")
    st.info("Please add your Groq API key to the .env file")
    st.stop()

# Initialize LLM only after API key check
try:
    llm = init_llm()
except Exception as e:
    st.error(f"Error initializing language model: {e}")
    st.info("Please check your API key and internet connection")
    st.stop()

def generate_plan(goal):
    """Generate a 4-week plan for the given goal"""
    messages = [
        SystemMessage(content="Break the user's goal into 4 clear, actionable weekly steps. Format as a numbered list (1. step one\n2. step two\n etc.). Each step should be specific and measurable."),
        HumanMessage(content=f"My goal is: {goal}")
    ]
    
    try:
        response = llm.invoke(messages)
        text = response.content.strip()
        
        steps = []
        for line in text.split("\n"):
            line = line.strip()
            if line and len(line) > 2 and line[0].isdigit() and ". " in line:
                step = line.split(". ", 1)[-1].strip()
                steps.append(step)
        
        return steps
    except Exception as e:
        st.error(f"Error generating plan: {e}")
        return []

def get_helpful_resources(goal, step):
    """Generate helpful tips and resources using LLM"""
    prompt = f"""
    The user is working on this goal: "{goal}"
    They need assistance with this specific step: "{step}"
    
    Provide professional guidance in this exact format:

    ACTIONABLE STRATEGIES:
    • [Strategy 1 - specific, professional advice]
    • [Strategy 2 - specific, professional advice] 
    • [Strategy 3 - specific, professional advice]

    RECOMMENDED LEARNING RESOURCES:
    • "[educational search term 1]"
    • "[educational search term 2]"
    • "[educational search term 3]"

    PROFESSIONAL RESOURCES:
    • [Platform/website 1 - specific recommendation]
    • [Platform/website 2 - specific recommendation]
    • [Platform/website 3 - specific recommendation]

    IMPLEMENTATION PLAN:
    Break this step into 3 manageable sub-tasks:
    1. [Sub-task 1]
    2. [Sub-task 2] 
    3. [Sub-task 3]

    Provide professional, actionable advice specific to their goal and current step.
    """
    
    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        return f"""
ACTIONABLE STRATEGIES:
• Break this step into smaller, manageable tasks with specific deadlines
• Create a structured timeline with measurable milestones
• Establish accountability measures to track your progress

RECOMMENDED LEARNING RESOURCES:
• "how to {step.lower()}"
• "{goal.lower()} professional guide" 
• "step by step {step.lower()}"

PROFESSIONAL RESOURCES:
• Industry-specific online communities and forums
• Professional development platforms and courses
• Relevant certification programs and training materials

IMPLEMENTATION PLAN:
1. Identify the key requirements and obstacles for this step
2. Research best practices and industry standards
3. Execute one concrete action toward completing this step
"""

# Main app header
st.title("Goal Mentor")
st.markdown("### Professional Goal Achievement Platform")

# Sidebar for progress tracking
with st.sidebar:
    st.header("Progress Dashboard")
    
    if st.session_state.goal:
        st.success(f"**Current Goal:** {st.session_state.goal}")
        
        if st.session_state.steps:
            st.write("**4-Week Implementation Plan:**")
            for i, step in enumerate(st.session_state.steps):
                if i < st.session_state.current_step_index:
                    st.write(f"✓ Week {i+1}: {step}")
                elif i == st.session_state.current_step_index:
                    st.write(f"→ Week {i+1}: {step}")
                else:
                    st.write(f"○ Week {i+1}: {step}")
            
            # Progress bar
            progress = st.session_state.current_step_index / len(st.session_state.steps)
            st.progress(progress)
            st.write(f"Completion Rate: {int(progress * 100)}%")
    
    st.divider()
    
    # Reset button
    if st.button("Start New Goal", type="secondary"):
        st.session_state.goal = None
        st.session_state.steps = []
        st.session_state.current_step_index = 0
        st.session_state.chat_history = []
        st.rerun()

# Main content area
if not st.session_state.goal:
    # Step 1: Get user's goal
    st.header("Goal Definition")
    st.markdown("Define your professional or personal development goal. Our system will create a structured 4-week implementation plan.")
    
    # Examples
    with st.expander("Example Goals"):
        st.write("""
        • Secure a position as a software developer
        • Master Python programming fundamentals
        • Achieve target weight loss of 10 pounds
        • Launch a side business venture
        • Gain fluency in Spanish language
        • Develop a professional portfolio website
        • Complete 12 books on professional development
        • Establish a consistent fitness routine
        """)
    
    goal_input = st.text_input("Enter your goal:", placeholder="e.g., Become a Machine Learning Engineer")
    
    if st.button("Generate Implementation Plan", type="primary"):
        if goal_input.strip():
            st.session_state.goal = goal_input.strip()
            
            with st.spinner("Creating your structured implementation plan..."):
                steps = generate_plan(st.session_state.goal)
                
                if steps:
                    st.session_state.steps = steps
                    st.session_state.current_step_index = 0
                    st.success("Implementation plan created successfully.")
                    st.rerun()
                else:
                    st.error("Unable to create implementation plan. Please refine your goal description.")
        else:
            st.warning("Please enter a goal before proceeding.")

else:
    # Step 2: Track progress
    st.header(f"Goal: {st.session_state.goal}")
    
    if st.session_state.steps:
        # Check if all steps completed FIRST
        if st.session_state.current_step_index >= len(st.session_state.steps):
            st.balloons()
            st.success("Congratulations! You have successfully completed all implementation steps.")
            st.markdown("### Goal Achievement Complete")
            
            if st.button("Set New Goal"):
                st.session_state.goal = None
                st.session_state.steps = []
                st.session_state.current_step_index = 0
                st.session_state.chat_history = []
                st.rerun()
        else:
            # Get current step only if we haven't completed all steps
            current_step = st.session_state.steps[st.session_state.current_step_index]
            
            # Current step tracking
            st.subheader(f"Week {st.session_state.current_step_index + 1} Implementation")
            
            st.info(f"**Current Milestone:** {current_step}")
            
            st.markdown("### Progress Status")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("Mark as Completed", type="primary"):
                    st.session_state.current_step_index += 1
                    st.success("Milestone completed. Proceeding to next phase.")
                    st.rerun()
            
            with col2:
                if st.button("Request Assistance"):
                    st.session_state.show_help = True
                    st.rerun()
            
            with col3:
                if st.button("Additional Resources"):
                    st.session_state.show_extra_help = True
                    st.rerun()
            
            with col4:
                if st.button("Skip to Next Phase"):
                    st.session_state.current_step_index += 1
                    st.warning("Phase skipped. You may return to this milestone later.")
                    st.rerun()
            
            # Show help if requested
            if hasattr(st.session_state, 'show_help') and st.session_state.show_help:
                st.divider()
                st.header("Professional Guidance")
                
                with st.spinner("Generating targeted assistance..."):
                    resources = get_helpful_resources(st.session_state.goal, current_step)
                
                st.markdown(resources)
                
                if st.button("Close Assistance Panel"):
                    st.session_state.show_help = False
                    st.rerun()
            
            # Show extra help if requested
            if hasattr(st.session_state, 'show_extra_help') and st.session_state.show_extra_help:
                st.divider()
                st.header("Comprehensive Support Resources")
                
                st.markdown("""
                ### Alternative Implementation Approaches:
                Consider different methodologies if your current approach requires adjustment.
                
                ### Progress Monitoring Tools:
                • Develop systematic tracking mechanisms
                • Establish measurable daily objectives  
                • Utilize productivity applications (Notion, Asana, Trello)
                • Implement visual progress indicators
                
                ### Professional Network Support:
                • Engage with industry-specific communities and forums
                • Establish mentorship or accountability partnerships
                • Consider professional coaching or consulting services
                • Leverage colleague and peer feedback systems
                
                ### Motivation and Consistency Strategies:
                • Clarify the strategic importance of this goal
                • Visualize long-term professional benefits
                • Implement milestone-based reward systems
                • Maintain focus on incremental progress
                
                ### Common Implementation Challenges:
                • **Time management constraints** → Block dedicated time slots for goal activities
                • **Lack of motivation** → Start with focused 15-minute daily sessions  
                • **Unclear next steps** → Break objectives into micro-tasks
                • **Feeling overwhelmed** → Focus exclusively on today's specific actions
                """)
                
                if st.button("Close Resource Panel"):
                    st.session_state.show_extra_help = False
                    st.rerun()

# Footer
st.divider()
st.markdown("**Goal Mentor** - Professional Goal Achievement Platform")