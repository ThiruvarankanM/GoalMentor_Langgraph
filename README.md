# Goal Mentor 

Goal Mentor LangGraph is a professional goal achievement platform powered by [LangGraph](https://github.com/langchain-ai/langgraph) and [Groq Cloud API](https://groq.com/). It provides an intelligent, interactive system that helps you break down goals into structured 4-week implementation plans, track progress systematically, and receive personalized guidance when facing obstacles.

---

## Features

- **Intelligent Goal Planning**: AI-powered 4-week structured implementation plans
- **Progress Tracking**: Visual progress dashboard with completion rates
- **Professional Guidance**: Contextual assistance with actionable strategies and resources
- **Interactive Web Interface**: Clean, professional Streamlit-based UI
- **Resource Recommendations**: Targeted learning resources and professional tools
- **Milestone Management**: Step-by-step progress tracking with completion validation

---

## Setup Guide

Follow these steps to set up and run the project locally.

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/GoalMentor_Langgraph.git
cd GoalMentor_Langgraph
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Required Packages
```bash
pip install streamlit python-dotenv langchain-community langchain-core langchain-openai openai
```

### 4. Configure Environment Variables
Create a `.env` file in the project root with your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the Application
```bash
streamlit run main.py
```

The application will open in your browser at `http://localhost:8501`

---

## How It Works

### Goal Definition Phase
1. Enter your professional or personal development goal
2. The system generates a structured 4-week implementation plan
3. Each week focuses on a specific milestone toward your goal

### Progress Tracking Phase
1. Mark milestones as completed when finished
2. Request professional guidance if you encounter obstacles
3. Access additional resources and implementation strategies
4. Skip phases if needed while maintaining overall progress

### Completion & Achievement
1. Visual progress tracking with completion percentages
2. Professional celebration upon goal achievement
3. Option to set new goals and continue development

---
## 🎥 Working Demo Video

Watch the final demonstration of the **Goal Mentor LangGraph** project here:  
[![Demo Video](https://img.youtube.com/vi/Z1ZeO8YydRI/0.jpg)](https://youtu.be/Z1ZeO8YydRI)

Or check thorugh this link: [https://youtu.be/Z1ZeO8YydRI](https://youtu.be/Z1ZeO8YydRI)

## Example Usage

### Goal Input
```
Goal: Become a Machine Learning Engineer
```

### Generated Implementation Plan
```
Week 1: Complete foundational mathematics and statistics courses
Week 2: Learn Python programming and data manipulation libraries
Week 3: Study machine learning algorithms and frameworks
Week 4: Build and deploy a machine learning project portfolio
```

### Progress Interface
- **Mark as Completed**: Advance to next milestone
- **Request Assistance**: Get targeted strategies and resources
- **Additional Resources**: Access comprehensive support materials
- **Skip to Next Phase**: Move forward while noting incomplete items

---

## Professional Guidance System

When requesting assistance, the system provides:

### Actionable Strategies
- Specific, measurable steps to overcome current obstacles
- Professional best practices for the current milestone
- Time management and productivity recommendations

### Learning Resources
- Targeted educational content and tutorials
- Industry-specific training materials
- Certification programs and professional courses

### Implementation Support
- Structured sub-task breakdowns
- Progress monitoring techniques
- Professional network recommendations

---

## Technical Architecture

- **Frontend**: Streamlit web application with professional UI
- **AI Engine**: Groq Cloud API with LLaMA 3 70B model
- **Framework**: LangChain for AI integration and prompt management
- **Session Management**: Streamlit session state for progress persistence
- **Error Handling**: Comprehensive exception handling with user feedback

---

## API Requirements

You need a valid [Groq Cloud API key](https://console.groq.com/) to use this application. Groq provides fast inference for large language models with competitive pricing.

---

## Project Structure

```
GoalMentor_Langgraph/
├── main.py              # Main Streamlit application
├── .env                 # Environment variables (not tracked)
├── .env.example         # Environment variables template
├── requirements.txt     # Python dependencies
├── README.md           # Project documentation
└── .venv/              # Virtual environment (not tracked)
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

---

## License

MIT License © 2025

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.

---
