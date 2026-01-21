# AI-Based Exam Scheduler

This project is an intelligent, AI-powered web application designed to automate and optimize the university exam scheduling process. It combines **Genetic Algorithms (GA)** for conflict-free timetable generation and **Machine Learning (Gradient Boosting Regressor)** for predicting the number of study days required per course based on difficulty and contextual factors.

## 🧠 Features

- **Automated Exam Scheduling**: Generates optimized, conflict-free exam timetables.
- **Study Day Prediction**: Predicts study time requirements using a trained ML model.
- **Constraint Handling**:
  - No overlapping exams for students.
  - No back-to-back hard exams.
  - No double-booked instructors.
  - All exams scheduled on weekdays between 9 AM–5 PM.
- **Interactive Dashboard**: Built using Streamlit with pages for dashboard, prediction, and calendar.
- **Visualizations**: Includes charts for difficulty vs. study time, instructor/course distribution, and prediction results.

## 💻 Used Technologies

- **Python**: 3.11.11  
- **Anaconda**: 2024.10-1  
- **Streamlit**: 1.45.1  
- **scikit-learn**: 1.2.2  
- **joblib**: 1.5.0  
- **pandas**: 1.5.3  
- **numpy**: 1.23.5  

## 🚀 How to Run

### Prerequisites

- Python 3.11 or Anaconda (2024.10-1)
- pip (Python package manager)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/mulitha/ai-exam-scheduler-demo-app.git
   python -m streamlit run app.py

### Host Link: https://ai-exam-scheduler-demo-app-bj7u7qrrvuuxfhs2wibhcu.streamlit.app/



![AI-bassed-exam-scheduler](https://github.com/mulitha/ai-exam-scheduler-demo-app/blob/efa3ab33a370537ef5bf098833fa0d10d07e0f5b/images/1.PNG)
![AI-bassed-exam-scheduler](https://github.com/mulitha/ai-exam-scheduler-demo-app/blob/efa3ab33a370537ef5bf098833fa0d10d07e0f5b/images/2.PNG)






  
