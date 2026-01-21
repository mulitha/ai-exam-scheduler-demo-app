import streamlit as st
import pandas as pd
import random
from datetime import datetime, timedelta
import joblib
import plotly.express as px

# Title of the application
st.title("AI-based Exam Scheduler")

# Sidebar menu for app navigation
st.sidebar.title("🧭 Menu")
st.sidebar.markdown("Navigate through the app")
menu = st.sidebar.radio("", ["Dashboard", "Prediction - Generate Timetable", "Calendar"])

try:
    # Load all required datasets from CSV files
    classrooms = pd.read_csv("sampleDataset/classrooms.csv")
    coursesDatSet = pd.read_csv("sampleDataset/courses.csv")
    instructors = pd.read_csv("sampleDataset/instructors.csv")
    scheduleDataSet = pd.read_csv("sampleDataset/schedule.csv")
    studentsDataSet = pd.read_csv("sampleDataset/students.csv")
    timeslotsDataSet = pd.read_csv("sampleDataset/timeslots.csv")
    shedulePredSet = pd.read_csv("sampleDataset/Prediction/New_Shedule_df.csv")

    # Check if essential datasets are loaded and non-empty
    if any(df is None or df.empty for df in [coursesDatSet, scheduleDataSet, studentsDataSet, timeslotsDataSet]):
        st.warning("Please upload all required CSV files (courses, schedule, students, timeslots) to continue.")
    else:
        # Convert time-related columns to datetime.time
        for col in ['start_time', 'end_time']:
            timeslotsDataSet[col] = pd.to_datetime(timeslotsDataSet[col], format='%H:%M').dt.time

        # --- Dashboard Section ---
        if menu == "Dashboard":
            st.subheader("📊 Dashboard")
            st.write("Welcome to the Exam Scheduler Dashboard!")

            # Split layout into two columns (first row)
            col1, col2 = st.columns(2)

            with col1:
                # Visualize average study days grouped by course difficulty
                st.markdown("### 🧠 Study Days by Difficulty")
                avg_study = shedulePredSet.groupby('difficulty_level')['study_date_count'].mean().reset_index()
                fig = px.pie(avg_study, values='study_date_count', names='difficulty_level', hole=0.4,
                            title="Average Study Days by Difficulty Level")
                st.plotly_chart(fig)

            with col2:
                # Bar chart showing number of instructors per department
                st.markdown("### 👩‍🏫 Instructor Count by Department")
                instructor_count = instructors['department'].value_counts().reset_index()
                instructor_count.columns = ['Department', 'Number of Instructors']
                st.bar_chart(instructor_count.set_index('Department'))

            # Full-width charts for courses by department and study day distribution
            st.markdown("### 📚 Course Count by Department")
            dept_course_count = coursesDatSet['department'].value_counts().reset_index()
            dept_course_count.columns = ['Department', 'Number of Courses']
            st.bar_chart(dept_course_count.set_index('Department'))

            st.markdown("### 📅 Distribution of Study Days")
            st.bar_chart(shedulePredSet['study_date_count'].value_counts().sort_index())


        # --- Timetable Generation Section ---
        elif menu == "Prediction - Generate Timetable":
            st.subheader("🧠 Generate Exam Timetable")

            # User input for filtering available timeslots
            selectedDays = st.multiselect("Select Exam Days", options=timeslotsDataSet['day'].unique(), default=timeslotsDataSet['day'].unique())
            min_break_minutes = st.number_input("Minimum Break Between Exams (in minutes)", min_value=0, value=240, step=30)

           

            # Filter available timeslots based on user input
            filteredTimeslotsDataFrame = timeslotsDataSet[timeslotsDataSet['day'].isin(selectedDays)]

            # Only allow weekdays (optional)
            weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            filteredTimeslotsDataFrame = filteredTimeslotsDataFrame[filteredTimeslotsDataFrame['day'].isin(weekdays)]

            # Only allow times between 9 AM and 5 PM
            filteredTimeslotsDataFrame = filteredTimeslotsDataFrame[
                (pd.to_datetime(filteredTimeslotsDataFrame['start_time'].astype(str)).dt.hour >= 9) &
                (pd.to_datetime(filteredTimeslotsDataFrame['end_time'].astype(str)).dt.hour <= 17)
            ]


            # Create helper dictionaries for scheduling
            studentCourseMap = scheduleDataSet.groupby('student_id')['course_id'].apply(set).to_dict()
            courseDifficulty = coursesDatSet.set_index('course_id')['difficulty_level'].to_dict()
            instructorCourseMap = scheduleDataSet.groupby('instructor_id')['course_id'].apply(set).to_dict()
            courseInstructorMap = scheduleDataSet.set_index('course_id')['instructor_id'].to_dict()

            # Genetic Algorithm Parameters
            POP_SIZE = 20
            GENERATIONS = 50
            MUTATION_RATE = 0.1


            # Define fitness function for evaluating a schedule
            def fitness(schedule):
                penalty = 0

                # ----- student conflict setup -----
              

                    # Instructor conflicts & break penalties
                instructorTimes = {inst: [] for inst in instructorCourseMap}
                for course_id, timeslot in schedule.items():
                    inst = courseInstructorMap.get(course_id)
                    if inst is not None:
                        instructorTimes[inst].append(timeslot)

                for inst, times in instructorTimes.items():
                    times.sort()
                    # 1) Penalize double-booking:
                    if len(times) != len(set(times)):
                        penalty += 10

                    # 2) Penalize insufficient break between any two exams:
                    for i in range(1, len(times)):
                        t1 = filteredTimeslotsDataFrame[filteredTimeslotsDataFrame['timeslot_id'] == times[i-1]]
                        t2 = filteredTimeslotsDataFrame[filteredTimeslotsDataFrame['timeslot_id'] == times[i]]
                        if not t1.empty and not t2.empty:
                            # extract end and start times safely
                            t1_end = t1.iloc[0]['end_time']
                            t2_start = t2.iloc[0]['start_time']
                            # convert to datetime for diff
                            dt1 = datetime.combine(datetime.today(), t1_end)
                            dt2 = datetime.combine(datetime.today(), t2_start)
                            if (dt2 - dt1).total_seconds() / 60 < min_break_minutes:
                                penalty += 5

                return -penalty

            # Generate a new random schedule (individual in population)
            def create_individual():
                return {
                    course_id: (
                        random.choice(filteredTimeslotsDataFrame['timeslot_id'].values),
                        random.choice(classrooms['classroom_id'].values),
                        courseInstructorMap.get(course_id)
                    )
                    for course_id in coursesDatSet['course_id'].values
                }


            # Combine two parent schedules into one child
            def crossover(parent1, parent2):
                child = {}
                for course_id in parent1:
                    if random.random() < 0.5:
                        child[course_id] = parent1[course_id]
                    else:
                        child[course_id] = parent2[course_id]
                return child


            # Mutate a schedule to introduce variation
            def mutate(individual):
                if random.random() < MUTATION_RATE:
                    course_id = random.choice(list(individual.keys()))
                    new_tid = random.choice(filteredTimeslotsDataFrame['timeslot_id'].values)
                    new_clid = random.choice(classrooms['classroom_id'].values)
                    iid = individual[course_id][2]  # keep the same instructor ID
                    individual[course_id] = (new_tid, new_clid, iid)
                return individual


            # Main genetic algorithm loop
            def genetic_algorithm():
                population = [create_individual() for _ in range(POP_SIZE)]
                for gen in range(GENERATIONS):
                    population.sort(key=fitness, reverse=True)
                    next_gen = population[:10]
                    while len(next_gen) < POP_SIZE:
                        parents = random.sample(population[:30], 2)
                        child = crossover(parents[0], parents[1])
                        child = mutate(child)
                        next_gen.append(child)
                    population = next_gen
                best = max(population, key=fitness)
                return best


            # When the Generate Schedule button is clicked
            if st.button("Generate Exam Schedule"):
                with st.spinner("Running Genetic Algorithm..."):
                    schedule = genetic_algorithm()
                    rows = []
                    for cid, (tid, clid, Iid) in schedule.items():
                        course_row = coursesDatSet[coursesDatSet['course_id'] == cid]
                        tslot_row = filteredTimeslotsDataFrame[filteredTimeslotsDataFrame['timeslot_id'] == tid]
                        class_row = classrooms[classrooms['classroom_id'] == clid]
                        instructor_row = instructors[instructors['instructor_id'] == Iid]
                        if not tslot_row.empty:
                            start_val = tslot_row['start_time'].values[0]
                            end_val = tslot_row['end_time'].values[0]
                            day_val = tslot_row['day'].values[0]

                            start = start_val.strftime('%H:%M') if hasattr(start_val, 'strftime') else str(start_val)
                            end = end_val.strftime('%H:%M') if hasattr(end_val, 'strftime') else str(end_val)
                            day = str(day_val)
                        else:
                            start = end = day = "N/A"

                        course_id = course_row['course_id'].values[0] if not course_row.empty else "N/A"
                        course_name = course_row['course_name'].values[0] if not course_row.empty else "N/A"
                        difficulty = course_row['difficulty_level'].values[0] if not course_row.empty else "N/A"
                        classroom_name = class_row['building_name'].iloc[0] if not class_row.empty else "N/A"
                        instructor = instructor_row['last_name'].iloc[0] if not instructor_row.empty else "N/A"
                        rows.append({
                        'Course_ID': course_row['course_id'].values[0] if not course_row.empty else cid,
                        'Course': course_row['course_name'].values[0] if not course_row.empty else "Unknown",
                        'Difficulty': course_row['difficulty_level'].values[0] if not course_row.empty else "N/A",
                        'Timeslot': tid,
                        'Day': day,
                        'Start': start,
                        'End': end,
                        'Class': class_row['building_name'].iloc[0] if not class_row.empty else "N/A",
                        'Instructor': instructor_row['last_name'].iloc[0] if not instructor_row.empty else "N/A"
                    })


                    schedule_df = pd.DataFrame(rows)
                    st.session_state['schedule_df'] = schedule_df

                   

                    # 1) Pivot schedule_df into a nested dict: week → day → list of exams
                    schedule_dict = {"Week 1": {}}
                    for _, r in schedule_df.iterrows():
                        wk  = "Week 1"
                        day = r["Day"]
                        # one room + one instructor per exam in this demo
                        schedule_dict.setdefault(wk, {}).setdefault(day, []).append({
                            "code":        r["Course_ID"],
                            "title":       r["Course"],     
                            "time":        r["Start"],
                            "rooms":       [r["Class"]],    
                            "instructors": [r["Instructor"]]
                        })# wrap in list for consistency

                    # 2) Rendering  in user friendly way
                    for week, days in schedule_dict.items():
                        st.markdown(f"## {week}")
                        for day, exams in days.items():
                            st.markdown(f"### {day}")
                            lines = ["\t" + "-"*88]
                            for ex in exams:
                                room_str = " | ".join(f"Building : {rm}" for i, rm in enumerate(ex["rooms"]))
            
                                
                                instructor = " | ".join(f"Instructor: {rm}" for i, rm in enumerate(ex["instructors"]))
                                lines.append(
                                    f"\t| CourseId: {ex['code']} | CourseName: {ex['title']} | StartTime: {ex['time']} | {room_str} | {instructor} |"
                                )
                           
                                
                                spacer = " " * (len(str(ex["code"])) + 3 + 28 + 3 + len(str(ex["time"])) + 3)

        
                                lines.append("\t" + "-"*88)
                            st.code("\n".join(lines), language=None)

            # === Prediction Input Section ===
            st.markdown("### 📘 Study Day Prediction Inputs")

            # Difficulty level selection
            difficulty_input = st.selectbox("Select Difficulty Level", options=['low', 'medium', 'high'])

            day_input = st.selectbox("Select Day", options=timeslotsDataSet['day'].unique())
            start_input = st.time_input("Select Start Time")
            end_input = st.time_input("Select End Time")

            # Match the timeslot_id based on inputs
            matched_timeslot = timeslotsDataSet[
                (timeslotsDataSet['day'] == day_input) &
                (timeslotsDataSet['start_time'] == start_input) &
                (timeslotsDataSet['end_time'] == end_input)
            ]

            if matched_timeslot.empty:
                st.error("No matching timeslot found. Please check your input.")
            else:
                timeslot_id = int(matched_timeslot['timeslot_id'].values[0])


            # Classroom and Course ID
            classroom_B = st.selectbox("building_name", options=classrooms['building_name'].unique())
            classroom_R = st.selectbox("room_number", options=classrooms['room_number'].unique())
            # Match the timeslot_id based on inputs
            matched_classroom = classrooms[
                (classrooms['building_name'] == classroom_B) &
                (classrooms['room_number'] == classroom_R)
            ]

            if matched_classroom.empty:
                st.error("No matching classroom found. Please check your input.")
            else:
                classroom_id = int(matched_classroom['classroom_id'].values[0])

            course_name = st.selectbox("Select Course name", options=coursesDatSet['course_name'].unique())
            
            # Match the timeslot_id based on inputs
            matched_coursename = coursesDatSet[
                (coursesDatSet['course_name'] == course_name)
            ]

            if matched_coursename.empty:
                st.error("No matching course found. Please check your input.")
            else:
                course_id = int(matched_coursename['course_id'].values[0])


            if st.button("Predict Study Days"):
                try:
                    # Load trained model
                    model = joblib.load('GradientBoostingRegressor_model.pkl')

                    # Map difficulty level to numerical values
                    difficulty_map = {'low': 0, 'medium': 1, 'high': 2}
                    if difficulty_input not in difficulty_map:
                        st.error("Please select a valid difficulty level.")
                    else:
                        difficulty_encoded = difficulty_map[difficulty_input]

                        
                    # Create input DataFrame for prediction
                    input_df = pd.DataFrame([{
                        'difficulty_level': difficulty_encoded,
                        'timeslot_id': timeslot_id,
                        'classroom_id': classroom_id,
                        'course_id': course_id
                            }])

                    # Predict number of study days
                    predicted_days = model.predict(input_df)[0]

                    # Show result
                    st.success(f"📘 Predicted Number of Study Days: **{predicted_days:.2f}**")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")


        # --- Calendar View Section ---
        elif menu == "Calendar":
            st.subheader("📅 Calendar View")

            if 'schedule_df' not in st.session_state:
                st.warning("Please generate an exam schedule first in the 'Prediction - Generate Timetable' section.")
            else:
                schedule_df = st.session_state['schedule_df']

                # Sort by Day and Start Time
                sorted_schedule = schedule_df.sort_values(by=["Day", "Start"])

                # Display grouped by Day
                for day in sorted_schedule['Day'].unique():
                    with st.expander(f"📅 {day}"):
                        day_schedule = sorted_schedule[sorted_schedule['Day'] == day][["Course", "Difficulty", "Start", "End", "Timeslot"]]
                        st.table(day_schedule.set_index("Course"))


except Exception as e:
    print(f"Error - Something went wrong: {e}")
