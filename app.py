import streamlit as st
import pandas as pd
import random
from datetime import datetime, timedelta

st.title("AI-based Exam Scheduler - DEMO")


try:


    ##used this while doing testings in streamlit online playground
    # st.markdown("""### 📥 Download Sample CSV Files""")
    # col1, col2 = st.columns(2)
    # with col1:
    #     with open("sampleDataset/courses.csv", "rb") as f:
    #         st.download_button("Download Courses CSV", f, file_name="courses.csv")

    #     with open("sampleDataset/students.csv", "rb") as f:
    #         st.download_button("Download Students CSV", f, file_name="students.csv")
    # with col2:
    #     with open("sampleDataset/schedule.csv", "rb") as f:
    #         st.download_button("Download Schedule CSV", f, file_name="schedule.csv")

    #     with open("sampleDataset/timeslots.csv", "rb") as f:
    #         st.download_button("Download Timeslots CSV", f, file_name="timeslots.csv")


    # # File upload
    # courcesFile = st.file_uploader("Upload Courses CSV", type="csv")
    # scheduleFile = st.file_uploader("Upload Schedule CSV", type="csv")
    # studentsFile = st.file_uploader("Upload Students CSV", type="csv")
    # timeslotsFile = st.file_uploader("Upload Timeslots CSV", type="csv")


    #Direct load data
    classrooms = pd.read_csv("sampleDataset/classrooms.csv")
    coursesDatSet = pd.read_csv("sampleDataset/courses.csv")
    instructors = pd.read_csv("sampleDataset./instructors.csv")
    scheduleDataSet = pd.read_csv("sampleDataset/schedule.csv")
    studentsDataSet = pd.read_csv("sampleDataset/students.csv")
    timeslotsDataSet = pd.read_csv("sampleDataset/timeslots.csv")


    #######################################################################


    # validation to check does the user upload the files or not
    if any(df is None or df.empty for df in [coursesDatSet, scheduleDataSet, studentsDataSet, timeslotsDataSet]):
        st.warning("Please upload all required CSV files (courses, schedule, students, timeslots) to continue.")
        
    else:

        # Load data
        # coursesDatSet = pd.read_csv(courcesFile)
        # scheduleDataSet = pd.read_csv(scheduleFile)
        # studentsDataSet = pd.read_csv(studentsFile)
        # timeslotsDataSet = pd.read_csv(timeslotsFile)


        # # Display loaded data
        # st.subheader("Courses")
        # st.dataframe(coursesDatSet.head())

        # st.subheader("Existing Schedule")
        # st.dataframe(scheduleDataSet.head())

        # st.subheader("Students")
        # st.dataframe(studentsDataSet.head())

        # st.subheader("Timeslots")
        # st.dataframe(timeslotsDataSet.head())


    #######################################################################

        
        # Convert time columns
        for col in ['start_time', 'end_time']:
            timeslotsDataSet[col] = pd.to_datetime(timeslotsDataSet[col], format='%H:%M').dt.time

        # User input parameters
        selectedDays = st.multiselect("Select Exam Days", options=timeslotsDataSet['day'].unique(), default=timeslotsDataSet['day'].unique())
        min_break_minutes = st.number_input("Minimum Break Between Exams (in minutes)", min_value=0, value=240, step=30)

        # Filter timeslots by selected days
        filteredTimeslotsDataFrame = timeslotsDataSet[timeslotsDataSet['day'].isin(selectedDays)]

        # Build helper structures
        studentCourseMap = scheduleDataSet.groupby('student_id')['course_id'].apply(set).to_dict()
        courseDifficulty = coursesDatSet.set_index('course_id')['difficulty_level'].to_dict()

        # Parameters
        POP_SIZE = 20
        GENERATIONS = 50
        MUTATION_RATE = 0.1

        # Fitness function
        def fitness(schedule):
            penalty = 0
            studentTimes = {s: [] for s in studentCourseMap}

            for course_id, timeslot in schedule.items():
                for student_id, courses in studentCourseMap.items():
                    if course_id in courses:
                        studentTimes[student_id].append((timeslot, courseDifficulty[course_id]))

            for times in studentTimes.values():
                times.sort()
                for i in range(len(times)):
                    for j in range(i+1, len(times)):
                        if times[i][0] == times[j][0]:
                            penalty += 5  # overlapping
                    if i > 0:
                        prev_slot = times[i-1]
                        curr_slot = times[i]
                        if prev_slot[1] == curr_slot[1] == 'hard':
                            t1 = datetime.strptime(filteredTimeslotsDataFrame.iloc[prev_slot[0]-1]['start_time'].strftime('%H:%M'), '%H:%M')
                            t2 = datetime.strptime(filteredTimeslotsDataFrame.iloc[curr_slot[0]-1]['start_time'].strftime('%H:%M'), '%H:%M')
                            if abs((t2 - t1).total_seconds()) < min_break_minutes * 60:
                                penalty += 3
            return -penalty

        # Initialization
        def create_individual():
            return {course_id: random.choice(filteredTimeslotsDataFrame['timeslot_id'].values) for course_id in coursesDatSet['course_id'].values}

        def crossover(parent1, parent2):
            child = {}
            for course_id in parent1:
                child[course_id] = parent1[course_id] if random.random() < 0.5 else parent2[course_id]
            return child

        def mutate(individual):
            if random.random() < MUTATION_RATE:
                course_id = random.choice(list(individual.keys()))
                individual[course_id] = random.choice(filteredTimeslotsDataFrame['timeslot_id'].values)
            return individual


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

        if st.button("Generate Exam Schedule"):
            with st.spinner("Running Genetic Algorithm..."):
                schedule = genetic_algorithm()
                schedule_df = pd.DataFrame([{
                    'Course': coursesDatSet[coursesDatSet['course_id'] == cid]['course_name'].values[0],
                    'Difficulty': coursesDatSet[coursesDatSet['course_id'] == cid]['difficulty_level'].values[0],
                    'Timeslot': tid,
                    'Day': filteredTimeslotsDataFrame[filteredTimeslotsDataFrame['timeslot_id'] == tid]['day'].values[0],
                    'Start': filteredTimeslotsDataFrame[filteredTimeslotsDataFrame['timeslot_id'] == tid]['start_time'].values[0],
                    'End': filteredTimeslotsDataFrame[filteredTimeslotsDataFrame['timeslot_id'] == tid]['end_time'].values[0]
                } for cid, tid in schedule.items()])

                st.success("Schedule Generated Successfully!")
                st.dataframe(schedule_df)


except EXception as e:
    st.error(f"Error - Something went wrong: {e}")

