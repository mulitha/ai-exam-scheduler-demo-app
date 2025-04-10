import streamlit as st
import pandas as pd
import random
from datetime import datetime, timedelta

st.title("AI-based Exam Scheduler (Genetic Algorithm)")

# File upload
courses_file = st.file_uploader("Upload Courses CSV", type="csv")
schedule_file = st.file_uploader("Upload Schedule CSV", type="csv")
students_file = st.file_uploader("Upload Students CSV", type="csv")
timeslots_file = st.file_uploader("Upload Timeslots CSV", type="csv")

if courses_file and schedule_file and students_file and timeslots_file:
    # Load data
    courses_df = pd.read_csv(courses_file)
    schedule_df = pd.read_csv(schedule_file)
    students_df = pd.read_csv(students_file)
    timeslots_df = pd.read_csv(timeslots_file)

    # Display data previews
    st.subheader("📚 Courses")
    st.dataframe(courses_df.head())

    st.subheader("👨‍🏫 Instructors")
    st.dataframe(instructors.head())

    st.subheader("📅 Existing Schedule")
    st.dataframe(schedule_df.head())

    st.subheader("👥 Students")
    st.dataframe(students_df.head())

    st.subheader("⏰ Timeslots")
    st.dataframe(timeslots_df.head())

    # Convert time columns
    for col in ['start_time', 'end_time']:
        timeslots_df[col] = pd.to_datetime(timeslots_df[col], format='%H:%M').dt.time

    # Build helper structures
    student_course_map = schedule_df.groupby('student_id')['course_id'].apply(set).to_dict()
    course_difficulty = courses_df.set_index('course_id')['difficulty_level'].to_dict()

    # Parameters
    POP_SIZE = 20
    GENERATIONS = 50
    MUTATION_RATE = 0.1

    # Fitness function
    def fitness(schedule):
        penalty = 0
        student_times = {s: [] for s in student_course_map}

        for course_id, timeslot in schedule.items():
            for student_id, courses in student_course_map.items():
                if course_id in courses:
                    student_times[student_id].append((timeslot, course_difficulty[course_id]))

        for times in student_times.values():
            times.sort()
            for i in range(len(times)):
                for j in range(i+1, len(times)):
                    if times[i][0] == times[j][0]:
                        penalty += 5  # overlapping
                if i > 0:
                    prev_slot = times[i-1]
                    curr_slot = times[i]
                    if prev_slot[1] == curr_slot[1] == 'hard':
                        t1 = datetime.strptime(timeslots_df.iloc[prev_slot[0]-1]['start_time'].strftime('%H:%M'), '%H:%M')
                        t2 = datetime.strptime(timeslots_df.iloc[curr_slot[0]-1]['start_time'].strftime('%H:%M'), '%H:%M')
                        if abs((t2 - t1).total_seconds()) < 14400:  # less than 4 hours
                            penalty += 3
        return -penalty

    # Initialization
    def create_individual():
        return {course_id: random.choice(timeslots_df['timeslot_id'].values) for course_id in courses_df['course_id'].values}

    def crossover(parent1, parent2):
        child = {}
        for course_id in parent1:
            child[course_id] = parent1[course_id] if random.random() < 0.5 else parent2[course_id]
        return child

    def mutate(individual):
        if random.random() < MUTATION_RATE:
            course_id = random.choice(list(individual.keys()))
            individual[course_id] = random.choice(timeslots_df['timeslot_id'].values)
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
                'Course': courses_df[courses_df['course_id'] == cid]['course_name'].values[0],
                'Timeslot': tid,
                'Day': timeslots_df[timeslots_df['timeslot_id'] == tid]['day'].values[0],
                'Start': timeslots_df[timeslots_df['timeslot_id'] == tid]['start_time'].values[0],
                'End': timeslots_df[timeslots_df['timeslot_id'] == tid]['end_time'].values[0]
            } for cid, tid in schedule.items()])

            st.success("Schedule Generated Successfully!")
            st.dataframe(schedule_df)
else:
    st.warning("Please upload all required CSV files (courses, schedule, students, timeslots) to continue.")
