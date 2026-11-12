
import json
import sqlite3
from pathlib import Path

import streamlit as st
import pandas as pd

import gapfinder as gf


# ---------------------------------------------------------
# PAGE CONFIG + STYLING
# ---------------------------------------------------------

st.set_page_config(
    page_title="GapFinder AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px;}
    [data-testid="stSidebar"] {border-right: 1px solid rgba(120,120,120,.18);}
    .app-subtitle {color: #7b8494; font-size: .95rem; margin-top: -.7rem;}
    .section-note {color: #7b8494; font-size: .88rem;}
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,.07);
        border: 1px solid rgba(128,128,128,.13);
        padding: 14px 16px;
        border-radius: 12px;
    }
    .source-chip {
        display: inline-block; padding: 4px 9px; border-radius: 20px;
        background: rgba(59,130,246,.10); color: #3975c6;
        font-size: .78rem; margin: 4px 0;
    }
    div.stButton > button {border-radius: 9px; font-weight: 600;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def load_report():
    if not gf.REPORT_PATH.exists():
        return None
    with open(gf.REPORT_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_progress():
    conn = sqlite3.connect(gf.DB_PATH)
    rows = conn.execute("""
        SELECT attempt_date, topic, score, total_questions
        FROM attempts
        ORDER BY attempt_date
    """).fetchall()
    conn.close()

    return pd.DataFrame(
        rows,
        columns=["Date", "Topic", "Score", "Questions"]
    )


def status_label(status):
    colors = {
        "WEAK": "🔴",
        "MEDIUM": "🟡",
        "STRONG": "🟢"
    }
    return f"{colors.get(status, '⚪')} {status.title()}"


def build_quiz():
    topics = gf.load_topics()
    questions = []

    progress = st.progress(0, text="Generating questions from your materials...")
    for index, topic in enumerate(topics):
        generated = gf.generate_questions(topic, number_of_questions=2)
        questions.extend(generated)
        progress.progress(
            (index + 1) / len(topics),
            text=f"Preparing questions: {topic['name']}"
        )
    progress.empty()

    if not questions:
        raise ValueError(
            "No questions could be generated. Check that your uploaded PDFs "
            "contain readable text and relevant topic content."
        )
    return questions


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

defaults = {
    "quiz_questions": None,
    "quiz_submitted": False,
    "quiz_report": None,
    "quiz_feedback": None,
    "revision_plan": None,
    "revision_days": 5,
    "revision_hours": 2.0,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

gf.initialize_database()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    st.markdown("## 🎯 GapFinder AI")
    st.markdown(
        '<p class="app-subtitle">Know your gaps. Study smarter.</p>',
        unsafe_allow_html=True
    )
    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Study materials",
            "Diagnostic quiz",
            "Gap analysis",
            "Revision plan",
            "Topic assistant",
            "Progress"
        ],
        label_visibility="visible"
    )

    st.divider()


# ---------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------

if page == "Overview":
    st.title("Your learning overview")
    st.markdown(
        '<p class="app-subtitle">A clear picture of what you know and what to revise next.</p>',
        unsafe_allow_html=True
    )

    topics = []
    if gf.TOPICS_PATH.exists():
        topics = gf.load_topics()

    report = load_report()
    material_count = len(list(gf.UPLOAD_DIR.glob("*.pdf")))

    col1, col2, col3 = st.columns(3)
    col1.metric("Study materials", material_count)
    col2.metric("Topics mapped", len(topics))
    col3.metric(
        "Topics diagnosed",
        len(report) if report else 0
    )

    st.markdown("### Start here")
    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown("#### 1. Add your study materials")
            st.write("Upload your syllabus, notes and past question papers.")

    with right:
        with st.container(border=True):
            st.markdown("#### 2. Take a diagnostic quiz")
            st.write("Answer questions generated from your own material.")

    if report:
        st.markdown("### Latest topic diagnosis")
        report_df = pd.DataFrame([
            {
                "Topic": topic,
                "Score": data["score"],
                "Status": status_label(data["status"]),
                "Exam importance": f"{data.get('exam_weight', 3)}/5"
            }
            for topic, data in report.items()
        ])
        st.dataframe(report_df, use_container_width=True, hide_index=True)
    else:
        st.info("Upload your PDFs and complete a diagnostic quiz to see your learning diagnosis.")


# ---------------------------------------------------------
# STUDY MATERIALS
# ---------------------------------------------------------

elif page == "Study materials":
    st.title("Study materials")
    st.markdown(
        '<p class="app-subtitle">Manage the PDFs that power your knowledge base.</p>',
        unsafe_allow_html=True
    )

    existing = sorted(gf.UPLOAD_DIR.glob("*.pdf"))
    st.markdown("### Current documents")
    if existing:
        st.caption(f"{len(existing)} PDF(s) currently in your knowledge base")
        for file in existing:
            file_col, action_col = st.columns([5, 1])
            file_col.write(f"📄 **{file.name}**")
            file_col.caption(f"{file.stat().st_size / 1024:.1f} KB")
            if action_col.button("Delete", key=f"delete_{file.name}"):
                try:
                    with st.spinner(f"Removing {file.name} and rebuilding the knowledge base..."):
                        gf.remove_document(file.name)
                    st.session_state.quiz_questions = None
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_report = None
                    st.session_state.quiz_feedback = None
                    st.session_state.revision_plan = None
                    st.success(f"{file.name} was removed. Remaining documents were re-indexed.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not remove {file.name}: {exc}")
    else:
        st.info("No PDFs uploaded yet.")

    st.divider()
    st.markdown("### Add study materials")
    upload_mode = st.radio(
        "How should these PDFs be added?",
        ["Add to existing materials", "Replace all existing materials"],
        horizontal=True,
        help="Replace removes every existing PDF and rebuilds the knowledge base using only the newly selected PDFs."
    )

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Use filenames containing 'syllabus' or 'paper' to identify those document types."
    )

    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) selected**")
        for file in uploaded_files:
            st.write(f"📄 {file.name} · {file.size / 1024:.1f} KB")

        if st.button("Process selected PDFs", type="primary"):
            try:
                if upload_mode == "Replace all existing materials":
                    gf.clear_uploaded_documents()

                for file in uploaded_files:
                    safe_name = Path(file.name).name
                    if not safe_name.lower().endswith(".pdf"):
                        raise ValueError(f"Only PDF files are supported: {safe_name}")
                    destination = gf.UPLOAD_DIR / safe_name
                    destination.write_bytes(file.getbuffer())

                with st.spinner("Extracting, chunking, embedding and indexing your PDFs..."):
                    gf.ingest_documents()

                st.session_state.quiz_questions = None
                st.session_state.quiz_submitted = False
                st.session_state.quiz_report = None
                st.session_state.quiz_feedback = None
                st.session_state.revision_plan = None
                st.success("Your knowledge base has been updated successfully.")
                st.rerun()
            except Exception as exc:
                st.error(f"Document processing failed: {exc}")

    st.divider()
    st.markdown("### Reset knowledge base")
    st.caption(
        "This removes all uploaded PDFs, indexed chunks, the extracted topic map, "
        "and the latest gap report. Historical quiz attempts in Progress are kept."
    )
    confirm_reset = st.checkbox(
        "I understand that all current study materials and their diagnosis will be removed.",
        key="confirm_reset_knowledge_base"
    )
    if st.button(
        "Remove all study materials",
        disabled=not confirm_reset or not existing,
        type="secondary"
    ):
        try:
            gf.clear_uploaded_documents()
            st.session_state.quiz_questions = None
            st.session_state.quiz_submitted = False
            st.session_state.quiz_report = None
            st.session_state.quiz_feedback = None
            st.session_state.revision_plan = None
            st.success("All study materials and derived knowledge-base data were removed.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not reset the knowledge base: {exc}")

    if gf.TOPICS_PATH.exists():
        st.markdown("### Extracted topic map")
        for topic in gf.load_topics():
            with st.expander(
                f"{topic['name']} · Exam importance {topic.get('exam_weight', 3)}/5"
            ):
                for subtopic in topic.get("subtopics", []):
                    st.write(f"• {subtopic}")

# ---------------------------------------------------------
# DIAGNOSTIC QUIZ
# ---------------------------------------------------------

elif page == "Diagnostic quiz":
    st.title("Diagnostic quiz")
    st.markdown(
        '<p class="app-subtitle">Questions are generated from retrieved sections of your uploaded study material.</p>',
        unsafe_allow_html=True
    )

    if not gf.TOPICS_PATH.exists():
        st.warning("Upload and process your study materials before starting a quiz.")
    else:
        if st.session_state.quiz_questions is None:
            st.info("The quiz contains two questions per mapped topic.")
            if st.button("Generate diagnostic quiz", type="primary"):
                try:
                    with st.spinner("Retrieving relevant material and generating questions..."):
                        st.session_state.quiz_questions = build_quiz()
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_report = None
                    st.session_state.quiz_feedback = None
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not generate quiz: {exc}")

        if st.session_state.quiz_questions:
            questions = st.session_state.quiz_questions

            if not st.session_state.quiz_submitted:
                st.caption(f"{len(questions)} questions · Select one answer for each question")

                with st.form("diagnostic_quiz_form"):
                    answers = {}
                    for idx, question in enumerate(questions):
                        st.markdown(f"**Question {idx + 1} · {question['topic']}**")
                        st.write(question["question"])
                        answers[idx] = st.radio(
                            "Choose your answer",
                            options=["A", "B", "C", "D"],
                            format_func=lambda option, q=question: (
                                f"{option}. {q['options'][option]}"
                            ),
                            key=f"quiz_answer_{idx}",
                            index=None,
                            label_visibility="collapsed"
                        )
                        st.caption(f"Source: {question['citation']}")
                        st.divider()

                    submitted = st.form_submit_button(
                        "Submit answers",
                        type="primary",
                        use_container_width=True
                    )

                if submitted:
                    unanswered = [
                        idx + 1 for idx, answer in answers.items()
                        if answer is None
                    ]
                    if unanswered:
                        st.warning(
                            "Please answer all questions. Missing: "
                            + ", ".join(map(str, unanswered))
                        )
                    else:
                        grouped = {}
                        feedback = []

                        for idx, question in enumerate(questions):
                            topic = question["topic"]
                            grouped.setdefault(
                                topic,
                                {"correct": 0, "total": 0, "mistakes": []}
                            )
                            grouped[topic]["total"] += 1

                            selected = answers[idx]
                            correct = question["correct_answer"].upper()
                            is_correct = selected == correct

                            if is_correct:
                                grouped[topic]["correct"] += 1
                            else:
                                grouped[topic]["mistakes"].append(
                                    question["question"]
                                )

                            feedback.append({
                                "question": question["question"],
                                "topic": topic,
                                "selected": selected,
                                "correct": correct,
                                "is_correct": is_correct,
                                "explanation": question["explanation"],
                                "citation": question["citation"],
                                "options": question["options"]
                            })

                        topic_scores = {}
                        for topic, data in grouped.items():
                            topic_scores[topic] = {
                                **data,
                                "score": (
                                    data["correct"] / data["total"]
                                ) * 100
                            }

                        gf.save_attempt(topic_scores)
                        report = gf.analyze_gaps(topic_scores)

                        st.session_state.quiz_feedback = feedback
                        st.session_state.quiz_report = report
                        st.session_state.quiz_submitted = True
                        st.rerun()

            else:
                report = st.session_state.quiz_report or {}
                total_correct = sum(
                    item["correct"] for item in report.values()
                )
                total_questions = sum(
                    item["total"] for item in report.values()
                )
                overall = (
                    total_correct / total_questions * 100
                    if total_questions else 0
                )

                st.success("Assessment completed.")
                m1, m2, m3 = st.columns(3)
                m1.metric("Overall score", f"{overall:.0f}%")
                m2.metric("Correct answers", total_correct)
                m3.metric("Questions", total_questions)

                st.markdown("### Answer review")
                for idx, item in enumerate(st.session_state.quiz_feedback, 1):
                    icon = "✅" if item["is_correct"] else "❌"
                    with st.expander(f"{icon} Question {idx} · {item['topic']}"):
                        st.write(item["question"])
                        st.write(f"Your answer: **{item['selected']}**")
                        st.write(f"Correct answer: **{item['correct']} — {item['options'][item['correct']]}**")
                        st.write(item["explanation"])
                        st.caption(f"Source: {item['citation']}")

                if st.button("Start a new diagnostic quiz"):
                    st.session_state.quiz_questions = None
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_report = None
                    st.session_state.quiz_feedback = None
                    st.rerun()


# ---------------------------------------------------------
# GAP ANALYSIS
# ---------------------------------------------------------

elif page == "Gap analysis":
    st.title("Knowledge gap analysis")
    st.markdown(
        '<p class="app-subtitle">Your topic-level strengths and weaknesses, based on your latest diagnostic attempt.</p>',
        unsafe_allow_html=True
    )

    report = load_report()
    if not report:
        st.info("Complete a diagnostic quiz to generate your gap report.")
    else:
        rows = []
        for topic, data in report.items():
            rows.append({
                "Topic": topic,
                "Score": data["score"],
                "Correct": f"{data['correct']}/{data['total']}",
                "Status": data["status"],
                "Exam importance": data.get("exam_weight", 3),
                "Priority": data.get("priority", 0)
            })

        df = pd.DataFrame(rows).sort_values("Priority", ascending=False)

        weak = sum(1 for item in report.values() if item["status"] == "WEAK")
        medium = sum(1 for item in report.values() if item["status"] == "MEDIUM")
        strong = sum(1 for item in report.values() if item["status"] == "STRONG")

        c1, c2, c3 = st.columns(3)
        c1.metric("Weak topics", weak)
        c2.metric("Medium topics", medium)
        c3.metric("Strong topics", strong)

        st.markdown("### Topic performance")
        chart_df = df.set_index("Topic")[["Score"]]
        st.bar_chart(chart_df, y="Score", use_container_width=True)

        st.markdown("### Diagnosis details")
        for topic, data in sorted(
            report.items(),
            key=lambda pair: pair[1].get("priority", 0),
            reverse=True
        ):
            with st.container(border=True):
                top_left, top_right = st.columns([4, 1])
                top_left.markdown(f"**{topic}**")
                top_left.caption(
                    f"{data['correct']} of {data['total']} correct · "
                    f"Exam importance {data.get('exam_weight', 3)}/5"
                )
                top_right.metric("Score", f"{data['score']:.0f}%")
                st.progress(int(data["score"]))

                st.write(status_label(data["status"]))

                if data.get("mistakes"):
                    st.caption("Areas to revisit based on incorrect answers:")
                    for mistake in data["mistakes"]:
                        st.write(f"• {mistake}")
                else:
                    st.caption("No incorrect answers in this attempt.")


# ---------------------------------------------------------
# REVISION PLAN
# ---------------------------------------------------------

elif page == "Revision plan":
    st.title("Personalized revision plan")
    st.markdown(
        '<p class="app-subtitle">Allocate your study time according to topic weakness and exam importance.</p>',
        unsafe_allow_html=True
    )

    report = load_report()
    if not report:
        st.info("Complete a diagnostic quiz before generating a revision plan.")
    else:
        col1, col2 = st.columns(2)
        days = col1.number_input(
            "Days until exam",
            min_value=1,
            max_value=90,
            value=int(st.session_state.revision_days)
        )
        hours = col2.number_input(
            "Study hours per day",
            min_value=0.5,
            max_value=16.0,
            value=float(st.session_state.revision_hours),
            step=0.5
        )

        if st.button("Build my revision plan", type="primary"):
            st.session_state.revision_days = days
            st.session_state.revision_hours = hours
            st.session_state.revision_plan = gf.generate_revision_plan(
                report, int(days), float(hours)
            )

        plan = st.session_state.revision_plan
        if plan:
            st.markdown("### Your schedule")
            st.caption(
                f"{int(days)} days · {hours:g} hours/day · "
                f"{days * hours:g} total study hours"
            )

            for day, tasks in plan.items():
                with st.expander(f"Day {day}", expanded=(day == 1)):
                    if not tasks:
                        st.write("Revision and practice buffer")
                    for task in tasks:
                        st.markdown(
                            f"**{task['topic']}** · "
                            f"{task['hours']:.2f} hours · "
                            f"{task['status'].title()}"
                        )
                        st.progress(
                            min(100, int(task["hours"] / float(hours) * 100))
                        )


# ---------------------------------------------------------
# TOPIC ASSISTANT
# ---------------------------------------------------------

elif page == "Topic assistant":
    st.title("Explain a topic")
    st.markdown(
        '<p class="app-subtitle">Ask about a concept. Answers are grounded in your indexed study materials.</p>',
        unsafe_allow_html=True
    )

    if not gf.TOPICS_PATH.exists():
        st.info("Upload and process study materials first.")
    else:
        topic_names = [item["name"] for item in gf.load_topics()]
        selected_topic = st.selectbox(
            "Choose a syllabus topic",
            options=topic_names
        )
        custom_question = st.text_input(
            "What would you like explained?",
            placeholder="e.g. Explain the stages of the light reaction"
        )

        if st.button("Explain using my materials", type="primary"):
            query = custom_question.strip() or selected_topic
            try:
                with st.spinner("Retrieving relevant sections and preparing an explanation..."):
                    docs = gf.retrieve_context(query, top_k=5)
                    context = gf.format_context(docs)

                    prompt = f"""
                    Explain the following topic using ONLY the retrieved
                    study material. Do not add outside knowledge.

                    Student question: {query}
                    Syllabus topic: {selected_topic}

                    Retrieved context:
                    {context}

                    Explain clearly in simple language. If the retrieved
                    material does not contain enough information, say so.
                    Include citations using the source IDs.
                    """
                    answer = gf.ask_llm(prompt)

                st.markdown("### Explanation")
                st.write(answer)

                st.markdown("### Retrieved sources")
                for doc in docs:
                    st.caption(
                        f"📄 {doc['source']} · Page {doc['page']}"
                    )
            except Exception as exc:
                st.error(f"Could not generate explanation: {exc}")


# ---------------------------------------------------------
# PROGRESS
# ---------------------------------------------------------

elif page == "Progress":
    st.title("Learning progress")
    st.markdown(
        '<p class="app-subtitle">Track average performance and attempts by topic.</p>',
        unsafe_allow_html=True
    )

    progress_df = load_progress()

    if progress_df.empty:
        st.info("Your progress will appear after your first diagnostic quiz.")
    else:
        progress_df["Date"] = pd.to_datetime(
            progress_df["Date"], errors="coerce"
        )
        progress_df["Score"] = progress_df["Score"].astype(float)

        latest_date = progress_df["Date"].max()
        latest = progress_df[progress_df["Date"] == latest_date]

        avg_score = latest["Score"].mean() if not latest.empty else 0
        attempts = len(progress_df)

        c1, c2 = st.columns(2)
        c1.metric("Latest average topic score", f"{avg_score:.1f}%")
        c2.metric("Topic attempts recorded", attempts)

        st.markdown("### Score history")
        chart_data = progress_df.pivot_table(
            index="Date",
            columns="Topic",
            values="Score",
            aggfunc="mean"
        ).sort_index()

        st.line_chart(chart_data, use_container_width=True)

        st.markdown("### Attempt records")
        display_df = progress_df.copy()
        display_df["Date"] = display_df["Date"].dt.strftime(
            "%d %b %Y, %H:%M"
        )
        st.dataframe(
            display_df.sort_values("Date", ascending=False),
            use_container_width=True,
            hide_index=True
        )
