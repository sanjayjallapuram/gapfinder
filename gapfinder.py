import os
import json
import sqlite3
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from dotenv import load_dotenv
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# =========================================================
# 1. CONFIGURATION
# =========================================================

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"
DB_PATH = DATA_DIR / "gapfinder.db"
TOPICS_PATH = DATA_DIR / "topics.json"
REPORT_PATH = DATA_DIR / "latest_report.json"

MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# LangChain embedding model and persistent Chroma vector store.
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError(
        "GROQ_API_KEY missing. Configure it in .env"
    )

client = Groq(api_key=API_KEY)

for directory in [DATA_DIR, UPLOAD_DIR, CHROMA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

vector_store = None
retriever = None


# =========================================================
# 2. DATABASE
# =========================================================

def initialize_database():

    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attempt_date TEXT,
            topic TEXT,
            score REAL,
            total_questions INTEGER
        )
    """)

    conn.commit()
    conn.close()


def save_attempt(results):

    conn = sqlite3.connect(DB_PATH)

    for topic, data in results.items():

        conn.execute("""
            INSERT INTO attempts
            (attempt_date, topic, score, total_questions)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            topic,
            data["score"],
            data["total"]
        ))

    conn.commit()
    conn.close()


def show_progress():

    conn = sqlite3.connect(DB_PATH)

    rows = conn.execute("""
        SELECT topic, AVG(score), COUNT(*)
        FROM attempts
        GROUP BY topic
        ORDER BY topic
    """).fetchall()

    conn.close()

    print("\n========== LEARNING PROGRESS ==========")

    if not rows:
        print("No quiz attempts found.")
        return

    for topic, score, attempts in rows:

        print(
            f"{topic}: {score:.1f}% "
            f"| Attempts: {attempts}"
        )


# =========================================================
# 3. LLM HELPER
# =========================================================

def ask_llm(prompt):

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are GapFinder AI, an educational "
                    "assessment assistant. Follow the user's "
                    "instructions strictly. Return valid JSON "
                    "when JSON is requested. Never invent "
                    "information beyond supplied context."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=4000
    )

    return response.choices[0].message.content


def parse_json(response):

    response = response.strip()

    if response.startswith("```"):
        response = response.split("\n", 1)[1]
        response = response.rsplit("```", 1)[0]

    return json.loads(response)


# =========================================================
# 4. PDF PROCESSING
# =========================================================

def process_documents(folder):
    """
    Load PDFs page-by-page with PyPDFLoader, preserve source metadata,
    and split pages into overlapping chunks with RecursiveCharacterTextSplitter.
    """
    print("\nReading PDF documents with PyPDFLoader...")

    pdf_files = sorted(Path(folder).glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {folder}")

    loaded_documents = []

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))
        page_documents = loader.load()

        for document in page_documents:
            # PyPDFLoader stores page indexes starting at zero.
            zero_based_page = document.metadata.get("page", 0)

            document.metadata["source"] = pdf_file.name
            document.metadata["page_number"] = int(zero_based_page) + 1
            document.metadata["document_type"] = classify_document(
                pdf_file.name
            )
            loaded_documents.append(document)

    if not loaded_documents:
        raise ValueError(
            "No readable text found. The PDFs may be scanned/image-only."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(loaded_documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    print(f"Pages loaded: {len(loaded_documents)}")
    print(f"Chunks created: {len(chunks)}")
    return chunks


def classify_document(filename):

    name = filename.lower()

    if "syllabus" in name:
        return "syllabus"

    if "paper" in name or "exam" in name or "question" in name:
        return "past_paper"

    return "study_material"


# =========================================================
# 5. VECTOR DATABASE
# =========================================================

def create_vector_store(chunks):
    """
    Create a persistent LangChain Chroma vector store and index chunks.
    Replaces the existing collection so ingestion can be repeated cleanly.
    """
    global vector_store, retriever

    print("\nCreating Chroma vector store...")

    try:
        old_store = Chroma(
            collection_name="gapfinder",
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR)
        )
        old_store.delete_collection()
    except Exception:
        pass

    vector_store = Chroma(
        collection_name="gapfinder",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    batch_size = 100
    for start_index in range(0, len(chunks), batch_size):
        batch = chunks[start_index:start_index + batch_size]
        ids = [
            f"gapfinder_chunk_{start_index + offset}"
            for offset in range(len(batch))
        ]
        vector_store.add_documents(documents=batch, ids=ids)

    # MMR improves result diversity by reducing near-duplicate chunks.
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 15,
            "lambda_mult": 0.7
        }
    )

    print(f"Vector store created. Indexed {len(chunks)} chunks.")


def get_retriever():
    """Load the persistent vector store and return its MMR retriever."""
    global vector_store, retriever

    if retriever is None:
        vector_store = Chroma(
            collection_name="gapfinder",
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR)
        )

        if vector_store._collection.count() == 0:
            raise ValueError(
                "Vector database is empty. Process PDF materials first."
            )

        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 15,
                "lambda_mult": 0.7
            }
        )

    return retriever



# =========================================================
# 6. RAG RETRIEVAL
# =========================================================

def retrieve_context(query, top_k=5):
    """
    Retrieve relevant Document chunks through LangChain's MMR retriever.
    """
    active_retriever = get_retriever()

    # Allow callers to request a different number of chunks.
    if top_k != 5:
        active_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": top_k,
                "fetch_k": max(top_k * 3, 10),
                "lambda_mult": 0.7
            }
        )

    documents = active_retriever.invoke(query)
    retrieved = []

    for index, document in enumerate(documents):
        metadata = document.metadata
        retrieved.append({
            "id": f"S{index + 1}",
            "text": document.page_content,
            "source": metadata.get("source", "Unknown source"),
            "page": metadata.get(
                "page_number",
                int(metadata.get("page", 0)) + 1
            )
        })

    return retrieved


def format_context(documents):
    context = ""

    for doc in documents:
        context += (
            f"\n[SOURCE_ID: {doc['id']}]\n"
            f"File: {doc['source']}\n"
            f"Page: {doc['page']}\n"
            f"Content: {doc['text']}\n"
        )

    return context


# =========================================================
# 7. TOPIC MAPPING
# =========================================================

def build_topic_map(chunks):

    print("\nBuilding topic map...")

    syllabus_text = "\n".join(
        item.page_content
        for item in chunks
        if item.metadata.get("document_type") == "syllabus"
    )

    if not syllabus_text.strip():

        syllabus_text = "\n".join(
            item.page_content for item in chunks
        )

        print(
            "No syllabus-named PDF found. "
            "Inferring topics from available documents."
        )

    past_paper_text = "\n".join(
        item.page_content
        for item in chunks
        if item.metadata.get("document_type") == "past_paper"
    )

    prompt = f"""
    Create a topic map for a student's study materials.

    SYLLABUS:
    {syllabus_text[:10000]}

    PAST PAPERS:
    {past_paper_text[:10000]}

    Extract 4 to 12 important topics and their subtopics.

    Estimate exam_weight from 1 to 5:
    5 = frequently appearing in past papers.
    1 = low or unknown exam importance.

    Do not invent syllabus topics.
    If past papers are unavailable, use exam_weight 3.

    Return JSON in this format:

    {{
        "topics": [
            {{
                "name": "Topic name",
                "subtopics": ["Subtopic 1", "Subtopic 2"],
                "exam_weight": 3
            }}
        ]
    }}
    """

    response = ask_llm(prompt)

    topic_data = parse_json(response)

    topics = topic_data.get("topics", [])

    if not topics:
        raise ValueError("No topics extracted.")

    with open(TOPICS_PATH, "w", encoding="utf-8") as file:
        json.dump(topics, file, indent=4)

    print("\nExtracted Topics:")

    for topic in topics:
        print(
            f"- {topic['name']} "
            f"(Exam importance: {topic.get('exam_weight', 3)}/5)"
        )

    return topics


def load_topics():

    if not TOPICS_PATH.exists():
        raise FileNotFoundError(
            "Topic map missing. Run document ingestion first."
        )

    with open(TOPICS_PATH, encoding="utf-8") as file:
        return json.load(file)


# =========================================================
# 8. QUESTION GENERATION USING RAG
# =========================================================

def generate_questions(topic, number_of_questions=2):

    print(f"\nGenerating questions on topic: {topic['name']}")

    query = (
        topic["name"] + " " +
        " ".join(topic.get("subtopics", []))
    )

    retrieved_docs = retrieve_context(query, top_k=5)

    context = format_context(retrieved_docs)

    prompt = f"""
    Generate {number_of_questions} multiple-choice questions.

    TOPIC:
    {topic['name']}

    SUBTOPICS:
    {topic.get('subtopics', [])}

    RETRIEVED STUDY MATERIAL:
    {context}

    RULES:
    1. Questions must be based ONLY on retrieved material.
    2. Do not use external knowledge.
    3. Every question must have four options.
    4. Correct answer must be A, B, C or D.
    5. Include an explanation grounded in the material.
    6. Each question must specify a valid SOURCE_ID
       from the retrieved context above.
    7. If there is insufficient information, generate
       fewer questions rather than inventing facts.

    Return valid JSON:

    {{
        "questions": [
            {{
                "question": "Question text",
                "options": {{
                    "A": "Option A",
                    "B": "Option B",
                    "C": "Option C",
                    "D": "Option D"
                }},
                "correct_answer": "A",
                "explanation": "Explanation",
                "source_id": "S1"
            }}
        ]
    }}
    """

    response = ask_llm(prompt)

    question_data = parse_json(response)

    questions = []

    source_map = {
        doc["id"]: doc for doc in retrieved_docs
    }

    for question in question_data.get("questions", []):

        source = source_map.get(
            question.get("source_id")
        )

        if not source:
            continue

        answer = question.get("correct_answer", "").upper()

        options = question.get("options", {})

        if answer not in ["A", "B", "C", "D"]:
            continue

        if not all(
            key in options for key in ["A", "B", "C", "D"]
        ):
            continue

        question["topic"] = topic["name"]

        # Citation comes from actual retrieved metadata,
        # not from an unverified LLM-generated filename.

        question["citation"] = (
            f"{source['source']} | Page {source['page']}"
        )

        questions.append(question)

    return questions


# =========================================================
# 9. DIAGNOSTIC QUIZ
# =========================================================

def conduct_quiz():

    topics = load_topics()

    all_questions = []

    print("\n========== DIAGNOSTIC QUIZ ==========")

    for topic in topics:

        questions = generate_questions(
            topic,
            number_of_questions=2
        )

        all_questions.extend(questions)

    if not all_questions:
        raise ValueError(
            "No questions generated. Check your documents."
        )

    print(
        f"\nTotal questions generated: {len(all_questions)}"
    )

    print("\nAnswer the following questions:\n")

    results = defaultdict(
        lambda: {"correct": 0, "total": 0, "mistakes": []}
    )

    for index, question in enumerate(all_questions, start=1):

        topic = question["topic"]

        print(f"\nQuestion {index}: {question['question']}")

        for key, value in question["options"].items():
            print(f"{key}. {value}")

        print(f"Source: {question['citation']}")

        answer = input(
            "\nYour answer (A/B/C/D): "
        ).strip().upper()

        while answer not in ["A", "B", "C", "D"]:

            answer = input(
                "Enter A, B, C or D: "
            ).strip().upper()

        correct_answer = question["correct_answer"]

        results[topic]["total"] += 1

        if answer == correct_answer:

            results[topic]["correct"] += 1

            print("\nCorrect answer!")

        else:

            results[topic]["mistakes"].append(
                question["question"]
            )

            print("\nIncorrect answer.")

        print(
            f"Correct answer: {correct_answer}"
        )

        print(
            f"Explanation: {question['explanation']}"
        )

    # Calculate topic-level scores.

    topic_scores = {}

    for topic, data in results.items():

        score = (
            data["correct"] / data["total"]
        ) * 100

        topic_scores[topic] = {
            "score": score,
            "correct": data["correct"],
            "total": data["total"],
            "mistakes": data["mistakes"]
        }

    save_attempt(topic_scores)

    return topic_scores


# =========================================================
# 10. GAP ANALYSIS
# =========================================================

def analyze_gaps(topic_scores):

    topics = load_topics()

    topic_metadata = {
        topic["name"]: topic for topic in topics
    }

    report = {}

    print("\n========== GAP ANALYSIS REPORT ==========")

    for topic, data in topic_scores.items():

        score = data["score"]

        if score < 50:
            status = "WEAK"

        elif score < 75:
            status = "MEDIUM"

        else:
            status = "STRONG"

        exam_weight = topic_metadata.get(
            topic, {}
        ).get("exam_weight", 3)

        priority = (
            0.7 * (100 - score) +
            0.3 * (exam_weight * 20)
        )

        report[topic] = {
            **data,
            "status": status,
            "exam_weight": exam_weight,
            "priority": priority
        }

        print(f"\nTopic: {topic}")
        print(f"Score: {score:.1f}%")
        print(f"Status: {status}")
        print(f"Exam importance: {exam_weight}/5")

        if data["mistakes"]:

            print("Reason for weakness:")

            for mistake in data["mistakes"]:
                print(f"- Incorrect response: {mistake}")

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    return report


def load_report():
    if not REPORT_PATH.exists():
        return None
    with open(REPORT_PATH, encoding="utf-8") as file:
        return json.load(file)


# =========================================================
# 11. PERSONALIZED REVISION PLAN
# =========================================================

def generate_revision_plan(report, days, hours_per_day):

    total_hours = days * hours_per_day

    ranked_topics = sorted(
        report.items(),
        key=lambda item: item[1]["priority"],
        reverse=True
    )

    # Allocate time proportionally to priority.
    weights = [
        max(data["priority"], 1)
        for _, data in ranked_topics
    ]

    total_weight = sum(weights)

    allocations = [
        total_hours * weight / total_weight
        for weight in weights
    ]

    print("\n========== PERSONALIZED REVISION PLAN ==========")

    print(f"Available days: {days}")
    print(f"Hours per day: {hours_per_day}")
    print(f"Total study hours: {total_hours}")

    daily_plan = defaultdict(list)

    for (topic, data), allocated_hours in zip(
        ranked_topics, allocations
    ):

        remaining = allocated_hours

        for day in range(1, days + 1):

            if remaining <= 0:
                break

            already_used = sum(
                item["hours"]
                for item in daily_plan[day]
            )

            available = max(
                0,
                hours_per_day - already_used
            )

            if available <= 0:
                continue

            assigned = min(
                remaining,
                available
            )

            daily_plan[day].append({
                "topic": topic,
                "hours": assigned,
                "status": data["status"]
            })

            remaining -= assigned

    for day in range(1, days + 1):

        print(f"\nDAY {day}")

        if not daily_plan[day]:
            print("Revision and practice buffer")
            continue

        for item in daily_plan[day]:

            print(
                f"- {item['topic']}: "
                f"{item['hours']:.2f} hours "
                f"({item['status']})"
            )

    return daily_plan


# =========================================================
# 12. EXPLAIN TOPIC - RAG TUTOR
# =========================================================

def explain_topic():

    topic = input(
        "\nEnter topic to explain: "
    ).strip()

    if not topic:
        print("Topic cannot be empty.")
        return

    docs = retrieve_context(topic, top_k=5)

    context = format_context(docs)

    prompt = f"""
    Explain the requested topic using ONLY the retrieved
    study material.

    TOPIC:
    {topic}

    RETRIEVED CONTEXT:
    {context}

    Instructions:
    - Explain in simple language.
    - Include important concepts.
    - Use examples only if supported by context.
    - Do not use external knowledge.
    - If information is insufficient, say so.
    - Cite sources using their source IDs.
    """

    print("\nGenerating explanation...\n")

    print(ask_llm(prompt))

    print("\nSources:")

    for doc in docs:
        print(
            f"- {doc['source']} | Page {doc['page']}"
        )


# =========================================================
# 13. DOCUMENT INGESTION PIPELINE
# =========================================================

def ingest_documents():

    chunks = process_documents(UPLOAD_DIR)

    if not chunks:
        raise ValueError(
            "No readable text found in uploaded PDFs."
        )

    create_vector_store(chunks)

    build_topic_map(chunks)

    print("\nDocument ingestion completed!")


def _delete_vector_collection():
    """Delete the persistent GapFinder collection and clear cached handles."""
    global vector_store, retriever

    try:
        store = Chroma(
            collection_name="gapfinder",
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR)
        )
        store.delete_collection()
    except Exception:
        # The collection may not exist yet. The on-disk directory is handled
        # by Chroma; do not delete it while a collection may be open.
        pass

    vector_store = None
    retriever = None


def _remove_derived_knowledge_files():
    """Remove topic mapping and latest diagnosis tied to the old documents."""
    for path in (TOPICS_PATH, REPORT_PATH):
        if path.exists():
            path.unlink()


def clear_uploaded_documents():
    """Remove every uploaded PDF and all derived RAG/topic/report data.

    Quiz-attempt history in SQLite is intentionally preserved.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for pdf_file in UPLOAD_DIR.glob("*.pdf"):
        pdf_file.unlink()

    _delete_vector_collection()
    _remove_derived_knowledge_files()


def remove_document(filename):
    """Remove one PDF, then rebuild the index and topic map from remaining PDFs."""
    safe_name = Path(filename).name
    if safe_name != filename or not safe_name.lower().endswith(".pdf"):
        raise ValueError("Invalid PDF filename.")

    target = UPLOAD_DIR / safe_name
    if not target.is_file():
        raise FileNotFoundError(f"Document not found: {safe_name}")

    target.unlink()
    remaining = list(UPLOAD_DIR.glob("*.pdf"))

    if remaining:
        # ingest_documents recreates the Chroma collection from only the
        # remaining files and regenerates the topic map.
        ingest_documents()
    else:
        _delete_vector_collection()
        _remove_derived_knowledge_files()

    return [path.name for path in sorted(UPLOAD_DIR.glob("*.pdf"))]


# =========================================================
# 14. APPLICATION MENU
# =========================================================

def main():

    initialize_database()

    while True:

        print("\n")
        print("=" * 50)
        print("          GAPFINDER AI")
        print("   Student Knowledge Gap Detection System")
        print("=" * 50)

        print("\n1. Upload / Process PDF Materials")
        print("2. Start Diagnostic Quiz")
        print("3. View Gap Analysis")
        print("4. Generate Revision Plan")
        print("5. Explain a Topic (RAG)")
        print("6. View Learning Progress")
        print("7. Exit")

        choice = input("\nChoose an option: ").strip()

        try:

            if choice == "1":

                ingest_documents()

            elif choice == "2":

                scores = conduct_quiz()

                report = analyze_gaps(scores)

                # Automatically generate revision plan.

                days = int(
                    input("\nHow many days until your exam? ")
                )

                hours = float(
                    input("Available study hours per day: ")
                )

                generate_revision_plan(
                    report,
                    days,
                    hours
                )

            elif choice == "3":

                if not REPORT_PATH.exists():
                    print("Take a diagnostic quiz first.")
                else:
                    with open(REPORT_PATH, encoding="utf-8") as file:
                        report = json.load(file)

                    print("\n========== LATEST GAP REPORT ==========")

                    for topic, data in report.items():
                        print(
                            f"\n{topic}: {data['score']:.1f}% "
                            f"| {data['status']}"
                        )

            elif choice == "4":
            
                if not REPORT_PATH.exists():
                    print("Take a diagnostic quiz first.")
                else:
                    with open(REPORT_PATH, encoding="utf-8") as file:
                        report = json.load(file)

                    days = int(input("Days until exam: "))
                    hours = float(input("Study hours per day: "))

                    if days <= 0 or hours <= 0:
                        print("Days and hours must be positive.")
                    else:
                        generate_revision_plan(
                            report,
                            days,
                            hours
                        )

            elif choice == "5":

                explain_topic()

            elif choice == "6":

                show_progress()

            elif choice == "7":

                print("\nThank you for using GapFinder AI!")
                break

            else:

                print("Invalid choice.")

        except Exception as error:

            print(f"\nError: {error}")
            print("Check your configuration and try again.")


if __name__ == "__main__":
    main()
