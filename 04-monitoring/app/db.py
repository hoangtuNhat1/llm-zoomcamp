import os
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

tz = ZoneInfo("Asia/Saigon")

def get_db_connection():
    db_path = os.getenv("SQLITE_DB_PATH", "course_assistant.db")
    return sqlite3.connect(db_path)

def init_db():
    conn = get_db_connection()
    try:
        print("Initalize data...")
        with conn:
            conn.execute("DROP TABLE IF EXISTS feedback")
            conn.execute("DROP TABLE IF EXISTS conversations")
            conn.execute("""
                CREATE TABLE conversations (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    course TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    response_time REAL NOT NULL,
                    relevance TEXT NOT NULL,
                    relevance_explanation TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    eval_prompt_tokens INTEGER NOT NULL,
                    eval_completion_tokens INTEGER NOT NULL,
                    eval_total_tokens INTEGER NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT REFERENCES conversations(id),
                    feedback INTEGER NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        conn.close()

def save_conversation(conversation_id, question, answer_data, course, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now(tz).isoformat()
    
    conn = get_db_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO conversations 
                (id, question, answer, course, model_used, response_time, relevance, 
                relevance_explanation, prompt_tokens, completion_tokens, total_tokens, 
                eval_prompt_tokens, eval_completion_tokens, eval_total_tokens, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    question,
                    answer_data["answer"],
                    course,
                    answer_data["model_used"],
                    answer_data["response_time"],
                    answer_data["relevance"],
                    answer_data["relevance_explanation"],
                    answer_data["prompt_tokens"],
                    answer_data["completion_tokens"],
                    answer_data["total_tokens"],
                    answer_data["eval_prompt_tokens"],
                    answer_data["eval_completion_tokens"],
                    answer_data["eval_total_tokens"],
                    timestamp,
                ),
            )
    finally:
        conn.close()

def save_feedback(conversation_id, feedback, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now(tz).isoformat()
    conn = get_db_connection()
    try:
        with conn:
            conn.execute(
                "INSERT INTO feedback (conversation_id, feedback, timestamp) VALUES (?, ?, ?)",
                (conversation_id, feedback, timestamp),
            )
    finally:
        conn.close()

def get_recent_conversations(limit=5, relevance=None):
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        query = """
            SELECT c.*, f.feedback
            FROM conversations c
            LEFT JOIN feedback f ON c.id = f.conversation_id
        """
        if relevance:
            query += f" WHERE c.relevance = ?"
            params = (relevance, limit)
        else:
            params = (limit,)
        query += " ORDER BY c.timestamp DESC LIMIT ?"

        cur.execute(query, params)
        columns = [column[0] for column in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]
    finally:
        conn.close()

def get_feedback_stats():
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                SUM(CASE WHEN feedback > 0 THEN 1 ELSE 0 END) as thumbs_up,
                SUM(CASE WHEN feedback < 0 THEN 1 ELSE 0 END) as thumbs_down
            FROM feedback
        """)
        result = cur.fetchone()
        return {"thumbs_up": result[0] or 0, "thumbs_down": result[1] or 0}
    finally:
        conn.close()