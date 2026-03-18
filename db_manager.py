"""
Database Manager for Smart Inventory Demand Forecasting System.
Handles SQLite operations for logging predictions, SHAP values, and chat history.
"""

import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict, Any


class DBManager:
    """
    Manages SQLite database operations for the forecasting application.
    Stores prediction history, SHAP values (as JSON), and user chat logs.
    """
    
    def __init__(self, db_path: str = "project_logs.db"):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self.init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Create and return a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
        return conn
    
    def init_db(self) -> None:
        """
        Initialize the database and create the logs table if it doesn't exist.
        
        Table Schema:
            - id: Primary key (auto-increment)
            - timestamp: When the event occurred
            - item_id: The item being predicted
            - prediction: The predicted sales value
            - shap_values_json: SHAP values serialized as JSON string
            - user_query: User's chat input (if applicable)
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    item_id TEXT,
                    prediction REAL,
                    shap_values_json TEXT,
                    user_query TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            print(f"Database initialized successfully at: {self.db_path}")
            
        except sqlite3.Error as e:
            print(f"Database initialization error: {e}")
            raise
    
    def log_event(
        self,
        item_id: Optional[str] = None,
        prediction: Optional[float] = None,
        shap_values: Optional[List[float]] = None,
        user_query: Optional[str] = None
    ) -> bool:
        """
        Log a prediction or chat event to the database.
        
        Args:
            item_id: The item ID for which prediction was made.
            prediction: The predicted sales value.
            shap_values: List of SHAP values (will be converted to JSON).
            user_query: User's chat query (if this is a chat event).
            
        Returns:
            True if logging was successful, False otherwise.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Convert SHAP values list to JSON string
            shap_json = None
            if shap_values is not None:
                shap_json = json.dumps(shap_values)
            
            # Get current timestamp
            timestamp = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT INTO logs (timestamp, item_id, prediction, shap_values_json, user_query)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, item_id, prediction, shap_json, user_query))
            
            conn.commit()
            conn.close()
            return True
            
        except sqlite3.Error as e:
            print(f"Error logging event: {e}")
            return False
    
    def get_recent_logs(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the most recent log entries.
        
        Args:
            limit: Maximum number of logs to retrieve (default: 5).
            
        Returns:
            List of dictionaries containing log data.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, timestamp, item_id, prediction, shap_values_json, user_query
                FROM logs
                ORDER BY id DESC
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            # Convert rows to list of dictionaries
            logs = []
            for row in rows:
                log_entry = {
                    'id': row['id'],
                    'timestamp': row['timestamp'],
                    'item_id': row['item_id'],
                    'prediction': row['prediction'],
                    'shap_values': json.loads(row['shap_values_json']) if row['shap_values_json'] else None,
                    'user_query': row['user_query']
                }
                logs.append(log_entry)
            
            return logs
            
        except sqlite3.Error as e:
            print(f"Error retrieving logs: {e}")
            return []
    
    def get_logs_by_item(self, item_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve logs for a specific item.
        
        Args:
            item_id: The item ID to filter by.
            limit: Maximum number of logs to retrieve.
            
        Returns:
            List of dictionaries containing log data for the specified item.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, timestamp, item_id, prediction, shap_values_json, user_query
                FROM logs
                WHERE item_id = ?
                ORDER BY id DESC
                LIMIT ?
            ''', (item_id, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            logs = []
            for row in rows:
                log_entry = {
                    'id': row['id'],
                    'timestamp': row['timestamp'],
                    'item_id': row['item_id'],
                    'prediction': row['prediction'],
                    'shap_values': json.loads(row['shap_values_json']) if row['shap_values_json'] else None,
                    'user_query': row['user_query']
                }
                logs.append(log_entry)
            
            return logs
            
        except sqlite3.Error as e:
            print(f"Error retrieving logs for item {item_id}: {e}")
            return []


# Quick test when running directly
if __name__ == "__main__":
    # Initialize database
    db = DBManager()
    
    # Test logging a prediction event
    db.log_event(
        item_id="FOODS_3_001",
        prediction=15.5,
        shap_values=[0.1, -0.05, 0.2, 0.15, -0.1]
    )
    
    # Test logging a chat event
    db.log_event(user_query="What is the predicted demand for FOODS_3_001?")
    
    # Retrieve and display recent logs
    print("\nRecent logs:")
    for log in db.get_recent_logs():
        print(log)
