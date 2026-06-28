
import sqlite3
import os

db_path = r'c:\Users\Ganesh Biradar\Downloads\O-SCAN-master\O-SCAN-master\instance\oral_cancer.db'

if not os.path.exists(db_path):
    print(f"Database not found at {db_path}")
else:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables: {tables}")
    
    if ('patient_record',) in tables:
        cursor.execute("SELECT image_path, prediction, confidence, timestamp FROM patient_record WHERE prediction LIKE '%Risk%' OR prediction LIKE '%Cancer%';")
        rows = cursor.fetchall()
        print(f"Found {len(rows)} records with cancer risk.")
        for row in rows:
            print(f"Image: {row[0]}, Prediction: {row[1]}, Confidence: {row[2]}, Timestamp: {row[3]}")
    else:
        print("Table 'patient_record' not found.")
    
    conn.close()
