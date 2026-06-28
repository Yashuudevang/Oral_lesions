import sqlite3
import os

db_path = os.path.join('instance', 'oral_cancer.db')
if not os.path.exists(db_path):
    print("Database not found")
    exit()

conn = sqlite3.connect(db_path)
cur = conn.cursor()

query = """
SELECT 
    d.username AS doctor_name, 
    u.username AS patient_name, 
    p.prediction, 
    p.timestamp
FROM patient_record p
JOIN user u ON p.user_id = u.id
LEFT JOIN user d ON p.doctor_id = d.id
WHERE p.prediction LIKE '%Risk%'
"""

cur.execute(query)
results = cur.fetchall()

if not results:
    print("No cancer-risk patients found.")
else:
    print("Found cancer-risk assignments:")
    for row in results:
        doc = row[0] if row[0] else "Unassigned"
        print(f"Doctor: {doc} | Patient: {row[1]} | Result: {row[2]}")

conn.close()
