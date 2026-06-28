
import sqlite3
import os

base_path = r'c:\Users\Ganesh Biradar\Downloads\O-SCAN-master\O-SCAN-master'
db_path = os.path.join(base_path, 'instance', 'oral_cancer.db')

timestamps_to_delete = ['20260420_231509', '20260420_232959']

if not os.path.exists(db_path):
    print(f"Database not found at {db_path}")
else:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    for ts in timestamps_to_delete:
        print(f"\nProcessing timestamp: {ts}")
        
        # Get image paths and pdf path before deleting from DB
        cursor.execute("SELECT image_path, pdf_path FROM patient_record WHERE timestamp = ?", (ts,))
        record = cursor.fetchone()
        
        if record:
            image_paths_str, pdf_path = record
            
            # Delete images
            if image_paths_str:
                for img_rel_path in image_paths_str.split(','):
                    img_abs_path = os.path.join(base_path, img_rel_path.strip().replace('/', os.sep).replace('\\', os.sep))
                    if os.path.exists(img_abs_path):
                        try:
                            os.remove(img_abs_path)
                            print(f"Deleted image: {img_abs_path}")
                        except Exception as e:
                            print(f"Error deleting image {img_abs_path}: {e}")
                    else:
                        print(f"Image not found: {img_abs_path}")
            
            # Delete PDF
            if pdf_path:
                pdf_abs_path = os.path.join(base_path, pdf_path.strip().replace('/', os.sep).replace('\\', os.sep))
                if os.path.exists(pdf_abs_path):
                    try:
                        os.remove(pdf_abs_path)
                        print(f"Deleted PDF: {pdf_abs_path}")
                    except Exception as e:
                        print(f"Error deleting PDF {pdf_abs_path}: {e}")
                else:
                    # Try checking static folder directly if path was partial
                    static_pdf_path = os.path.join(base_path, 'static', f'report_{ts}.pdf')
                    if os.path.exists(static_pdf_path):
                        os.remove(static_pdf_path)
                        print(f"Deleted PDF (fallback): {static_pdf_path}")
                    else:
                        print(f"PDF not found: {pdf_abs_path}")
            
            # Delete from DB
            cursor.execute("DELETE FROM patient_record WHERE timestamp = ?", (ts,))
            print(f"Deleted DB record for timestamp {ts}")
        else:
            print(f"No record found for timestamp {ts}")
            
    conn.commit()
    conn.close()
    print("\nCleanup complete.")
