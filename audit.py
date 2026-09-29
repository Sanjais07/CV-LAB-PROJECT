import os
import datetime
import pandas as pd
import json

root_folder = r"D:\9 Sem\CV Proj 2\CSV_MINERS"
report_path = r"D:\9 Sem\CV Proj 2\audit_report.txt"

def get_file_magic(path):
    # a basic approximation of the file type since python-magic might not be installed
    try:
        with open(path, 'rb') as f:
            header = f.read(2048)
        if b'\0' in header:
            return "Binary data"
        else:
            return "Text data"
    except Exception as e:
        return f"Error reading: {e}"

with open(report_path, "w", encoding="utf-8") as out:
    out.write("STEP 1 - Full recursive folder listing\n")
    out.write("="*40 + "\n")
    
    candidates = []
    non_data = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for f in filenames:
            full_path = os.path.join(dirpath, f)
            rel_path = os.path.relpath(full_path, root_folder)
            
            try:
                st = os.stat(full_path)
                size = st.st_size
                mtime = datetime.datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            except Exception as e:
                out.write(f"[{rel_path}] ERROR reading stats: {e}\n")
                continue
                
            file_type = get_file_magic(full_path)
            out.write(f"Path: {rel_path}\n")
            out.write(f"Size: {size} bytes\n")
            out.write(f"Modified: {mtime}\n")
            out.write(f"Type: {file_type}\n")
            
            is_mac = "__MACOSX" in rel_path or f.startswith("._") or f.startswith("__")
            if is_mac or size < 5120:
                out.write("Flag: Possible AppleDouble/Mac shadow or non-data file (<5KB).\n")
                non_data.append((rel_path, full_path, size))
            else:
                out.write("Flag: Plausible dataset file.\n")
                candidates.append((rel_path, full_path, size))
            out.write("-" * 20 + "\n")
            
    out.write("\nSTEP 2 & 3 - Identify and Audit Real Dataset\n")
    out.write("="*40 + "\n")
    
    for rel_path, full_path, size in candidates:
        out.write(f"\nCandidate: {rel_path} (Size: {size} bytes)\n")
        try:
            df = pd.read_csv(full_path, low_memory=False)
            out.write(f"Status: Successfully opened via pd.read_csv()\n")
            out.write(f"Columns: {list(df.columns)}\n")
            out.write(f"First 5 rows:\n{df.head().to_string()}\n")
            
            out.write(f"\nStep 3 Audit:\n")
            out.write(f"Exact column dtypes:\n{df.dtypes.to_string()}\n")
            out.write(f"Total row count (len(df)): {len(df)}\n")
            out.write(f"Null/missing value count per column:\n{df.isnull().sum().to_string()}\n")
            
            # Numeric columns
            numeric_cols = df.select_dtypes(include='number').columns
            if len(numeric_cols) > 0:
                out.write(f"Basic summary stats:\n{df[numeric_cols].describe().to_string()}\n")
                
            # Class balance (checking for anything that looks like label/coin)
            for col in df.columns:
                if 'label' in col.lower() or 'coin' in col.lower() or 'class' in col.lower():
                    out.write(f"\nValue counts for '{col}':\n{df[col].value_counts().to_string()}\n")
                    
            # Check for 5-tuple
            out.write("\nFlow-identifying fields (5-tuple):\n")
            tuple_fields = []
            for col in df.columns:
                cl = col.lower()
                if 'ip' in cl or 'port' in cl or 'proto' in cl:
                    tuple_fields.append(col)
            out.write(f"Found apparent 5-tuple fields: {tuple_fields}\n")
            
            # Check for timestamp/duration
            out.write("\nTimestamp/duration fields:\n")
            time_fields = []
            for col in df.columns:
                cl = col.lower()
                if 'time' in cl or 'duration' in cl or 'date' in cl:
                    time_fields.append(col)
            out.write(f"Found apparent time/duration fields: {time_fields}\n")
            
            # Step 4 check
            out.write("\nSTEP 4 - Cross-check against paper\n")
            total_rows = len(df)
            out.write(f"Paper claimed 84,344 total flows. Actual: {total_rows}\n")
            
        except Exception as e:
            out.write(f"Error opening as CSV: {e}\n")

    out.write("\nSTEP 5 - Non-data files\n")
    out.write("="*40 + "\n")
    for rel_path, full_path, size in non_data:
        out.write(f"\nNon-dataset file: {rel_path} (Size: {size})\n")
        try:
            with open(full_path, 'rb') as f:
                content = f.read(100)
            out.write(f"First 100 bytes: {content}\n")
        except Exception as e:
            out.write(f"Error reading: {e}\n")
