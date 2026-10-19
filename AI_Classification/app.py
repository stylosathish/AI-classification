import os
import threading
from flask import Flask, request, render_template, send_file, jsonify
import pandas as pd
from google import genai
from google.genai import types
import json
import time
import difflib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

API_KEY = "Gemini API key"

# Global dictionary to track progress
progress_state = {
    'status': 'idle', # idle, running, completed, error
    'current_row': 0,
    'total_rows': 0,
    'message': '',
    'output_file': ''
}

def process_file_background(input_path, tax_path, output_path):
    global progress_state
    progress_state['status'] = 'running'
    progress_state['message'] = 'Initializing AI client...'
    progress_state['current_row'] = 0
    progress_state['total_rows'] = 0
    
    try:
        client = genai.Client(api_key=API_KEY)
        MODEL_ID = 'gemini-flash-lite-latest' 
        
        progress_state['message'] = 'Loading Taxonomy...'
        tax_df = pd.read_excel(tax_path)
        valid_categories = tax_df['Category_Name'].dropna().unique().tolist()
        
        progress_state['message'] = 'Loading Input File...'
        df = pd.read_excel(input_path)
        
        if 'Correct_Taxonomy' not in df.columns:
            df['Correct_Taxonomy'] = None
            
        total_rows = len(df)
        progress_state['total_rows'] = total_rows
        
        BATCH_SIZE = 30
        idx = 0
        
        while idx < total_rows:
            end_idx = min(idx + BATCH_SIZE, total_rows)
            progress_state['current_row'] = idx
            progress_state['message'] = f'Processing rows {idx+1} to {end_idx}...'
            
            batch_data = []
            for i in range(idx, end_idx):
                item_data = {'id': i}
                for col in df.columns:
                    if col not in ['Category_Name', 'Correct_Taxonomy', 'Error', 'Allocation', 'Batch ID']:
                        if pd.notna(df.at[i, col]):
                            val = str(df.at[i, col]).strip()
                            if val != "":
                                item_data[col] = val
                batch_data.append(item_data)
                
            prompt = f"""
You are an expert product categorizer.
CRITICAL RULES:
1. Prioritize the core item type over the material.
2. You are ENCOURAGED to use your expert knowledge to categorize items based on Brand and Part Number. Make your best educated match!
3. ONLY return "UNKNOWN" if completely blank or impossible to determine.

VALID TAXONOMY CATEGORIES:
{json.dumps(valid_categories)}

PRODUCT DATA:
{json.dumps(batch_data, indent=2)}

Return JSON array of objects with "id" and "correct_taxonomy" (must be exact match or "UNKNOWN").
"""
            try:
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                results = json.loads(response.text.strip())
                
                for res in results:
                    row_id = res.get('id')
                    if row_id is not None and row_id >= idx and row_id < end_idx:
                        ai_cat = res.get('correct_taxonomy', 'ERROR')
                        
                        # Validation logic
                        if ai_cat not in ['UNKNOWN', 'ERROR'] and ai_cat not in valid_categories:
                            substring_match = None
                            for vc in valid_categories:
                                if ai_cat.lower() in str(vc).lower():
                                    substring_match = vc
                                    break
                            if substring_match:
                                ai_cat = substring_match
                            else:
                                closest = difflib.get_close_matches(ai_cat, valid_categories, n=1, cutoff=0.75)
                                if closest:
                                    ai_cat = closest[0]
                                else:
                                    ai_cat = f"UNKNOWN (AI guessed: {ai_cat})"
                                    
                        df.at[row_id, 'Correct_Taxonomy'] = ai_cat
            except Exception as e:
                progress_state['message'] = f'Rate limit hit. Waiting 65s...'
                time.sleep(65)
                continue
                
            # Fill skips
            for i in range(idx, end_idx):
                if pd.isna(df.at[i, 'Correct_Taxonomy']):
                    df.at[i, 'Correct_Taxonomy'] = "SKIP"
                    
            df.to_excel(output_path, index=False)
            
            time.sleep(10)
            idx += BATCH_SIZE
            
        progress_state['current_row'] = total_rows
        progress_state['message'] = 'Classification Complete! You can now download the file.'
        progress_state['status'] = 'completed'
        progress_state['output_file'] = output_path
        
    except Exception as e:
        progress_state['status'] = 'error'
        progress_state['message'] = str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global progress_state
    if progress_state['status'] == 'running':
        return jsonify({'error': 'A job is already running! Please wait for it to finish.'}), 400
        
    if 'input_file' not in request.files or 'tax_file' not in request.files:
        return jsonify({'error': 'Missing files'}), 400
        
    input_file = request.files['input_file']
    tax_file = request.files['tax_file']
    
    in_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.xlsx')
    tax_path = os.path.join(app.config['UPLOAD_FOLDER'], 'taxonomy.xlsx')
    out_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.xlsx')
    
    input_file.save(in_path)
    tax_file.save(tax_path)
    
    # Start the background task
    thread = threading.Thread(target=process_file_background, args=(in_path, tax_path, out_path))
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': 'Started successfully'})

@app.route('/progress')
def progress():
    return jsonify(progress_state)

@app.route('/download')
def download():
    out_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.xlsx')
    return send_file(out_path, as_attachment=True, download_name='classified_output.xlsx')

if __name__ == '__main__':
    # host='0.0.0.0' allows other computers on the same network to access the web app
    app.run(host='0.0.0.0', debug=True, port=5009)
