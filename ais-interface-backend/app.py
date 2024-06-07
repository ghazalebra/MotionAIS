from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import *

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

DATA_PATH = 'data'
os.makedirs(DATA_PATH, exist_ok=True)
app.config['DATA_PATH'] = DATA_PATH

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400
    
    files = request.files.getlist('files')
    saved_files = []

    os.makedirs(app.config['DATA_PATH']+'/uploads', exist_ok=True)

    for file in files:
        if file.filename == '':
            return jsonify({"error": "One or more files have no filename"}), 400
        file_path = os.path.join(app.config['DATA_PATH']+'/uploads', file.filename)
        file.save(file_path)
        saved_files.append(file.filename)

    return jsonify({"message": "Files successfully uploaded", "files": saved_files}), 200

@app.route('/sequence', methods=['GET'])
def get_sequence():
    sequence = []
    frames = os.listdir(DATA_PATH + "/uploads/")
    for frame in frames:
        frame_path = DATA_PATH + "/uploads/" + str(frame)
        with open(frame_path) as frame_file:
            sequence.append(json.load(frame_file))
    outpath = DATA_PATH + "/results/"
    os.makedirs(outpath, exist_ok=True)
    if len(os.listdir(outpath)):
        outputs = []
        for frame in frames:
            frame_path = outpath + str(frame)
            with open(frame_path) as frame_file:
                outputs.append(json.load(frame_file))
    else:
        outputs = find_segments_sequence(sequence)
        for output, frame in zip(outputs, frames):
            with open(outpath + f'{frame[:-5]}.json', 'w') as f:
                json.dump(output, f)
    return jsonify({"sequence": sequence, "results": outputs}), 200


if __name__ == '__main__':
    app.run(debug=True)
