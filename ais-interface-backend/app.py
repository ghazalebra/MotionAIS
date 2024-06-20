from flask import Flask, request, jsonify, send_file
# from flask_socketio import SocketIO, emit
# import eventlet
from flask_cors import CORS
# from flask_sse import sse
import os
import json
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import *

# eventlet.monkey_patch()

app = Flask(__name__)

# Access-Control-Allow-Origin: http://localhost:3000

DATA_PATH = 'data'
os.makedirs(DATA_PATH, exist_ok=True)
app.config['DATA_PATH'] = DATA_PATH

# sse config for passing progress to the frontend
# app.config["REDIS_URL"] = "redis://localhost:6379/0"
# app.register_blueprint(sse, url_prefix='/stream')

# app.config['SECRET_KEY'] = 'secret!'
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow requests from localhost:3000
# socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# @app.route('/start_loop', methods=['POST', 'GET'])
# def start_loop():
#     total_iterations = 10
#     for i in range(total_iterations):
#         time.sleep(1)  # Simulate a time-consuming task
#         progress = (i + 1) / total_iterations * 100
#         socketio.emit('progress', {'progress': progress})
#     return jsonify({"message": "Loop completed"}), 200

# @socketio.on('connect')
# def handle_connect():
#     print('Client connected')


# @app.route('/', methods=['POST'])
# def update_progress():
#     for i in range(10):
#         emit('progress', {'progress': i/10})

@app.route('/upload', methods=['POST'])
def upload_file():

    sequence_name = request.form.get('sequence_name')

    if sequence_name is None:
        sequence_name = 'random'
    
    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400
    
    files = request.files.getlist('files')
    saved_files = []

    sequence_upload_path = app.config['DATA_PATH']+ '/' + sequence_name + '/uploads/'
    os.makedirs(sequence_upload_path, exist_ok=True)

    for file in files:
        if file.filename == '':
            return jsonify({"error": "One or more files have no filename"}), 400
        file_path = os.path.join(sequence_upload_path, file.filename)
        file.save(file_path)
        saved_files.append(file.filename)

    return jsonify({"message": "Files successfully uploaded", "files": saved_files}), 200

# @socketio.on('connect')
# def publish_progress(step, total_steps):
#     progress = (step + 1) / total_steps * 100
#     print('progress ' + str(progress))
#     socketio.emit('progress', {'progress': progress}, broadcast=True)

# @socketio.on('connect')
# def handle_connect():
#     print('Client connected')
#     emit('progress', {'progress': 40})  # Emit a test progress message

@app.route('/sequence', methods=['GET'])
def get_sequence(find_landmarks=False, calculate_metrics_=False):

    # Load sequence
    sequence_name = request.args.get('sequenceName')
    if not sequence_name:
        return jsonify({"error": "No sequence name provided"}), 400

    sequence = []
    sequence_path = os.path.join(DATA_PATH, sequence_name)

    frames = sorted(os.listdir(sequence_path + "/uploads/"))
    for frame in frames:
        frame_path = sequence_path + "/uploads/" + str(frame)
        with open(frame_path) as frame_file:
            sequence.append(json.load(frame_file))

    # Find landmarks
    outpath = sequence_path + "/results/"
    os.makedirs(outpath, exist_ok=True)
    if len(os.listdir(outpath)):
        outputs = []
        for i, frame in enumerate(frames):
            # publish_progress(i, len(frames))
            frame_path = outpath + str(frame)
            with open(frame_path) as frame_file:
                outputs.append(json.load(frame_file))
    else:
        # socketio.emit('progress', {'data': 'checking'})
        # outputs = find_segments_sequence(sequence=sequence, publish_progress=publish_progress)
        outputs = find_segments_sequence(sequence=sequence)
        for output, frame in zip(outputs, frames):
            with open(outpath + f'{frame[:-5]}.json', 'w') as f:
                json.dump(output, f)

    # Motion analysis
    # metrics_path = sequence_path + "/metrics/"
    # os.makedirs(metrics_path, exist_ok=True)
    # if len(os.listdir(metrics_path)) or calculate_metrics_:
    #     with open(metrics_path+'metrics.json', 'w') as metrics_file:
    #         metrics = json.load(metrics_file)
    # else:    
    #     metrics = calculate_metrics(outputs)

    return jsonify({"sequence": sequence, "results": outputs}), 200

@app.route('/analyze')
def get_plots(new=False):
    # Load sequence
    sequence_name = request.args.get('sequenceName')
    if not sequence_name:
        return jsonify({"error": "No sequence name provided"}), 400

    sequence = []
    sequence_path = os.path.join(DATA_PATH, sequence_name)

    metrics_path = sequence_path + "/metrics/"

    if os.path.isdir(metrics_path) and os.path.isfile(metrics_path + "metrics.json") and not new:
        with open(metrics_path+'metrics.json', 'r') as metrics_file:
            metrics = json.load(metrics_file)
    else:
        outpath = sequence_path + "/results/"
        if os.path.isdir(outpath) and len(os.listdir(outpath)):
            frames = sorted(os.listdir(outpath))
            outputs = []
            for i, frame in enumerate(frames):
                # publish_progress(i, len(frames))
                frame_path = outpath + str(frame)
                with open(frame_path) as frame_file:
                    outputs.append(json.load(frame_file))
            metrics = calculate_metrics(outputs)
            os.makedirs(metrics_path, exist_ok=True)
            with open(metrics_path + 'metrics.json', 'w') as metrics_file:
                json.dump(metrics, metrics_file)
        else: 
            return jsonify({"error": "Landmarks not found"}), 400
    
    img_plots = plot_metrics(metrics, metrics_names=['Scoliosis Angle'])
    
    return send_file(img_plots, mimetype='image/png')

    # return jsonify({"sequence": sequence, "results": outputs}), 200




@app.route('/sequencelist', methods=['GET'])
def get_sequence_list():
    base_path = request.args.get('path', '')
    full_path = os.path.join(app.config['DATA_PATH'], base_path)
    if not os.path.exists(full_path) or not os.path.isdir(full_path):
        return jsonify({"error": "Invalid path"}), 400

    sequencelist = [d for d in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, d))]
    return jsonify({"sequencelist": sequencelist}), 200


if __name__ == '__main__':
    app.run(debug=True)
    # socketio.run(app, debug=True)
