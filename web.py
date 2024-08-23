import os
import uuid
import datetime
from flask import Flask, render_template_string, request, send_from_directory, redirect, url_for, session
from threading import Thread
import shutil
from test import (
    load_subtitles_from_file, subriptime_to_seconds, load_video_from_file, 
    concatenate_videoclips, get_segments_using_srt, generate_srt_from_txt_and_audio,
    adjust_segment_duration,crop_to_aspect_ratio,replace_video_segments, split_by_computer_vision,
    refine_subtitles_based_on_computer_vision
    )
from pathlib import Path
import pysrt
import logging


MAE_THRESHOLD: float = 4.2
GLITCH_IGNORE_THRESHOLD: float = 0.27

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for session management
app.config['UPLOAD_FOLDER'] = 'uploads'

def generate_unique_id():
    return str(uuid.uuid4())

def generate_datetime_alias():
    current_time = datetime.datetime.now()
    return current_time.strftime("%Y-%m-%d_%H-%M-%S")

@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Scene Optimisation Bot</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f2f2f2;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                }
                .container {
                    background-color: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    width: 400px;
                    text-align: center;
                }
                h1 {
                    color: #333;
                }
                input[type="file"], input[type="text"], input[type="submit"] {
                    margin-bottom: 10px;
                    width: 100%;
                    padding: 8px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
                input[type="submit"] {
                    background-color: #4CAF50;
                    color: white;
                    cursor: pointer;
                }
                input[type="submit"]:hover {
                    background-color: #45a049;
                }
                #waitMessage {
                    margin-top: 20px;
                    color: #555;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Scene Optimisation Bot</h1>
                <form action="/process" method="post" enctype="multipart/form-data" onsubmit="displayMessage()">
                    Video File <input type="file" name="video_file" required><br>
                    MP3 File <input type="file" name="mp3_file" required><br>
                    Text File <input type="file" name="text_file" required><br>
                    TTF Font File <input type="file" name="font_file" required><br>
                    Font Size <input type="text" name="font_size" value="36" required><br>
                    Font Color <input type="text" name="font_color" value="#fffff2" equired><br>
                    Background Color <input type="text" name="bg_color" value="#000000" required><br>
                    <input type="text" name="margin" value="26" readonly style="display: none;"><br>
                    <input type="submit" value="Process">
                </form>
                <h1 id="waitMessage"></h1>
            </div>
        </body>
        </html>
    ''')



@app.route('/process', methods=['POST'])
def process():
    global global_font_size, global_box_color, global_bg_color, global_margin
    global global_font_file_path
    
    static_out_file_server = os.path.join('static', 'output_root')
    tmp = os.path.join(os.getcwd(), 'tmp')
    final_out_path = os.path.join('static', 'output_root', 'final')
    outpath = os.path.join(static_out_file_server, 'output')
    
    try:
        # Cleanup old files
        remove_all_files_in_directory(os.path.join(outpath, 'videos'))
        remove_all_files_in_directory(os.path.join(outpath, 'audios'))
        remove_all_files_in_directory(final_out_path)
        
        if os.path.exists(tmp):
            tmp_dirs = os.listdir(tmp)
            for dir in tmp_dirs:
                remove_all_files_in_directory(os.path.join(tmp, dir))
            remove_all_files_in_directory(tmp)
    except Exception as e:
        return f"An error occurred during cleanup: {e}", 500
    
    try:
        # Create necessary directories
        os.makedirs(outpath, exist_ok=True)
        os.makedirs(os.path.join(outpath, 'audios'), exist_ok=True)
        os.makedirs(os.path.join(outpath, 'videos'), exist_ok=True)
        os.makedirs(final_out_path, exist_ok=True)
        os.makedirs(tmp, exist_ok=True)  # Ensure the tmp directory exists
    except Exception as e:
        return f"An error occurred during directory creation: {e}", 500
    
    unique_special_id = os.path.join(tmp, generate_unique_id())
    
    video_dir = os.path.join(unique_special_id, "video")
    clips_dir = os.path.join(unique_special_id, "clips")
    mp3_dir = os.path.join(unique_special_id, "mp3")
    text_dir = os.path.join(unique_special_id, "text")
    font_dir = os.path.join(unique_special_id, "font")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(mp3_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(font_dir, exist_ok=True)

    try:
        # Save uploaded files
        video_file = request.files.get('video_file')
        mp3_file = request.files.get('mp3_file')
        text_file = request.files.get('text_file')
        font_file = request.files.get('font_file')
        
        # Debug print to check if files were uploaded
        print(f"[DEBUG] Video File: {video_file}", flush=True)
        print(f"[DEBUG] MP3 File: {mp3_file}", flush=True)
        print(f"[DEBUG] Text File: {text_file}", flush=True)
        print(f"[DEBUG] Font File: {font_file}", flush=True)

        if not video_file or not mp3_file or not text_file or not font_file:
            return "Missing required files", 400

        video_file_path = os.path.join(video_dir, video_file.filename)
        mp3_file_path = os.path.join(mp3_dir, mp3_file.filename)
        text_file_path = os.path.join(text_dir, text_file.filename)
        font_file_path = os.path.join(font_dir, font_file.filename)
        
        video_file.save(video_file_path)
        mp3_file.save(mp3_file_path)
        text_file.save(text_file_path)
        font_file.save(font_file_path)
        
        
        # Debug print to confirm files have been saved
        print(f"[DEBUG] Video File Saved: {video_file_path}", flush=True)
        print(f"[DEBUG] MP3 File Saved: {mp3_file_path}", flush=True)
        print(f"[DEBUG] Text File Saved: {text_file_path}", flush=True)
        print(f"[DEBUG] Font File Saved: {font_file_path}", flush=True)


    except Exception as e:
        print(f"[ERROR] Failed to save files: {e}", flush=True)
        return f"An error occurred while saving files: {e}", 500
    
    
    # New parameters
    global_font_size = int(request.form.get('font_size'))
    global_box_color = str(request.form.get('font_color'))
    global_bg_color = str(request.form.get('bg_color'))
    global_margin = int(request.form.get('margin', 20)) 
    global_font_file_path = font_file_path
    
    print(f"[DEBUG] Font Size: {global_font_size}", flush=True)
    print(f"[DEBUG] Box Color: {global_box_color}", flush=True)
    print(f"[DEBUG] Background Color: {global_bg_color}", flush=True)
    print(f"[DEBUG] Margin: {global_margin}", flush=True)

    if not global_font_size or not global_box_color or not global_bg_color:
        return "Missing required form data", 400

    # Generate the SRT file from TXT and MP3 files
    try:
        srt_file = generate_srt_from_txt_and_audio(Path(text_file_path), Path(mp3_file_path), Path(tmp))
    except Exception as e:
        return f"Failed to generate SRT file: {e}", 500
    
    # final_font_path = os.path.join('uploads', 'final_font_file.ttf')
    # shutil.move(font_file_path, final_font_path)


    # Move the SRT file to uploads directory for further processing
    final_srt_path = os.path.join('uploads', 'original_subtitles.srt')
    shutil.move(srt_file, final_srt_path)
    
    # Move the video file to uploads directory for further processing
    final_video_path = os.path.join('uploads', 'original_video.mp4')
    shutil.move(video_file_path, final_video_path)
    
    return redirect(url_for('video_processing_page'))

@app.route('/video_processing')
def video_processing_page():
    # Clear the session replacements at the start of the new session
    session.pop('replacements', None)
    
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Scene Optimisation Bot - Video Editor</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background-color: #f0f2f5;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                }

                .container {
                    background-color: #ffffff;
                    padding: 30px;
                    border-radius: 12px;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
                    width: 80%;
                    max-width: 900px;
                    text-align: center;
                }

                h1 {
                    color: #333;
                    font-size: 28px;
                    margin-bottom: 20px;
                }

                p.instructions {
                    color: #666;
                    font-size: 16px;
                    margin-bottom: 25px;
                }

                .video-container {
                    position: relative;
                    display: flex;
                    justify-content: center;
                    margin-bottom: 30px;
                }

                video {
                    width: 100%;
                    max-width: 400px;
                    height: auto;
                    border-radius: 8px;
                    border: 1px solid #ddd;
                }

                .btn {
                    padding: 12px 20px;
                    font-size: 16px;
                    font-weight: 600;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    cursor: pointer;
                    transition: background-color 0.3s ease;
                    margin-top: 10px;
                }

                .btn-primary {
                    background-color: #007bff;
                }

                .btn-primary:hover {
                    background-color: #0056b3;
                }

                .btn-success {
                    background-color: #28a745;
                }

                .btn-success:hover {
                    background-color: #218838;
                }

                .spinner {
                    display: none;
                    width: 50px;
                    height: 50px;
                    border: 5px solid rgba(0, 0, 0, 0.1);
                    border-top: 5px solid #007bff;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                    margin: 20px auto;
                }

                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
            <script>
                let selectedSegments = [];

                function getSceneIndex(currentTime) {
                    fetch(`/get_srt_index?time=${currentTime}`)
                        .then(response => response.json())
                        .then(data => {
                            if (data.srt_index !== -1) {
                                var newFileInput = document.createElement('input');
                                newFileInput.type = 'file';
                                newFileInput.accept = 'video/*';
                                newFileInput.onchange = function(event) {
                                    var file = event.target.files[0];

                                    let formData = new FormData();
                                    formData.append('scene', file);
                                    formData.append('srt_index', data.srt_index);

                                    fetch('/upload_new_scene', {
                                        method: 'POST',
                                        body: formData
                                    }).then(response => {
                                        if (response.ok) {
                                            alert('Segment uploaded and stored for replacement.');
                                        } else {
                                            alert('Failed to upload the segment.');
                                        }
                                    });
                                };
                                newFileInput.click();
                            }
                        })
                        .catch(error => console.error('Error:', error));
                }
                
                function processSegments() {
                    document.getElementById('spinner').style.display = 'block';  // Show the spinner
                    fetch('/process_video', {
                        method: 'POST'
                    }).then(response => {
                        document.getElementById('spinner').style.display = 'none';  // Hide the spinner
                        if (response.ok) {
                            // Append a timestamp to the video source URL to force reload
                            let videoPlayer = document.getElementById("videoPlayer");
                            let newVideoSrc = `/uploads/original_video.mp4?timestamp=${new Date().getTime()}`;
                            videoPlayer.src = newVideoSrc;
                            videoPlayer.load();
                            alert('Video compiled and segments replaced successfully!');

                            // Create a download button for the user to download the processed video
                            let downloadButton = document.createElement('a');
                            downloadButton.href = newVideoSrc;
                            downloadButton.download = 'processed_video.mp4';
                            downloadButton.className = 'btn btn-primary';
                            downloadButton.textContent = 'Download Processed Video';

                            let container = document.querySelector('.container');
                            container.appendChild(downloadButton);
                        } else {
                            alert('Failed to compile and replace segments.');
                        }
                    }).catch(error => {
                        document.getElementById('spinner').style.display = 'none';  // Hide the spinner
                        console.error('Error during processing:', error);
                        alert('An error occurred during processing.');
                    });
                }

                document.addEventListener('DOMContentLoaded', function() {
                    var videoPlayer = document.getElementById('videoPlayer');
                    videoPlayer.addEventListener('click', function(event) {
                        if (event.shiftKey) {
                            event.preventDefault();
                            event.stopPropagation();
                            getSceneIndex(videoPlayer.currentTime);
                        }
                    });
                });
            </script>
        </head>
        <body>
        <div class="container">
            <h1>Video Editor</h1>
            <p class="instructions">
                Use this video editor to replace segments of the video. <br>
                <strong>Instructions:</strong> <br>
                - Play the video and pause it at the point you want to replace. <br>
                - Hold <strong>Shift</strong> and click on the video to choose a new video segment to upload and replace the current segment.
            </p>
            <div class="video-container">
                <video id="videoPlayer" controls>
                    <source src="/uploads/original_video.mp4" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>
            <div id="spinner" class="spinner"></div> <!-- Loading Spinner -->
            <button class="btn btn-success" onclick="processSegments()">Process</button>
        </div>
        </body>
        </html>
    ''')
    

@app.route('/get_srt_index')
def get_srt_index():
    current_time = float(request.args.get('time'))
    subtitles = load_subtitles_from_file(Path('uploads/original_subtitles.srt'))
    
    # Iterate over the subtitles to find which one matches the current time
    for index, subtitle in enumerate(subtitles):
        start_time = subriptime_to_seconds(subtitle.start)
        end_time = subriptime_to_seconds(subtitle.end)
        if start_time <= current_time <= end_time:
            return {"srt_index": index}
    
    return {"srt_index": -1}  # Return -1 if no matching subtitle is found

def process_multiple_video_segment_replacements(original_video_path, subtitles_path, replacements, font_path, font_size, font_color, bg_color, margin):
    # Load original video and subtitles
    video = load_video_from_file(Path(original_video_path))
    subtitles = load_subtitles_from_file(Path(subtitles_path))
    timestamps = split_by_computer_vision(Path(original_video_path))
    
    for ts in timestamps:
        if ts['confidence'] > MAE_THRESHOLD:
            logging.debug(f"Frame: {ts['frame_number']}, Timestamp: {ts['timestamp']}, Confidence: {ts['confidence']}")
    logging.info("Video loaded successfully")
    cropped_video = crop_to_aspect_ratio(video, 4 / 5)
    logging.info("Video cropped to desired aspect ratio")
    
    
    refined_subtitles = refine_subtitles_based_on_computer_vision(subtitles, timestamps, replacements)
    refined_srt_file = Path(subtitles_path).with_name(Path(subtitles_path).stem + "_refined.srt")
    
    # Save the refined subtitles
    refined_subtitles.save(refined_srt_file)
    logging.info(f"Refined subtitles saved to {refined_srt_file}")

    # Attempt to move the refined SRT file to the final location
    try:
        final_srt_path = os.path.join('uploads', 'original_subtitles.srt')
        shutil.move(refined_srt_file, final_srt_path)
        logging.info(f"Moved refined SRT file to {final_srt_path}")
    except FileNotFoundError as e:
        logging.error(f"Error moving refined SRT file: {e}")
        raise

    # Segment the original video based on the subtitles
    video_segments, subtitle_segments = get_segments_using_srt(video, refined_subtitles)

    # Process each replacement
    for replacement in replacements:
        srt_index = replacement['srt_index']
        replacement_video_path = replacement['scene_path']
        
        # Load the replacement video segment
        replacement_video = load_video_from_file(Path(replacement_video_path))
        cropped_replacement_video = crop_to_aspect_ratio(replacement_video, video.aspect_ratio)
        
        # Debug prints for the parameters
        print(f"[DEBUG] Font Path: {font_path}", flush=True)
        print(f"[DEBUG] Font Size: {font_size}", flush=True)
        print(f"[DEBUG] Font Color: {font_color}", flush=True)
        print(f"[DEBUG] Background Color: {bg_color}", flush=True)
        print(f"[DEBUG] Margin: {margin}", flush=True)

        # Replace the specific segment in the video
        video_segments[srt_index] = replace_video_segments(
            video_segments, 
            {srt_index: cropped_replacement_video}, 
            subtitles, 
            video, 
            font_path, 
            font_size, 
            font_color, 
            bg_color, 
            margin
        )[srt_index]  # only replace the specific segment

    # Concatenate the updated video segments into a final video
    final_video = concatenate_videoclips(video_segments)
    original_audio = video.audio.subclip(0, final_video.duration)
    final_video_with_audio = final_video.set_audio(original_audio)

    # Save the final video with all the replaced segments
    temp_final_video_path = Path('uploads') / 'temp_final_video.mp4'
    final_video_with_audio.write_videofile(temp_final_video_path.as_posix(), codec="libx264", audio_codec="aac")

    # Replace the original video with the new one
    os.remove(original_video_path)  # Remove the old file
    temp_final_video_path.rename(original_video_path)  # Rename temp file to original path

    return "Success"


@app.route('/process_video', methods=['POST'])
def process_video():
    original_video_path = 'uploads/original_video.mp4'
    subtitles_path = 'uploads/original_subtitles.srt'

    # Load all replacements from session
    replacements = session.get('replacements', [])
    
    # Debug print to check if replacements exist
    print(f"[DEBUG] Replacements: {replacements}", flush=True)
    
    # Ensure we have replacements to process
    if not replacements:
        return "No segments to replace", 400
    
    if not os.path.exists(global_font_file_path):
        print(f"[ERROR] Font file not found at: {global_font_file_path}", flush=True)
    else:
        print(f"[ERROR] Font file was found at: {global_font_file_path}", flush=True)

    # Process all replacements
    process_multiple_video_segment_replacements(
        original_video_path=original_video_path,
        subtitles_path=subtitles_path,
        replacements=replacements,
        font_path=global_font_file_path,
        font_size=int(global_font_size),
        font_color=str(global_box_color),
        bg_color=str(global_bg_color),
        margin=int(global_margin)
    )

    # Clear the session replacements after processing
    session.pop('replacements', None)

    return "Video processing and segment replacement completed successfully."


@app.route('/upload_new_scene', methods=['POST'])
def upload_new_scene():
    srt_index = int(request.form['srt_index'])
    new_scene = request.files['scene']

    temp_scene_path = os.path.join('uploads', new_scene.filename)
    new_scene.save(temp_scene_path)

    # Store the replacement details in session
    if 'replacements' not in session:
        session['replacements'] = []
    
    
    
    if not any(r['srt_index'] == srt_index for r in session['replacements']):
        session['replacements'].append({
            'srt_index': srt_index,
            'scene_path': temp_scene_path
        })
    
    
    # Mark the session as modified to ensure it gets saved
    session.modified = True

    # Debug prints
    print(f"[DEBUG] Uploaded SRT Index: {srt_index}", flush=True)
    print(f"[DEBUG] Temporary Scene Path: {temp_scene_path}", flush=True)
    print(f"[DEBUG] Current Replacements in Session: {session['replacements']}", flush=True)

    return "Scene uploaded and stored for replacement."



@app.route('/uploads/<filename>')
def download_file(filename):
    return send_from_directory('uploads', filename)

def remove_all_files_in_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"An error occurred while removing {file_path}: {e}")
    else:
        print(f"Directory {directory} does not exist")

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
