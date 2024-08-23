import logging
from pathlib import Path
from typing import List, Dict
import os
import subprocess
import json
import pysrt
import textwrap
import shutil
from moviepy.editor import (
    AudioFileClip, ColorClip, CompositeVideoClip, concatenate_videoclips,
    TextClip, VideoFileClip
)
from logging import info, error, debug
from moviepy.video.fx.crop import crop
from moviepy.video.fx.loop import loop
import matplotlib.colors as mcolors
import cv2
import numpy as np
import sys

# Initialization
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)

# Set the path to the ImageMagick executable
os.environ['IMAGEMAGICK_BINARY'] = '/usr/bin/convert'


MAE_THRESHOLD: float = 4.2
GLITCH_IGNORE_THRESHOLD: float = 0.27

def split_by_computer_vision(video_path: str = 'your_video.mp4'):
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    # Define the region of interest (ROI) for the subtitle area
    # Adjust these values based on your video resolution and subtitle area
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    BLEEDING = 40
    LINE_HEIGHT = 60

    roi_top = HEIGHT - BLEEDING - LINE_HEIGHT
    roi_bottom = HEIGHT - BLEEDING
    roi_left = BLEEDING
    roi_right = WIDTH - BLEEDING

    # Initialize variables
    prev_frame = None
    timestamps = []

    # Process each frame
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop the subtitle area
        subtitle_area = frame[roi_top:roi_bottom, roi_left:roi_right]
        
        # Convert to grayscale
        gray = cv2.cvtColor(subtitle_area, cv2.COLOR_BGR2GRAY)
        
        # Apply a binary threshold to create a binary image
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # save the binary image for debugging
        # cv2.imwrite(f'tmp/binary_{frame_number}.png', binary)
        
        # Compare with the previous frame
        if prev_frame is not None:
            # Calculate the difference between the current frame and the previous frame
            diff = cv2.absdiff(prev_frame, binary)
            
            # Calculate the percentage of different pixels
            non_zero_count = np.count_nonzero(diff)
            total_count = diff.size
            diff_percentage = (non_zero_count / total_count) * 100
            timestamp = frame_number / fps
            timestamps.append({
                'frame_number': frame_number,
                'timestamp': timestamp,
                'confidence': diff_percentage
            })
        
        # Update the previous frame
        prev_frame = binary
        frame_number += 1

    # Release the video capture
    cap.release()

    return timestamps

def load_video_from_file(file: Path) -> VideoFileClip:
    if not file.exists():
        raise FileNotFoundError(f"Video file not found: {file}")
    logging.info(f"Loading video file: {file}")
    return VideoFileClip(file.as_posix())


def crop_to_aspect_ratio(video: VideoFileClip, desired_aspect_ratio: float) -> VideoFileClip:
    video_aspect_ratio = video.w / video.h
    if video_aspect_ratio > desired_aspect_ratio:
        new_width = int(desired_aspect_ratio * video.h)
        new_height = video.h
        x1 = (video.w - new_width) // 2
        y1 = 0
    else:
        new_width = video.w
        new_height = int(video.w / desired_aspect_ratio)
        x1 = 0
        y1 = (video.h - new_height) // 2
    x2 = x1 + new_width
    y2 = y1 + new_height
    return crop(video, x1=x1, y1=y1, x2=x2, y2=y2)


def load_subtitles_from_file(srt_file: Path) -> pysrt.SubRipFile:
    if not srt_file.exists():
        raise FileNotFoundError(f"SRT File not found: {srt_file}")
    return pysrt.open(srt_file)


def adjust_segment_duration(segment: VideoFileClip, duration: float) -> VideoFileClip:
    current_duration = segment.duration
    if current_duration < duration:
        return loop(segment, duration=duration)
    elif current_duration > duration:
        return segment.subclip(0, duration)
    return segment


def adjust_segment_properties(segment: VideoFileClip, original: VideoFileClip) -> VideoFileClip:
    segment = segment.set_fps(original.fps)
    segment = segment.set_duration(segment.duration)
    segment = segment.resize(newsize=(original.w, original.h))
    return segment


def subriptime_to_seconds(srt_time: pysrt.SubRipTime) -> float:
    return srt_time.hours * 3600 + srt_time.minutes * 60 + srt_time.seconds + srt_time.milliseconds / 1000.0


def get_segments_using_srt(video: VideoFileClip, subtitles: pysrt.SubRipFile) -> (List[VideoFileClip], List[pysrt.SubRipItem]):
    subtitle_segments = []
    video_segments = []
    for subtitle in subtitles:
        start = subriptime_to_seconds(subtitle.start)
        end = subriptime_to_seconds(subtitle.end)
        video_segment = video.subclip(start, end)
        subtitle_segments.append(subtitle)
        video_segments.append(video_segment)
    return video_segments, subtitle_segments


def convert_color(color):
    """Convert color to an RGB tuple."""
    if isinstance(color, str):
        if color.startswith('#'):
            # Convert hex string to RGB tuple
            return tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        elif color.startswith('(') and color.endswith(')'):
            # Convert string in the form '(r, g, b)' to RGB tuple
            return tuple(map(int, color.strip('()').split(',')))
        else:
            # Convert named color to RGB tuple
            rgb = mcolors.to_rgb(color)
            return tuple(int(c * 255) for c in rgb)
    else:
        raise ValueError("Color format not recognized. Provide a hex string, named color, or RGB tuple as a string.")


def add_subtitles_to_clip(
    clip: VideoFileClip,
    subtitle: pysrt.SubRipItem,
    font_path: str,
    font_size: int = 36,
    font_color: str = "white",
    bg_color: str = "black",
    margin: int = 26,
) -> VideoFileClip:
    logging.info(f"Adding subtitle: {subtitle.text}")

    # Maximum width allowed for the subtitle box
    max_box_width = clip.w - 2 * margin

    # Create a TextClip to measure the width of text without wrapping
    measuring_clip = TextClip(
        subtitle.text,
        fontsize=font_size,
        color=font_color,
        font=font_path,
        stroke_color=font_color,
        stroke_width=1
    )
    text_width, text_height = measuring_clip.size
    measuring_clip.close()

    padding = 6
    subtitle_duration = subriptime_to_seconds(subtitle.end) - subriptime_to_seconds(subtitle.start)

    # Wrap the text manually based on max_box_width
    def wrap_text(text, max_width):
        # Create a TextClip with a temporary small width to avoid wrapping
        temp_clip = TextClip(text, fontsize=font_size, font=font_path, color=font_color)
        lines = textwrap.wrap(text, width=max_width)
        return lines

    wrapped_lines = wrap_text(subtitle.text, max_box_width-padding)
    
    # Measure each line
    max_line_width = 0
    for line in wrapped_lines:
        line_clip = TextClip(line, fontsize=font_size, color=font_color, font=font_path)
        line_width, _ = line_clip.size
        max_line_width = max(max_line_width, line_width)
        line_clip.close()

    # Set the width of the subtitle box
    box_width = min(max_line_width + padding, max_box_width)

    # Create the subtitle clip with the exact maximum width
    subtitle_clip = TextClip(
        subtitle.text,
        fontsize=font_size,
        color=font_color,
        font=font_path,
        method="caption",
        align="center",
        size=(box_width, None)
    ).set_duration(subtitle_duration)

    text_width, text_height = subtitle_clip.size

    # Add padding and create the background box
    box_height = text_height + 2 * padding

    # Convert bg_color to RGB tuple if necessary
    bg_color = convert_color(bg_color)
    bg_opacity = 0.5

    # Create the background box clip
    box_clip = (
        ColorClip(size=(box_width, box_height), color=bg_color)
        .set_opacity(bg_opacity)
        .set_duration(subtitle_clip.duration)
    )

    # Calculate positions
    box_position = ("center", clip.h - box_height - margin)
    subtitle_position = ("center", clip.h - box_height - margin + (box_height - text_height) / 2)

    # Set the positions of the box and the text clip
    box_clip = box_clip.set_position(box_position)
    subtitle_clip = subtitle_clip.set_position(subtitle_position)

    # Return composite video clip with the added subtitle and background box
    return CompositeVideoClip([clip, box_clip, subtitle_clip])


def replace_video_segments(
    original_segments: List[VideoFileClip],
    replacement_videos: Dict[int, VideoFileClip],
    subtitles: pysrt.SubRipFile,
    original_video: VideoFileClip,
    font_path: str,
    font_size: int,
    font_color: str,
    bg_color: str,
    margin: int 
) -> List[VideoFileClip]:
    combined_segments = original_segments.copy()
    for replace_index, replacement_video in replacement_videos.items():
        if 0 <= replace_index < len(combined_segments):
            target_duration = combined_segments[replace_index].duration
            start = subriptime_to_seconds(subtitles[replace_index].start)
            end = subriptime_to_seconds(subtitles[replace_index].end)

            # here, in the thrid clip, the video length is only 3 seconds in lenght, but the subtitle asks to crop 4~6
            # the problem would only be worse since the subtitle timestamp is getting longer and longer, but clips are always short
            # a potential fix here is to always crop the video to the subtitle length, from 0 to subtitle length
            
            subtitle_length = end - start
            if subtitle_length >= replacement_video.duration: # if the subtitle length is longer than the replacement video
                subtitle_length = replacement_video.duration
            
            replacement_segment = replacement_video.subclip(0, subtitle_length) # always crop to subtitle length, from 0
            replacement_segment = adjust_segment_duration(replacement_segment, target_duration)
            adjusted_segment = adjust_segment_properties(replacement_segment, original_video)
            adjusted_segment_with_subtitles = add_subtitles_to_clip(adjusted_segment, 
                                                                    subtitles[replace_index], font_path, font_size, font_color, bg_color, margin)
            combined_segments[replace_index] = adjusted_segment_with_subtitles
    return combined_segments


def generate_srt_from_txt_and_audio(txt_file: Path, audio_file: Path, output_folder: Path) -> Path:
    output_file_path = txt_file.with_name(txt_file.stem + "_aligned.json")
    command = f'{sys.executable} -m aeneas.tools.execute_task "{audio_file}" "{txt_file}" "task_language=eng|is_text_type=plain|os_task_file_format=json" "{output_file_path}"'
    logging.info(f"Running command: {command}")
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    logging.info(f"Command output: {result.stdout.decode('utf-8')}")
    logging.error(f"Command error: {result.stderr.decode('utf-8')}")

    if not output_file_path.exists():
        raise FileNotFoundError(f"The output file {output_file_path} was not created. Check the command output above for errors.")

    with open(output_file_path, 'r') as f:
        sync_map = json.load(f)

    def convert_time(seconds):
        milliseconds = int((seconds - int(seconds)) * 1000)
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

    aligned_output = []
    for index, fragment in enumerate(sync_map['fragments']):
        start = convert_time(float(fragment['begin']))
        end = convert_time(float(fragment['end']))
        text = fragment['lines'][0].strip()
        aligned_output.append(f"{index + 1}\n{start} --> {end}\n{text}\n")

    srt_file = txt_file.with_name(txt_file.stem + "_with_timestamps.srt")
    with open(srt_file, 'w') as file:
        for line in aligned_output:
            file.write(line + "\n")

    return srt_file

def refine_subtitles_based_on_computer_vision(subtitles: pysrt.SubRipFile, timestamps: List[Dict], replacements: List[Dict]) -> pysrt.SubRipFile:
    candidate_timestamps = []
    logging.debug(f"Refining {len(subtitles)} subtitles")
    for ts in timestamps:
        if ts['confidence'] > MAE_THRESHOLD:
            if len(candidate_timestamps) == 0 or (ts["timestamp"] - candidate_timestamps[-1]["timestamp"] > GLITCH_IGNORE_THRESHOLD):
                candidate_timestamps.append(ts)
                logging.debug(f"Frame: {ts['frame_number']}, Timestamp: {ts['timestamp']}, Confidence: {ts['confidence']}")
    logging.info(f"Found {len(candidate_timestamps)} candidate timestamps for subtitle changes")
    if len(candidate_timestamps) != len(subtitles) - 1:
        logging.warning(f"The number of candidate timestamps does not match the number of subtitles, {len(subtitles)} vs {len(candidate_timestamps)}")

    clips = [replacement['srt_index'] for replacement in replacements]

    cursor = 0
    last_subtitle_record = None
    for subtitle in subtitles:
        if last_subtitle_record is not None:
            if last_subtitle_record.end > subtitle.start or last_subtitle_record.end < subtitle.start:
                logging.warning(f"Subtitle overlap detected: [{last_subtitle_record.text}] - [{subtitle.text}]")
                subtitle.start = pysrt.SubRipTime(
                    hours=last_subtitle_record.end.hours,
                    minutes=last_subtitle_record.end.minutes,
                    seconds=last_subtitle_record.end.seconds,
                    milliseconds=last_subtitle_record.end.milliseconds
                )
        while cursor < len(candidate_timestamps):
            candidate = candidate_timestamps[cursor]
            subtitle_end = subriptime_to_seconds(subtitle.end)
            candidate['timestamp'] = candidate['timestamp'] + 0.05
            if candidate['timestamp'] < subtitle_end - 0.25:
                cursor += 1
                logging.debug(f"Skipping candidate timestamp {candidate['timestamp']} for subtitle [{subtitle.text}]")
                continue
            if candidate['timestamp'] > subtitle_end + 1.5:
                logging.debug(f"Not found for subtitle [{subtitle.text}]")
                break
            logging.debug(f"Found candidate timestamp {candidate['timestamp']} for subtitle [{subtitle.text}]")
            subtitle.end = pysrt.SubRipTime(
                hours=int(candidate['timestamp'] // 3600),
                minutes=int((candidate['timestamp'] % 3600) // 60),
                seconds=int(candidate['timestamp'] % 60),
                milliseconds=int((candidate['timestamp'] % 1) * 1000)
            )
            cursor += 1
            break
        last_subtitle_record = subtitle
    
    logging.debug(f"All Clips: {clips}")
    for i, subtitle in enumerate(subtitles):
        if i > 0:
            if i in clips and i - 1 not in clips:
                logging.debug(f"Subtitle #{i} is a clip start, starting at: {subriptime_to_seconds(subtitle.start)}, line: {subtitle.text}")
                subtitle_start = subriptime_to_seconds(subtitle.start)
                subtitle_start -= 0.1
                subtitle.start = pysrt.SubRipTime(
                    hours=int(subtitle_start // 3600),
                    minutes=int((subtitle_start % 3600) // 60),
                    seconds=int(subtitle_start % 60),
                    milliseconds=int((subtitle_start % 1) * 1000)
                )
                subtitles[i-1].end = pysrt.SubRipTime(
                    hours=subtitle.start.hours,
                    minutes=subtitle.start.minutes,
                    seconds=subtitle.start.seconds,
                    milliseconds=subtitle.start.milliseconds
                )
    
    return subtitles




def main(video_clips_path, my_video, mp3_file_of_same_video, txt_file_of_same_video, output_folder, font_path, font_size, font_color, bg_color,margin):
    input_video_file = Path(my_video)
    replacement_base_folder = Path(video_clips_path)

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Generate SRT file from TXT and MP3
    srt_file = generate_srt_from_txt_and_audio(Path(txt_file_of_same_video), Path(mp3_file_of_same_video), output_folder)
    logging.info("Generated SRT file from TXT and MP3")

    video = load_video_from_file(input_video_file)
    timestamps = split_by_computer_vision(input_video_file)
    for ts in timestamps:
        if ts['confidence'] > MAE_THRESHOLD:
            logging.debug(f"Frame: {ts['frame_number']}, Timestamp: {ts['timestamp']}, Confidence: {ts['confidence']}")
    logging.info("Video loaded successfully")
    cropped_video = crop_to_aspect_ratio(video, 4 / 5)
    logging.info("Video cropped to desired aspect ratio")
    subtitles = load_subtitles_from_file(srt_file)
    refined_subtitles = refine_subtitles_based_on_computer_vision(subtitles, timestamps, replacement_base_folder)
    refined_srt_file = srt_file.with_name(srt_file.stem + "_refined.srt")
    # avoid float precision error
    refined_subtitles.save(refined_srt_file, encoding='utf-8')
    logging.info("Loaded SRT Subtitles from the provided subtitle file")
    video_segments, subtitle_segments = get_segments_using_srt(video, refined_subtitles)
    logging.info("Segmented Input video based on the SRT Subtitles generated for it")
    output_video_segments = []
    start = 0
    for video_segment, new_subtitle_segment in zip(video_segments, refined_subtitles):
        end = subriptime_to_seconds(new_subtitle_segment.end)
        required_duration = end - start
        new_video_segment = adjust_segment_duration(video_segment, required_duration)
        output_video_segments.append(new_video_segment.without_audio())
        start = end

    replacement_videos_per_combination = []

    for folder in replacement_base_folder.iterdir():
        if not folder.is_dir():
            continue

        folder_name = folder.name
        if not folder_name.isdigit():
            logging.warning(f"Folder name {folder_name} is not a valid segment index. Skipping...")
            continue

        replace_index = int(folder_name) - 1
        replacement_video_files = list(folder.glob("*.mp4"))
        logging.info(f"Found {len(replacement_video_files)} replacement video files in {folder}")

        for replacement_video_file in replacement_video_files:
            replacement_video = load_video_from_file(replacement_video_file)
            cropped_replacement_video = crop_to_aspect_ratio(replacement_video, 4 / 5)
            logging.info(f"Replacement video {replacement_video_file} cropped to desired aspect ratio")
            if len(replacement_videos_per_combination) < len(replacement_video_files):
                replacement_videos_per_combination.append({})
            replacement_videos_per_combination[replacement_video_files.index(replacement_video_file)][replace_index] = cropped_replacement_video

    for i, replacement_videos in enumerate(replacement_videos_per_combination):
        final_video_segments = replace_video_segments(
            output_video_segments, replacement_videos, subtitles, video , font_path, font_size,font_color, bg_color,margin
        )
        concatenated_video = concatenate_videoclips(final_video_segments)
        logging.debug(f'Duration: {concatenated_video.duration}')
        original_audio = video.audio.subclip(0, concatenated_video.duration)
        final_video_with_audio = concatenated_video.set_audio(original_audio)
        #tmp_path = Path('tmp')
        output_file = output_folder / f"output_variation_{i+1}.mp4"
        final_video_with_audio.write_videofile(output_file.as_posix(), codec="libx264", audio_codec="aac")
        #shutil.move(tmp_path, output_file)
        logging.info(f"Generated output video: {output_file}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Process video files")
    parser.add_argument("--input_clips", "-ic", required=True, help="Input clips directory")
    parser.add_argument("--input_video", "-iv", required=True, help="Input video file")
    parser.add_argument("--input_mp3", "-im", required=True, help="Input mp3 file")
    parser.add_argument("--input_txt", "-it", required=True, help="Input txt file")
    parser.add_argument("--output_dir", "-o", required=True, help="Output directory")
    parser.add_argument("--font_file", "-fn", default="Montserrat-SemiBold.ttf", help="Font path for subtitles")
    parser.add_argument("--font_size", "-fs", default=33, type=int, help="Font size for subtitles")
    parser.add_argument("--font_color", "-fc", default="white", help="Font color for subtitles")
    parser.add_argument("--bg_color", "-bc", default="black", help="Background color for subtitles")
    parser.add_argument("--margin", "-m", default=20, type=int, help="Margin for subtitles")
    
    args = parser.parse_args()
    main(args.input_clips, args.input_video, args.input_mp3, args.input_txt, Path(args.output_dir),args.font_file, args.font_size, args.font_color, args.bg_color, args.margin)

