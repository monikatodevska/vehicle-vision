import os
import subprocess


def extract_frames(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all .avi files in the input folder
    video_files = [f for f in os.listdir(input_folder) if f.endswith('.avi')]

    for idx, video_file in enumerate(video_files):
        input_path = os.path.join(input_folder, video_file)

        # Define the output file pattern (e.g., image000000.bmp)
        output_pattern = os.path.join(output_folder, f'image{idx:06d}.bmp')

        # Run ffmpeg command to extract frames
        subprocess.run([
            'ffmpeg',
            '-i', input_path,
            '-vf', 'fps=1',  # Change the fps value as needed
            output_pattern
        ])


if __name__ == "__main__":
    input_folder = r'D:\Monika\VideosTest\KAMIONI'
    output_folder = r'D:\Monika\VideosTest\KAMIONI\frames'

    extract_frames(input_folder, output_folder)
