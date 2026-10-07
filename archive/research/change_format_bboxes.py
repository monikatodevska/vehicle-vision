import os

def convert_bbox_format(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    converted_lines = []
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) == 4:
            try:
                ymin, xmin, ymax, xmax = map(int, parts)
                h = ymax - ymin
                w = xmax - xmin
                new_line = f"{ymin},{xmin},{h},{w}\n"
                converted_lines.append(new_line)
            except ValueError:
                # Handle bad lines
                print(f"Skipping invalid line in {file_path}: {line.strip()}")
        else:
            print(f"Skipping malformed line in {file_path}: {line.strip()}")

    with open(file_path, 'w') as f:
        f.writelines(converted_lines)

def convert_all_txt_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.txt'):
                file_path = os.path.join(dirpath, filename)
                convert_bbox_format(file_path)
                print(f"Converted: {file_path}")

# Set your root directory or use current directory
convert_all_txt_files(r'D:\Monika\VideosTest\Vozila\brzini_test_bbox\Cam46')
