import os

def convert_bboxes_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if not os.path.isfile(file_path):
            continue  # skip directories

        with open(file_path, 'r') as file:
            lines = file.readlines()

        converted_lines = []
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) != 4:
                continue  # skip malformed lines

            y1, x1, y2, x2 = map(int, parts)
            h = y2 - y1
            w = x2 - x1
            converted_lines.append(f"{y1},{x1},{h},{w}\n")

        with open(file_path, 'w') as file:
            file.writelines(converted_lines)

# Example usage
folder_path = 'D:\Monika\VideosTest\Vozila\proba_annot\Cam46\kola_710'
convert_bboxes_in_folder(folder_path)
