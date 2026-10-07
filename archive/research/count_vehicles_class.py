import os

# Set your root folder path here
root_folder = r'D:\Monika\VideosTest\All_Cameras_annnotated\Annotations\Renamed'

# import os

# Set your root folder path here
# root_folder = '/path/to/root/folder'

# Initialize counters for each class
classes_of_interest = [1, 2, 3, 4]
class_counts = {cls: 0 for cls in classes_of_interest}
image_counts = {cls: 0 for cls in classes_of_interest}
total_images = 0

for dirpath, dirnames, filenames in os.walk(root_folder):
    for file in filenames:
        if file.endswith('.txt'):
            total_images += 1
            class_found = {cls: False for cls in classes_of_interest}
            file_path = os.path.join(dirpath, file)

            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 5:
                        try:
                            cls = int(parts[4])
                            if cls in class_counts:
                                class_counts[cls] += 1
                                class_found[cls] = True
                        except ValueError:
                            continue

            for cls in classes_of_interest:
                if class_found[cls]:
                    image_counts[cls] += 1

# Output
print(f"Total images (with annotations): {total_images}")
for cls in classes_of_interest:
    print(f"Class {cls}: {class_counts[cls]} bboxes in {image_counts[cls]} images")
