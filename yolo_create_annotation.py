import os

def create_yolo_annotations(root_dir, output_dir, class_mapping):
    os.makedirs(output_dir, exist_ok=True)
    for class_name, class_id in class_mapping.items():
        class_path = os.path.join(root_dir, class_name)
        
        if not os.path.isdir(class_path):
            print(f"Skipping {class_name}: Directory does not exist.")
            continue
        
        print(f"Processing class: {class_name} (ID: {class_id})")
        
        # Get all images in the class folder
        for image_name in os.listdir(class_path):
            if image_name.endswith(('.JPG', '.png', '.jpeg', '.jpg')):
                # Debug: Processing image
                # print(f"Processing image: {image_name}")
                
                # Create a .txt file for each image
                image_base = os.path.splitext(image_name)[0]
                txt_file_path = os.path.join(output_dir, f"{class_name}_{image_base}.txt")
                # YOLO format (class_id, bbox - using full image as 1x1 bounding box)
                with open(txt_file_path, "w") as f:
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
            else:
                # Debug: Skipped file
                print(f"Skipped file: {image_name} (not an image)")


                
# Map folder names to class IDs
class_mapping = {
    "Tomato___healthy": 0,
    "Tomato___Bacterial_spot": 1,
    "Tomato___Early_blight": 2,
    "Tomato___Late_blight": 3,
    "Tomato___Leaf_Mold": 4,
    "Tomato___Septoria_leaf_spot": 5,
    "Tomato___Spider_mites Two-spotted_spider_mite": 6,
    "Tomato___Target_Spot": 7,
    "Tomato___Tomato_mosaic_virus": 8,
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 9,
    "Apple___Apple_scab": 10,
    "Apple___Black_rot": 11,
    "Apple___Cedar_apple_rust": 12,
    "Apple___healthy": 13,
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": 14,
    "Corn_(maize)___Common_rust_": 15,
    "Corn_(maize)___healthy": 16,
    "Corn_(maize)___Northern_Leaf_Blight": 17,
    "Grape___Black_rot": 18,
    "Grape___Esca_(Black_Measles)": 19,
    "Grape___healthy": 20,
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": 21,
    "Potato___Early_blight": 22,
    "Potato___healthy": 23,
    "Potato___Late_blight": 24,
}

create_yolo_annotations("new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train", "dataset/labels/train", class_mapping)
create_yolo_annotations("New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid", "dataset/labels/val", class_mapping)
