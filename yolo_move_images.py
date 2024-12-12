import os
import shutil

def move_images(root_dir, output_dir, allowed_classes):
    os.makedirs(output_dir, exist_ok=True)
    for class_name in os.listdir(root_dir):
        # Skip classes not in the allowed list
        if class_name not in allowed_classes:
            print(f"Skipping class: {class_name}")
            continue

        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            print(f"Skipping {class_name}: Not a directory.")
            continue

        print(f"Processing images from class: {class_name}")
        
        for image_name in os.listdir(class_path):
            # Check if the file is an image
            if image_name.lower().endswith(('.JPG','.jpg', '.png', '.jpeg')):
                # Copy the image to the output directory
                src_path = os.path.join(class_path, image_name)
                dst_path = os.path.join(output_dir, f"{class_name}_{image_name}")
                shutil.copy(src_path, dst_path)
                print(f"Copied: {image_name} to {output_dir}")
            else:
                print(f"Skipped non-image file: {image_name}")

# Allowed classes
allowed_classes = ["Tomato___healthy", "Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight","Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot","Tomato___Tomato_mosaic_virus","Tomato___Tomato_Yellow_Leaf_Curl_Virus",
                    "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy", 
                    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_","Corn_(maize)___healthy","Corn_(maize)___Northern_Leaf_Blight",
                    "Grape___Black_rot","Grape___Esca_(Black_Measles)","Grape___healthy","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
                    "Potato___Early_blight","Potato___healthy","Potato___Late_blight"]

# Move train and validation images for the specified classes
move_images("new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train", "dataset/images/train", allowed_classes)
move_images("new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/valid", "dataset/images/val", allowed_classes)
