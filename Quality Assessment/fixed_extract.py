import os
import cv2

# Update these paths to match your folder structure
path = "./data/aligned faces"  # Your custom aligned faces folder
outpath = "./data/quality_data"

dataset_name = "custom_dataset"  # Your custom dataset name
rel_img_path = os.path.join(outpath.split("/")[-1], dataset_name, "images")
outpath = os.path.join(outpath, dataset_name)

# Create output directories
if not os.path.exists(outpath):
    os.makedirs(outpath)
    os.makedirs(os.path.join(outpath, "images"))

# Since you have individual images, not organized by person folders
align_path = path  # Points directly to your aligned faces folder


def copy_img(img_filename):
    """Copy and convert image from BGR to RGB format"""
    src_path = os.path.join(align_path, img_filename)
    if not os.path.exists(src_path):
        print(f"Warning: Image {src_path} not found")
        return None
    
    img = cv2.imread(src_path)
    if img is None:
        print(f"Warning: Could not read image {src_path}")
        return None
    
    # Convert BGR to RGB (as noted in the instructions)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    dst_path = os.path.join(outpath, "images", img_filename)
    cv2.imwrite(dst_path, img)
    return img_filename


def create_image_list_from_custom_dataset():
    """Create image list from your custom aligned face dataset"""
    
    # Get all image files from your aligned faces folder
    image_files = [f for f in os.listdir(align_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) == 0:
        print("No image files found in the data folder!")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Create image_path_list.txt
    txt_file = open(os.path.join(outpath, "image_path_list.txt"), "w")
    pair_list = open(os.path.join(outpath, "pair_list.txt"), "w")

    # Process all images
    processed_images = []
    for img_file in image_files:
        copied_img = copy_img(img_file)
        if copied_img:
            processed_images.append(copied_img)
            txt_file.write(os.path.join(rel_img_path, copied_img) + "\n")
    
    # Create simple pairs (each image with itself for quality assessment)
    # Since we're doing quality assessment, not face verification, 
    # we can create simple pairs
    for i, img in enumerate(processed_images):
        # Create pairs for quality assessment
        if i < len(processed_images) - 1:
            pair_list.write(f"{img} {processed_images[i+1]} 0\n")  # 0 = different images
    
    txt_file.close()
    pair_list.close()
    print(f"Processed {len(processed_images)} images")
    print(f"Created {len(processed_images)-1} pairs")


# Run the extraction
create_image_list_from_custom_dataset()
print("Custom dataset successfully extracted and prepared!")
