"""Prepare the FACET dataset for cross-dataset evaluation."""
import os
import argparse
import pandas as pd
import json
import tarfile
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def extract_bbox(bbox_str):
    try:
        bbox_dict = json.loads(bbox_str)
        return [bbox_dict['x'], bbox_dict['y'], bbox_dict['width'], bbox_dict['height']]
    except:
        return None

def process_facet_annotations(annotations_csv_path, output_dir):
    print("Loading annotations...")
    df = pd.read_csv(annotations_csv_path)
    
    print("Filtering for single-person images...")
    counts = df['filename'].value_counts()
    one_person_files = counts[counts == 1].index
    df = df[df['filename'].isin(one_person_files)]
    
    print("Filtering for binary gender presentation...")
    df = df[(df['gender_presentation_masc'] == 1) | (df['gender_presentation_fem'] == 1)].copy()
    
    print("Extracting labels and bounding boxes...")
    df['label'] = df.apply(lambda row: 'male' if row['gender_presentation_masc'] == 1 else 'female', axis=1)
    df['encoded_label'] = df['label'].map({'female': 0, 'male': 1})
    df['bbox'] = df['bounding_box'].apply(extract_bbox)
    
    # Rename columns for consistency
    df = df.rename(columns={'filename': 'image_name'})
    
    # Drop rows with missing bboxes just in case
    df = df.dropna(subset=['bbox'])
    
    print(f"Final dataset size: {len(df)}")
    print(df['label'].value_counts())
    
    # Stratified split
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['label'], random_state=42)
    
    os.makedirs(os.path.join(output_dir, 'splits'), exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'splits', 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'splits', 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'splits', 'test.csv'), index=False)
    
    return df['image_name'].tolist()

def extract_images(gz_files, needed_images, output_img_dir):
    os.makedirs(output_img_dir, exist_ok=True)
    needed_set = set(needed_images)
    found_count = 0
    
    for gz_file in gz_files:
        if not os.path.exists(gz_file):
            print(f"File {gz_file} not found, skipping...")
            continue
            
        print(f"Processing {gz_file}...")
        try:
            with tarfile.open(gz_file, 'r:gz') as tar:
                for member in tqdm(tar):
                    if not member.isfile():
                        continue
                        
                    filename = os.path.basename(member.name)
                    if filename in needed_set:
                        # Extract this file directly to the output dir
                        member.name = filename # Strip directory paths
                        tar.extract(member, output_img_dir)
                        found_count += 1
                        needed_set.remove(filename)
                        
                    if not needed_set:
                        break # Found all images
        except Exception as e:
            print(f"Error reading {gz_file}: {e}")
            
        if not needed_set:
            break
            
    print(f"Extracted {found_count} images. {len(needed_set)} images missing.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='data/facet_processed')
    args = parser.parse_args()
    
    print(f"Preparing FACET dataset from {args.data_dir} to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    annotations_path = os.path.join(args.data_dir, 'annotations', 'annotations.csv')
    if not os.path.exists(annotations_path):
        print(f"Annotations not found at {annotations_path}. Please extract annotations.gz first.")
        return
        
    needed_images = process_facet_annotations(annotations_path, args.output_dir)
    
    gz_files = [
        os.path.join(args.data_dir, 'imgs_1.gz'),
        os.path.join(args.data_dir, 'imgs_2.gz'),
        os.path.join(args.data_dir, 'imgs_3.gz')
    ]
    
    output_img_dir = os.path.join(args.output_dir, 'images')
    
    # Check if images are already extracted
    existing_images = []
    if os.path.exists(output_img_dir):
        existing_images = os.listdir(output_img_dir)
    
    if len(existing_images) >= len(needed_images) * 0.9: # 90% threshold just in case some are missing
        print(f"Found {len(existing_images)} images already extracted. Skipping extraction.")
    else:
        extract_images(gz_files, needed_images, output_img_dir)
        
    print("FACET preparation complete.")

if __name__ == '__main__':
    main()
