import os
import pydicom
import pandas as pd
import numpy as np
from PIL import Image
import json
from pathlib import Path
from tqdm import tqdm

def normalize_dicom(dicom_data):
    """归一化DICOM图像数据"""
    img = dicom_data.pixel_array
    # 归一化到[0, 255]
    if img.max() != img.min():
        img = ((img - img.min()) * 255.0 / (img.max() - img.min())).astype(np.uint8)
    else:
        img = (img * 255.0).astype(np.uint8)
    return img

def create_image_text_pair(row, series_info_df, image_base_path, output_base_path):
    """为每个标签创建图像-文本对"""
    study_id = str(row['study_id'])
    
    # 获取该study的所有系列
    study_series = series_info_df[series_info_df['study_id'] == int(study_id)]
    
    pairs = []
    for _, series in study_series.iterrows():
        series_id = str(series['series_id'])
        series_desc = series['series_description']
        
        # 构建DICOM文件路径
        dicom_dir = os.path.join(image_base_path, study_id, series_id)
        if not os.path.exists(dicom_dir):
            continue
            
        # 处理该系列中的所有DICOM文件
        for dicom_file in sorted(os.listdir(dicom_dir)):
            if not dicom_file.endswith('.dcm'):
                continue
                
            dicom_path = os.path.join(dicom_dir, dicom_file)
            
            try:
                # 读取和处理DICOM文件
                ds = pydicom.dcmread(dicom_path)
                img = normalize_dicom(ds)
                
                # 创建输出目录
                output_dir = os.path.join(output_base_path, 'images', study_id, series_id)
                os.makedirs(output_dir, exist_ok=True)
                
                # 保存为PNG
                img_filename = f"{os.path.splitext(dicom_file)[0]}.png"
                img_path = os.path.join(output_dir, img_filename)
                Image.fromarray(img).save(img_path)
                
                # 创建描述文本
                conditions = []
                for col in row.index:
                    if col != 'study_id' and not pd.isna(row[col]):
                        condition_name = col.replace('_', ' ').title()
                        severity = row[col]
                        conditions.append(f"{condition_name}: {severity}")
                
                text = f"Lumbar spine MRI, {series_desc}. Findings: {'; '.join(conditions)}"
                
                # 添加到配对列表
                pairs.append({
                    'image_path': os.path.relpath(img_path, output_base_path),
                    'text': text,
                    'study_id': study_id,
                    'series_id': series_id,
                    'instance_number': dicom_file.split('.')[0],
                    'series_description': series_desc
                })
                
            except Exception as e:
                print(f"Error processing {dicom_path}: {str(e)}")
                
    return pairs

def main():
    # 设置路径
    base_path = os.path.dirname(os.path.abspath(__file__))
    train_csv_path = os.path.join(base_path, 'train.csv')
    train_series_desc_path = os.path.join(base_path, 'train_series_descriptions.csv')
    train_images_path = os.path.join(base_path, 'train_images')
    output_base_path = os.path.join(base_path, 'processed_data')
    
    # 创建输出目录
    os.makedirs(os.path.join(output_base_path, 'images'), exist_ok=True)
    
    # 读取CSV文件
    print("Reading CSV files...")
    train_df = pd.read_csv(train_csv_path)
    series_info_df = pd.read_csv(train_series_desc_path)
    
    # 处理每个训练样本
    print("Processing training data...")
    all_pairs = []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        pairs = create_image_text_pair(row, series_info_df, train_images_path, output_base_path)
        all_pairs.extend(pairs)
    
    # 保存数据集信息
    print("Saving dataset information...")
    dataset_info = {
        'num_samples': len(all_pairs),
        'image_text_pairs': all_pairs
    }
    
    with open(os.path.join(output_base_path, 'dataset.json'), 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # 创建训练集CSV
    df = pd.DataFrame(all_pairs)
    df.to_csv(os.path.join(output_base_path, 'train_dataset.csv'), index=False)
    
    print(f"Processing completed. Total samples: {len(all_pairs)}")
    print(f"Output saved to: {output_base_path}")

if __name__ == "__main__":
    main() 