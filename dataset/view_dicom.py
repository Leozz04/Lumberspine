import os
import pydicom
import matplotlib.pyplot as plt
import numpy as np

def view_dicom(dicom_path):
    """查看单个DICOM文件"""
    # 读取DICOM文件
    ds = pydicom.dcmread(dicom_path)
    
    # 获取图像数据
    img = ds.pixel_array
    
    # 显示图像信息
    print("\nDICOM文件信息:")
    print(f"图像大小: {img.shape}")
    print(f"像素数据类型: {img.dtype}")
    if hasattr(ds, 'PatientID'):
        print(f"患者ID: {ds.PatientID}")
    if hasattr(ds, 'StudyDescription'):
        print(f"检查描述: {ds.StudyDescription}")
    if hasattr(ds, 'SeriesDescription'):
        print(f"序列描述: {ds.SeriesDescription}")
    
    # 保存图像
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('DICOM Image')
    output_path = 'dicom_preview.png'
    plt.savefig(output_path)
    plt.close()
    print(f"\n图像已保存为: {output_path}")

def main():
    # 设置示例DICOM文件路径（使用训练集中的第一个文件）
    train_path = 'train_images'
    
    # 获取第一个病例的第一个DICOM文件
    patient_dirs = os.listdir(train_path)
    if patient_dirs:
        patient_path = os.path.join(train_path, patient_dirs[0])
        study_dirs = os.listdir(patient_path)
        if study_dirs:
            study_path = os.path.join(patient_path, study_dirs[0])
            dicom_files = os.listdir(study_path)
            if dicom_files:
                dicom_path = os.path.join(study_path, dicom_files[0])
                print(f"查看DICOM文件: {dicom_path}")
                view_dicom(dicom_path)
            else:
                print("未找到DICOM文件")
        else:
            print("未找到研究目录")
    else:
        print("未找到患者目录")

if __name__ == "__main__":
    main() 