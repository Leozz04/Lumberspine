import os
import zipfile
from tqdm import tqdm

def get_zip_info(zip_path):
    """获取zip文件的信息"""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # 获取所有文件的总大小
        total_size = sum(info.file_size for info in zf.filelist)
        return {
            'total_size': total_size,
            'num_files': len(zf.filelist)
        }

def extract_with_progress(zip_path, extract_path):
    """带进度条地解压文件"""
    # 获取zip信息
    info = get_zip_info(zip_path)
    total_size = info['total_size']
    num_files = info['num_files']
    
    # 创建解压目录
    os.makedirs(extract_path, exist_ok=True)
    
    # 初始化进度条
    pbar = tqdm(total=total_size, unit='B', unit_scale=True)
    
    # 开始解压
    with zipfile.ZipFile(zip_path, 'r') as zf:
        print(f"\nExtracting {num_files} files...")
        for file_info in zf.filelist:
            try:
                zf.extract(file_info, extract_path)
                pbar.update(file_info.file_size)
            except Exception as e:
                print(f"\nError extracting {file_info.filename}: {str(e)}")
    
    pbar.close()

def main():
    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(current_dir, 'rsna-2024-lumbar-spine-degenerative-classification.zip')
    extract_path = current_dir
    
    print(f"Starting extraction of {zip_path}")
    print("This may take a while for a 29GB file...")
    
    try:
        extract_with_progress(zip_path, extract_path)
        print("\nExtraction completed successfully!")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    main() 