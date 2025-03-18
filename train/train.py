import os
import json
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
from tqdm import tqdm
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)

class LumbarSpineDataset(Dataset):
    def __init__(self, data_root, csv_file, tokenizer, image_size=448):
        self.data_root = data_root
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.image_size = image_size
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 加载图像
        image_path = os.path.join(self.data_root, row['image_path'])
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size))
        
        # 准备文本
        text = row['text']
        
        # 编码文本
        encoding = self.tokenizer(text, 
                                truncation=True, 
                                max_length=512,
                                padding='max_length',
                                return_tensors='pt')
        
        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

def setup_model_and_tokenizer(config):
    # 从预训练目录加载模型和tokenizer
    pretrained_path = os.path.join('checkpoints', 'pretrained')
    
    logging.info(f"Loading tokenizer from {pretrained_path}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    
    logging.info(f"Loading model from {pretrained_path}")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_path,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    
    # 配置LoRA
    if config['model_config']['use_lora']:
        logging.info("Configuring LoRA")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            **config['model_config']['lora_config']
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer

def train(config):
    # 设置加速器
    accelerator = Accelerator(
        mixed_precision=config['training_config']['mixed_precision'],
        log_with="tensorboard",
        project_dir=os.path.join('logs', 'tensorboard')
    )
    
    # 设置随机种子
    torch.manual_seed(config['training_config']['seed'])
    
    # 加载模型和tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # 准备数据集
    dataset = LumbarSpineDataset(
        data_root='processed_data',
        csv_file='processed_data/train_dataset.csv',
        tokenizer=tokenizer,
        image_size=config['model_config']['image_size']
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=config['training_config']['batch_size'],
        shuffle=True,
        num_workers=config['data_config']['num_workers']
    )
    
    # 设置优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training_config']['learning_rate'],
        weight_decay=config['training_config']['weight_decay']
    )
    
    # 准备训练
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )
    
    # 训练循环
    for epoch in range(config['training_config']['num_epochs']):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}')
        
        for batch in progress_bar:
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['input_ids'],
                    return_dict=True
                )
                
                loss = outputs.loss
                total_loss += loss.detach().float()
                
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), config['training_config']['max_grad_norm'])
                optimizer.step()
                optimizer.zero_grad()
                
                progress_bar.set_postfix({'loss': loss.item()})
        
        # 计算平均损失
        avg_loss = total_loss / len(dataloader)
        logging.info(f'Epoch {epoch + 1} average loss: {avg_loss}')
        
        # 保存检查点
        if accelerator.is_main_process:
            checkpoint_dir = os.path.join('checkpoints', f'epoch-{epoch + 1}')
            accelerator.save_state(checkpoint_dir)
            logging.info(f'Saved checkpoint to {checkpoint_dir}')

def main():
    # 创建必要的目录
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    logging.info("Starting training process...")
    config = load_config()
    train(config)
    logging.info("Training completed!")

if __name__ == '__main__':
    main() 