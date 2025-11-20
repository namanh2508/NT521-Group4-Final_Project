import torch
from torch.utils.data import DataLoader, Dataset
from transformers import  get_linear_schedule_with_warmup
from torch.optim import AdamW
from config import TRAINING_CONFIG, FG_CODEBERT_PATH, DATA_BASE_PATH
from utils import load_data_from_csv, preprocess_single_data, tokenizer
from model import ExploitGen
from transformers import RobertaConfig

class ExploitGenDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        raw_nl = item['raw_nl']
        raw_code = item['raw_code']
        
        # Tiền xử lý
        processed = preprocess_single_data(raw_nl, raw_code)
        template_nl = processed['template_nl']
        
        # Tokenize
        raw_encodings = self.tokenizer(
            raw_nl, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        temp_encodings = self.tokenizer(
            template_nl, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        code_encodings = self.tokenizer(
            raw_code, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'raw_input_ids': raw_encodings['input_ids'].squeeze(),
            'temp_input_ids': temp_encodings['input_ids'].squeeze(),
            'attention_mask': raw_encodings['attention_mask'].squeeze(),
            'labels': code_encodings['input_ids'].squeeze(),
            'raw_nl': raw_nl,
            'raw_code': raw_code
        }

def main():
    # 1. Tải dữ liệu
    print("Đang tải dữ liệu...")
    python_data = load_data_from_csv(DATA_BASE_PATH, 'python')
    train_dataset = ExploitGenDataset(python_data['train'], tokenizer, TRAINING_CONFIG['max_length'])
    train_dataloader = DataLoader(train_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=True)

    # 2. Khởi tạo mô hình
    print("Đang khởi tạo mô hình...")
    # LƯU Ý: FG_CODEBERT_PATH phải trỏ đến mô hình đã được adaptive pre-training
    config = RobertaConfig.from_pretrained(FG_CODEBERT_PATH)
    model = ExploitGen.from_pretrained(FG_CODEBERT_PATH, config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3. Thiết lập huấn luyện
    optimizer = AdamW(model.parameters(), lr=TRAINING_CONFIG['learning_rate'])
    num_training_steps = len(train_dataloader) * TRAINING_CONFIG['epochs']
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # 4. Vòng lặp huấn luyện
    print("Bắt đầu huấn luyện...")
    for epoch in range(TRAINING_CONFIG['epochs']):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs['loss']
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{TRAINING_CONFIG['epochs']}, Average Loss: {avg_loss:.4f}")
        
        # Lưu checkpoint
        model.save_pretrained(f"./checkpoint-epoch-{epoch+1}")

if __name__ == '__main__':
    main()