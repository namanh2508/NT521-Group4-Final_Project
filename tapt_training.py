# tapt_training.py

import os
import pandas as pd
from datasets import Dataset
from transformers import RobertaTokenizerFast, RobertaForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from tqdm import tqdm
import torch

# Import các hàm tiền xử lý từ utils.py
from utils import apply_template_parser

# --- CẤU HÌNH ---
# Đường dẫn đến dữ liệu của bạn
DATA_BASE_PATH = './data'

# Đường dẫn đến model đã huấn luyện ở giai đoạn DAPT
DAPT_MODEL_PATH = './codeBERT-dapt'

# Thư mục đầu ra cho model FG-CodeBERT
FG_CODEBERT_OUTPUT_DIR = './fg_codebert_model'

# Tham số huấn luyện 
CONFIG = {
    'max_length': 128,
    'mlm_probability': 0.15,
    'num_train_epochs': 3,
    'per_device_train_batch_size': 32,      # Tùy chỉnh theo VRAM. Nếu VRAM < 12GB, giảm xuống 4.
    'gradient_accumulation_steps': 1,         # Giả lập batch size = 8 * 2 = 16. Rất hữu ích khi VRAM hạn chế.
    'learning_rate': 4e-5,
    'weight_decay': 0.01,
    'save_steps': 5000,
    'save_total_limit': 2,
    'logging_steps': 1000,
    'logging_dir': './logs/tapt',
    'fp16': True,                              # BẮT mixed precision để tăng tốc độ và tiết kiệm VRAM
    'dataloader_pin_memory': True,              # Tăng tốc độ truyền dữ liệu
    'report_to': 'tensorboard',                 # Ghi log để theo dõi trên TensorBoard
}
# --- KẾT THÚC CẤU HÌNH ---

def load_and_preprocess_exploit_data(base_path):
    """
    Tải và tiền xử lý dữ liệu từ các file CSV.
    Đọc thẳng các cột đã được tiền xử lý sẵn (temp_nl, temp_code) để tăng tốc độ.
    Trả về một list các chuỗi văn bản đã được ghép nối.
    """
    all_texts = []
    languages = ['python', 'assembly']

    # Tải tokenizer trước để có sep_token
    print("Đang tải tokenizer từ model DAPT...")
    tokenizer = RobertaTokenizerFast.from_pretrained(DAPT_MODEL_PATH)

    for language in languages:
        train_path = os.path.join(base_path, language, 'train.csv')
        if not os.path.exists(train_path):
            print(f"Cảnh báo: Không tìm thấy file huấn luyện cho {language} tại {train_path}. Bỏ qua.")
            continue
        
        print(f"Đang tải và xử lý dữ liệu cho {language} từ: {train_path}")
        df = pd.read_csv(train_path)
        
        # Sử dụng itertuples() để duyệt DataFrame nhanh hơn nhiều
        for row in tqdm(df.itertuples(index=False, name='PandasRow'), total=len(df), desc=f"Xử lý {language}"):
            # Đọc thẳng từ các cột có sẵn trong file CSV
            raw_nl = row.raw_nl
            raw_code = row.raw_code
            template_nl = row.temp_nl
            template_code = row.temp_code
            
            # Tạo 3 cặp NL-Code theo mô tả trong bài báo
            # 1. raw NL ⊕ raw code
            all_texts.append(f"{raw_nl} {tokenizer.sep_token} {raw_code}")
            # 2. raw NL ⊕ template-argument code
            all_texts.append(f"{raw_nl} {tokenizer.sep_token} {template_code}")
            # 3. template-argument NL ⊕ template-argument code
            all_texts.append(f"{template_nl} {tokenizer.sep_token} {template_code}")
            
    return all_texts

def main():
    # 0. Kiểm tra môi trường
    if not os.path.exists(DAPT_MODEL_PATH):
        raise FileNotFoundError(f"Không tìm thấy model DAPT tại: {DAPT_MODEL_PATH}. Vui lòng chạy giai đoạn DAPT trước.")
    
    if not torch.cuda.is_available():
        print("Cảnh báo: Không phát hiện GPU. Huấn luyện trên CPU sẽ rất chậm!")
    else:
        print(f"Phát hiện GPU: {torch.cuda.get_device_name(0)}")

    # 1. Tải và tiền xử lý dữ liệu
    exploit_texts = load_and_preprocess_exploit_data(DATA_BASE_PATH)
    print(f"Tổng cộng có {len(exploit_texts)} mẫu huấn luyện cho TAPT.")

    # Chuyển đổi list chuỗi thành Dataset object của thư viện datasets
    exploit_dataset = Dataset.from_dict({"text": exploit_texts})

    # 2. Tải tokenizer và model (tiếp tục từ DAPT)
    print(f"Đang tải model và tokenizer từ: {DAPT_MODEL_PATH}")
    tokenizer = RobertaTokenizerFast.from_pretrained(DAPT_MODEL_PATH)
    model = RobertaForMaskedLM.from_pretrained(DAPT_MODEL_PATH)

    # In thông tin mô hình
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tổng cộng tham số mô hình: {total_params:,}")
    print(f"Số lượng tham số có thể huấn luyện: {trainable_params:,}")

    # 3. Tokenize toàn bộ dữ liệu
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=CONFIG['max_length'])

    print("Đang tokenzie dữ liệu...")
    tokenized_exploit = exploit_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 4. Thiết lập Data Collator cho nhiệm vụ MLM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=CONFIG['mlm_probability'])

    # 5. Thiết lập tham số huấn luyện từ CONFIG
    training_args = TrainingArguments(
        output_dir=FG_CODEBERT_OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=CONFIG['num_train_epochs'],
        per_device_train_batch_size=CONFIG['per_device_train_batch_size'],
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
        save_steps=CONFIG['save_steps'],
        save_total_limit=CONFIG['save_total_limit'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        logging_dir=CONFIG['logging_dir'],
        logging_steps=CONFIG['logging_steps'],
        fp16=CONFIG['fp16'],
        dataloader_pin_memory=CONFIG['dataloader_pin_memory'],
        report_to=CONFIG['report_to'],
    )

    # 6. Khởi tạo và chạy Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_exploit,
    )

    print("Bắt đầu huấn luyện TAPT...")
    trainer.train()
    print("Hoàn thành huấn luyện TAPT.")

    # 7. Lưu model và tokenizer cuối cùng (FG-CodeBERT)
    print(f"Đang lưu model và tokenizer vào: {FG_CODEBERT_OUTPUT_DIR}")
    trainer.save_model(FG_CODEBERT_OUTPUT_DIR)
    tokenizer.save_pretrained(FG_CODEBERT_OUTPUT_DIR)
    print("Đã lưu xong.")
    print("Để theo dõi quá trình huấn luyện, hãy chạy: tensorboard --logdir logs/tapt")

if __name__ == '__main__':
    # Tạo thư mục đầu ra nếu chưa có
    os.makedirs(FG_CODEBERT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(CONFIG['logging_dir'], exist_ok=True)
    
    main()