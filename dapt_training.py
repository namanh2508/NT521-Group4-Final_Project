# dapt_training.py

import os
import pandas as pd
from datasets import Dataset
from transformers import RobertaTokenizerFast, RobertaForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer

# --- CẤU HÌNH ---
# 1. Đường dẫn đến file dữ liệu huấn luyện SPoC
SPOC_TRAIN_PATH = './data/spoc/train/spoc-train.tsv'

# 2. Tên model đầu ra cho giai đoạn DAPT
DAPT_OUTPUT_DIR = './codeBERT-dapt'

# 3. Tham số huấn luyện
DAPT_TRAINING_ARGS = {
    'output_dir': DAPT_OUTPUT_DIR,
    'overwrite_output_dir': True,
    'num_train_epochs': 5,          # Có thể tăng lên 5-10 nếu có đủ tài nguyên
    'per_device_train_batch_size': 8, # Tùy chỉnh theo VRAM của GPU (16 cho 16GB+, 8 cho 12GB)
    'save_steps': 10_000,
    'save_total_limit': 2,
    'prediction_loss_only': True,
    'learning_rate': 4e-5,
    'weight_decay': 0.01,
    'logging_dir': './logs/dapt',
    'logging_steps': 1000,
    'fp16': True, # Sử dụng mixed precision để tăng tốc độ và tiết kiệm bộ nhớ
}
# --- KẾT THÚC CẤU HÌNH ---

def load_and_preprocess_spoc_data(tsv_path):
    
    """
    Tải và tiền xử lý dữ liệu từ file spoc-train.tsv.
    Trả về một list các chuỗi văn bản đã được ghép nối (NL + Code).
    """
    print(f"Đang tải dữ liệu từ: {tsv_path}")
    
    # Tải file TSV bằng pandas
    df = pd.read_csv(tsv_path, sep='\t')
    
    # Loại bỏ các dòng có giá trị NaN ở cột 'text' hoặc 'code'
    df.dropna(subset=['text', 'code'], inplace=True)
    
    # Chuyển đổi cột 'text' sang kiểu chuỗi và thay thế NaN bằng chuỗi rỗng
    df['text'] = df['text'].astype(str)
    
    texts = []
    for index, row in df.iterrows():
        nl_text = row['text']
        cpp_code = row['code']
        
        # Theo README.md: nếu 'text' rỗng, chỉ dùng 'code'
        if nl_text.strip() == '':
            combined_text = cpp_code
        else:
            # Sẽ dùng tokenizer.sep_token để ghép, ở đây tạm dùng </s>
            combined_text = f"{nl_text} </s> {cpp_code}"
        
        texts.append(combined_text)
        
    print(f"Đã tải và tiền xử lý xong {len(texts)} mẫu.")
    return texts

def main():
    # 1. Tải và tiền xử lý dữ liệu
    spoc_texts = load_and_preprocess_spoc_data(SPOC_TRAIN_PATH)
    
    # Chuyển đổi list chuỗi thành Dataset object của thư viện datasets
    spoc_dataset = Dataset.from_dict({"text": spoc_texts})

    # 2. Tải tokenizer và model CodeBERT gốc
    model_name = "microsoft/codebert-base"
    print(f"Đang tải model và tokenizer: {model_name}")
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    model = RobertaForMaskedLM.from_pretrained(model_name)

    # 3. Tokenize toàn bộ dữ liệu
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=128)

    print("Đang tokenzie dữ liệu...")
    tokenized_spoc = spoc_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 4. Thiết lập Data Collator cho nhiệm vụ MLM (Masked Language Modeling)
    # Data collator sẽ tự động che đi 15% token trong mỗi batch
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    # 5. Thiết lập tham số huấn luyện
    training_args = TrainingArguments(**DAPT_TRAINING_ARGS)

    # 6. Khởi tạo và chạy Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_spoc,
    )

    print("Bắt đầu huấn luyện DAPT...")
    trainer.train()
    print("Hoàn thành huấn luyện DAPT.")

    # 7. Lưu model và tokenizer đã được fine-tune
    print(f"Đang lưu model và tokenizer vào: {DAPT_OUTPUT_DIR}")
    trainer.save_model(DAPT_OUTPUT_DIR)
    tokenizer.save_pretrained(DAPT_OUTPUT_DIR)
    print("Đã lưu xong.")

if __name__ == '__main__':
    # Tạo thư mục đầu ra nếu chưa có
    os.makedirs(DAPT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(DAPT_TRAINING_ARGS['logging_dir'], exist_ok=True)
    
    main()