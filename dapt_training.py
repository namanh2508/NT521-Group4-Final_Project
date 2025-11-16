# dapt_training.py

import os
import pandas as pd
from datasets import Dataset
from transformers import RobertaTokenizerFast, RobertaForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from tqdm import tqdm
import torch
torch.backends.cudnn.benchmark = True
# --- CẤU HÌNH ---
# 1. Đường dẫn đến file dữ liệu huấn luyện SPoC
SPOC_TRAIN_PATH = './data/spoc/train/spoc-train.tsv'

# 2. Tên model đầu ra cho giai đoạn DAPT
DAPT_OUTPUT_DIR = './codeBERT-dapt'

# 3. Tham số huấn luyện
DAPT_TRAINING_ARGS = {
    'output_dir': DAPT_OUTPUT_DIR,
    'overwrite_output_dir': True,
    'num_train_epochs': 3,
    'per_device_train_batch_size': 32,      # Tùy chỉnh theo VRAM. Nếu VRAM < 12GB, giảm xuống 4.
    'gradient_accumulation_steps': 1,         # Giả lập batch size = 8 * 2 = 16. Rất hữu ích khi VRAM hạn chế.
    'save_steps': 5000,
    'save_total_limit': 2,
    'prediction_loss_only': True,
    'learning_rate': 4e-5,
    'weight_decay': 0.01,
    'logging_dir': './logs/dapt',
    'logging_steps': 1000,
    'fp16': True,                              # Sử dụng mixed precision để tăng tốc độ và tiết kiệm bộ nhớ
    'dataloader_pin_memory': True,              # Tăng tốc độ truyền dữ liệu từ CPU sang GPU
    'report_to': 'tensorboard',                 # Ghi log để theo dõi trên TensorBoard
}
# --- KẾT THÚC CẤU HÌNH ---

def load_and_preprocess_spoc_data(tsv_path, tokenizer):
    """
    Tải và tiền xử lý dữ liệu từ file spoc-train.tsv một cách tối ưu.
    Sử dụng itertuples() để tăng tốc độ so với iterrows().
    Trả về một list các chuỗi văn bản đã được ghép nối (NL + Code).
    """
    print(f"Đang kiểm tra file dữ liệu tại: {tsv_path}")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {tsv_path}. Vui lòng kiểm tra lại đường dẫn.")
    
    print("Đang tải dữ liệu bằng pandas (có thể mất vài phút với file lớn)...")
    df = pd.read_csv(tsv_path, sep='\t')
    
    # Loại bỏ các dòng có giá trị NaN ở cột 'text' hoặc 'code'
    df.dropna(subset=['text', 'code'], inplace=True)
    
    # Chuyển đổi cột 'text' sang kiểu chuỗi để tránh lỗi
    df['text'] = df['text'].astype(str)
    
    texts = []
    # Sử dụng itertuples() để duyệt DataFrame nhanh hơn nhiều so với iterrows()
    print("Đang tiền xử lý dữ liệu, vui lòng chờ...")
    for row in tqdm(df.itertuples(index=False, name='PandasRow'), total=len(df), desc="Tiền xử lý SPoC"):
        nl_text = row.text
        cpp_code = row.code
        
        # Theo README.md: nếu 'text' rỗng, chỉ dùng 'code'
        if nl_text.strip() == '':
            combined_text = cpp_code
        else:
            # Sử dụng sep_token của tokenizer để đảm bảo nhất quán
            combined_text = f"{nl_text} {tokenizer.sep_token} {cpp_code}"
        
        texts.append(combined_text)
        
    print(f"Đã tải và tiền xử lý xong {len(texts)} mẫu.")
    return texts

def main():
    # Kiểm tra GPU
    if not torch.cuda.is_available():
        print("Cảnh báo: Không phát hiện GPU. Huấn luyện trên CPU sẽ rất chậm!")
    else:
        print(f"Phát hiện GPU: {torch.cuda.get_device_name(0)}")

    # 1. Tải tokenizer (cần thiết cho bước tiền xử lý)
    model_name = "microsoft/codebert-base"
    print(f"Đang tải tokenizer: {model_name}")
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

    # 2. Tải và tiền xử lý dữ liệu (bây giờ truyền tokenizer vào hàm)
    spoc_texts = load_and_preprocess_spoc_data(SPOC_TRAIN_PATH, tokenizer)
    
    # Chuyển đổi list chuỗi thành Dataset object của thư viện datasets
    spoc_dataset = Dataset.from_dict({"text": spoc_texts})

    # 3. Tải model CodeBERT gốc
    print(f"Đang tải model: {model_name}")
    model = RobertaForMaskedLM.from_pretrained(model_name)

    # In thông tin mô hình
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tổng cộng tham số mô hình: {total_params:,}")
    print(f"Số lượng tham số có thể huấn luyện: {trainable_params:,}")

    # 4. Tokenize toàn bộ dữ liệu
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=64)

    print("Đang tokenzie dữ liệu...")
    tokenized_spoc = spoc_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 5. Thiết lập Data Collator cho nhiệm vụ MLM (Masked Language Modeling)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    # 6. Thiết lập tham số huấn luyện
    training_args = TrainingArguments(**DAPT_TRAINING_ARGS)

    # 7. Khởi tạo và chạy Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_spoc,
    )

    print("Bắt đầu huấn luyện DAPT...")
    trainer.train()
    print("Hoàn thành huấn luyện DAPT.")

    # 8. Lưu model và tokenizer đã được fine-tune
    print(f"Đang lưu model và tokenizer vào: {DAPT_OUTPUT_DIR}")
    trainer.save_model(DAPT_OUTPUT_DIR)
    tokenizer.save_pretrained(DAPT_OUTPUT_DIR)
    print("Đã lưu xong.")
    print("Để theo dõi quá trình huấn luyện, hãy chạy: tensorboard --logdir logs/dapt")

if __name__ == '__main__':
    # Tạo thư mục đầu ra nếu chưa có
    os.makedirs(DAPT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(DAPT_TRAINING_ARGS['logging_dir'], exist_ok=True)
    
    main()