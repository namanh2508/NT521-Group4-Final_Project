import os
import pandas as pd
from datasets import Dataset
from transformers import RobertaTokenizerFast, RobertaForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer

# Import hàm tiền xử lý đã viết ở utils.py
# Giả sử file utils.py nằm cùng cấp
from utils import apply_template_parser, tokenize_nl, tokenize_python, tokenize_asm

# 1. Tải và tiền xử lý dữ liệu của bạn
def load_and_preprocess_data(base_path, language):
    all_texts = []
    for split in ['train', 'dev', 'test']:
        file_path = os.path.join(base_path, language, f'{split}.csv')
        if not os.path.exists(file_path):
            continue
        
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            nl, code = row['nl'], row['code']
            
            # Áp dụng Template Parser
            template_nl, _ = apply_template_parser(nl)
            template_code, _ = apply_template_parser(code)
            
            # Tạo 3 cặp NL-Code theo mô tả trong bài báo
            # 1. raw NL ⊕ raw code
            all_texts.append(f"{nl} {tokenizer.sep_token} {code}")
            # 2. raw NL ⊕ template-argument code
            all_texts.append(f"{nl} {tokenizer.sep_token} {template_code}")
            # 3. template-argument NL ⊕ template-argument code
            all_texts.append(f"{template_nl} {tokenizer.sep_token} {template_code}")
            
    return all_texts

# Lấy dữ liệu từ cả Python và Assembly
python_texts = load_and_preprocess_data('./data', 'python')
assembly_texts = load_and_preprocess_data('./data', 'assembly')
all_exploit_texts = python_texts + assembly_texts

print(f"Tổng cộng có {len(all_exploit_texts)} mẫu huấn luyện cho TAPT.")

# Chuyển đổi thành Dataset object
exploit_dataset = Dataset.from_dict({"text": all_exploit_texts})

# 2. Tải tokenizer và model (tiếp tục từ DAPT)
dapt_model_path = "./codebert-dapt"
tokenizer = RobertaTokenizerFast.from_pretrained(dapt_model_path)
model = RobertaForMaskedLM.from_pretrained(dapt_model_path)

# 3. Tokenize dữ liệu
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=128)

tokenized_exploit = exploit_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 4. Thiết lập Data Collator cho MLM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# 5. Thiết lập tham số huấn luyện
training_args = TrainingArguments(
    output_dir="./fg_codebert_model",
    overwrite_output_dir=True,
    num_train_epochs=5,          # Có thể cần nhiều hơn
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    learning_rate=4e-5,
    weight_decay=0.01,
    logging_dir='./logs_tapt',
    logging_steps=1000,
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

# Lưu model và tokenizer cuối cùng (FG-CodeBERT)
trainer.save_model("./fg_codebert_model")
tokenizer.save_pretrained("./fg_codebert_model")