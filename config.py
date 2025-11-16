# Cấu hình chung cho dự án
import torch

# Đường dẫn đến dữ liệu
DATA_BASE_PATH = './data'

# Đường dẫn đến mô hình FG-CodeBERT đã được huấn luyện sẵn
# Bạn cần tự huấn luyện (theo mô tả trong bài báo) và lưu vào đường dẫn này
FG_CODEBERT_PATH = './fg_codebert_model'

# Tham số huấn luyện
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 4e-5,
    'epochs': 10,
    'max_length': 64, # Độ dài tối đa cho NL và code
}

# Tham số mô hình
# Sử dụng cấu hình của RoBERTa/CodeBERT
from transformers import RobertaConfig
MODEL_CONFIG = RobertaConfig.from_pretrained('microsoft/codebert-base')