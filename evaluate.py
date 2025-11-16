import torch
from torch.utils.data import DataLoader
from transformers import RobertaConfig
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from easy_rouge import rouge_w
from config import TRAINING_CONFIG, FG_CODEBERT_PATH, DATA_BASE_PATH
from utils import load_data_from_csv, preprocess_single_data, tokenizer, apply_template_parser
from model import ExploitGen
from train import ExploitGenDataset

def generate_code(raw_nl, model, tokenizer, device, max_length=64, num_beams=10):
    """Sinh mã khai thác từ một mô tả NL."""
    model.eval()
    
    # 1. Tiền xử lý
    processed = preprocess_single_data(raw_nl, "dummy_code") # code không quan trọng ở bước này
    template_nl = processed['template_nl']
    slot_map = processed['slot_map']

    # 2. Tokenize
    raw_encodings = tokenizer(raw_nl, return_tensors='pt').to(device)
    temp_encodings = tokenizer(template_nl, return_tensors='pt').to(device)

    # 3. Sinh mã có template bằng Beam Search
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=raw_encodings['input_ids'],
            attention_mask=raw_encodings['attention_mask'],
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            num_return_sequences=1
        )
    
    template_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # 4. Hậu xử lý: thay thế placeholder
    final_code = template_code
    for placeholder, original_token in slot_map.items():
        final_code = final_code.replace(placeholder, original_token)
        
    return final_code

def evaluate_model(model, dataloader, tokenizer, device):
    """Đánh giá mô hình trên toàn bộ tập dữ liệu."""
    model.eval()
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Lấy raw NL từ batch (cần cách để lấy lại, hiện tại dataloader chỉ có ids)
            # Giải pháp: tạo một dataloader khác chỉ chứa NL hoặc sửa ExploitGenDataset
            # Đơn giản ở đây, giả sử ta có list các raw NL
            pass # Cần điều chỉnh để lấy lại NL gốc

    # Giả sử ta có predictions và references
    # predictions = [...]
    # references = [...]

    # Tính các chỉ số
    # BLEU-4
    bleu_score = corpus_bleu([[ref.split()] for ref in all_references], [pred.split() for pred in all_predictions], weights=(0.25, 0.25, 0.25, 0.25))
    
    # ROUGE-W
    rouge_score = rouge_w(all_predictions, all_references)
    
    # Exact Match
    exact_match = sum(1 for p, r in zip(all_predictions, all_references) if p == r) / len(all_predictions)
    
    print(f"BLEU-4: {bleu_score:.4f}")
    print(f"ROUGE-W: {rouge_score:.4f}")
    print(f"Exact Match: {exact_match:.4f}")

def main():
    # 1. Tải dữ liệu và mô hình
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = RobertaConfig.from_pretrained(FG_CODEBERT_PATH)
    model = ExploitGen.from_pretrained(FG_CODEBERT_PATH, config=config)
    model.to(device)

    # 2. Suy luận cho một ví dụ
    sample_nl = "set the variable z to bitwise not x"
    generated_code = generate_code(sample_nl, model, tokenizer, device)
    print(f"NL: {sample_nl}")
    print(f"Generated Code: {generated_code}")

    # 3. Đánh giá trên tập test
    # python_data = load_data_from_csv(DATA_BASE_PATH, 'python')
    # test_dataset = ExploitGenDataset(python_data['test'], tokenizer, TRAINING_CONFIG['max_length'])
    # test_dataloader = DataLoader(test_dataset, batch_size=TRAINING_CONFIG['batch_size'])
    # evaluate_model(model, test_dataloader, tokenizer, device)

if __name__ == '__main__':
    main()