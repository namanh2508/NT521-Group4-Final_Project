import torch
from transformers import RobertaConfig
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
try:
    from easy_rouge import rouge_w
    _ROUGE_W_AVAILABLE = True
except Exception:
    # fallback to rouge_score (already in requirements.txt)
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    _ROUGE_W_AVAILABLE = False
from config import TRAINING_CONFIG, FG_CODEBERT_PATH, DATA_BASE_PATH
from utils import load_data_from_csv, preprocess_single_data, tokenizer, apply_template_parser
from model import ExploitGen

def generate_code_simple(raw_nl, model, tokenizer, device, max_length=64):
    """Sinh mã khai thác từ một mô tả NL bằng cách đơn giản hóa."""
    model.eval()
    
    # 1. Tiền xử lý
    processed = preprocess_single_data(raw_nl, "dummy_code") # code không quan trọng ở bước này
    template_nl = processed['template_nl']
    slot_map = processed['slot_map']

    # 2. Tokenize
    raw_encodings = tokenizer(raw_nl, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
    temp_encodings = tokenizer(template_nl, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)

    # 3. Sinh mã bằng cách lấy token có xác suất cao nhất
    with torch.no_grad():
        outputs = model(
            raw_input_ids=raw_encodings['input_ids'],
            temp_input_ids=temp_encodings['input_ids'],
            attention_mask=raw_encodings['attention_mask']
        )
        
        # Lấy token có xác suất cao nhất cho mỗi vị trí
        predicted_ids = torch.argmax(outputs['logits'], dim=-1)
    
    template_code = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    
    # 4. Hậu xử lý: thay thế placeholder
    final_code = template_code
    for placeholder, original_token in slot_map.items():
        final_code = final_code.replace(placeholder, original_token)
        
    return final_code

def evaluate_model_simple(model, test_data, tokenizer, device, max_samples=100):
    """Đánh giá mô hình trên một mẫu của tập dữ liệu."""
    model.eval()
    all_predictions = []
    all_references = []
    
    # Giới hạn số lượng mẫu để đánh giá nhanh
    test_data = test_data[:max_samples]
    
    with torch.no_grad():
        for item in test_data:
            raw_nl = item['raw_nl']
            reference_code = item['raw_code']
            
            # Generate code
            generated_code = generate_code_simple(raw_nl, model, tokenizer, device)
            all_predictions.append(generated_code)
            all_references.append(reference_code)

    # Calculate metrics
    # BLEU-4
    bleu_score = corpus_bleu([[ref.split()] for ref in all_references], [pred.split() for pred in all_predictions], weights=(0.25, 0.25, 0.25, 0.25))
    
    # ROUGE-W
    if _ROUGE_W_AVAILABLE:
        rouge_score = rouge_w(all_predictions, all_references)
    else:
        # Fallback to ROUGE-L if ROUGE-W is not available
        rouge_scores = [scorer.score(ref, pred)['rougeL'].fmeasure for ref, pred in zip(all_references, all_predictions)]
        rouge_score = sum(rouge_scores) / len(rouge_scores)
    
    # Exact Match
    exact_match = sum(1 for p, r in zip(all_predictions, all_references) if p == r) / len(all_predictions)
    
    print(f"Đánh giá trên {len(test_data)} mẫu:")
    print(f"BLEU-4: {bleu_score:.4f}")
    print(f"ROUGE-W: {rouge_score:.4f}")
    print(f"Exact Match: {exact_match:.4f}")
    
    # In một vài ví dụ
    print("\nVí dụ về kết quả sinh ra:")
    for i in range(min(5, len(test_data))):
        print(f"NL: {test_data[i]['raw_nl']}")
        print(f"Reference: {test_data[i]['raw_code']}")
        print(f"Generated: {all_predictions[i]}")
        print("-" * 50)

def main():
    # 1. Tải dữ liệu và mô hình
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        config = RobertaConfig.from_pretrained(FG_CODEBERT_PATH)
        model = ExploitGen.from_pretrained(FG_CODEBERT_PATH, config=config)
        model.to(device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have trained the model or have the correct path in config.py")
        return

    # 2. Suy luận cho một ví dụ
    sample_nl = "set the variable z to bitwise not x"
    try:
        generated_code = generate_code_simple(sample_nl, model, tokenizer, device)
        print(f"NL: {sample_nl}")
        print(f"Generated Code: {generated_code}")
    except Exception as e:
        print(f"Error generating code: {e}")

    # 3. Đánh giá trên tập test
    try:
        python_data = load_data_from_csv(DATA_BASE_PATH, 'python')
        test_data = python_data['test']
        evaluate_model_simple(model, test_data, tokenizer, device)
    except Exception as e:
        print(f"Error evaluating model: {e}")

if __name__ == '__main__':
    main()