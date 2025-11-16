import pandas as pd
import os
import re
import spacy
import ast
from transformers import RobertaTokenizerFast

# Tải spaCy model
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
# Tải tokenizer của CodeBERT
tokenizer = RobertaTokenizerFast.from_pretrained('microsoft/codebert-base')

# Định nghĩa các quy tắc regex cho Template Parser (từ Bảng 1 trong bài báo)
TEMPLATE_RULES = {
    'byte_array': re.compile(r'b?\'?\\x[0-9a-f]+\'?'),
    'hex_array': re.compile(r'(?<=\W)[, ]*0x[a-f0-9]+'),
    'hex_token': re.compile(r'(?<=\W)0x[a-f0-9]+'),
    'camel_case': re.compile(r'(?<=\W)[a-z]*[A-Z]\w+'),
    'underline': re.compile(r'(?<=\W)[a-z]+_\w+'),
    'function_name': re.compile(r'def\s+([\w]+)|([\w]+)(?=\s*function|routine)'),
    'bracket_values': re.compile(r'\[(.*?)\]'),
    'quote_values': re.compile(r'([\'"])(.*?)\1'),
    'math_expr': re.compile(r'((\d+\.?\d*|\w+)(\s*[\+\-\*/%]\s*)+(\d+\.?\d*|\w+))')
}

def load_data_from_csv(base_path, language):
    """Tải dữ liệu từ file CSV cho một ngôn ngữ cụ thể."""
    data_splits = {}
    for split in ['train', 'dev', 'test']:
        file_path = os.path.join(base_path, language, f'{split}.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Giả sử các cột là 'nl' và 'code'
            data_splits[split] = df[['raw_nl',  'temp_nl','raw_code', 'temp_code']].to_dict('records')
        else:
            print(f"Cảnh báo: Không tìm thấy file tại {file_path}")
            data_splits[split] = []
    return data_splits

def tokenize_nl(text):
    """Tokenize văn bản tự nhiên."""
    doc = nlp(text)
    return [token.text for token in doc]

def tokenize_asm(code):
    """Tokenize mã Assembly."""
    return re.findall(r'\w+|[^\w\s]', code)

def tokenize_python(code):
    """Tokenize mã Python (phiên bản đơn giản)."""
    try:
        ast.parse(code)
        return code.split() # Fallback đơn giản, AST phức tạp hơn
    except:
        return code.split()

def apply_template_parser(text):
    """
    Áp dụng Template Parser cho một đoạn văn bản.
    Trả về (text_with_placeholders, slot_map).
    """
    slot_map = {}
    placeholder_map = {}
    placeholder_id = 0
    processed_text = text
    
    all_matches = []
    for rule_name, pattern in TEMPLATE_RULES.items():
        for match in re.finditer(pattern, processed_text):
            token = match.group(0)
            if token not in placeholder_map:
                placeholder = f"var{placeholder_id}"
                placeholder_map[token] = placeholder
                slot_map[placeholder] = token
                placeholder_id += 1
            all_matches.append((match.start(), match.end(), placeholder_map[token]))
    
    all_matches.sort(key=lambda x: x[0])
    
    final_text_parts = []
    last_end = 0
    for start, end, placeholder in all_matches:
        final_text_parts.append(processed_text[last_end:start])
        final_text_parts.append(placeholder)
        last_end = end
    final_text_parts.append(processed_text[last_end:])
    
    return "".join(final_text_parts), slot_map

def preprocess_single_data(nl, code):
    """Tiền xử lý một cặp (nl, code)."""
    template_nl, slot_map_nl = apply_template_parser(nl)
    template_code, slot_map_code = apply_template_parser(code)
    
    # Ưu tiên slot_map từ code
    final_slot_map = {**slot_map_nl, **slot_map_code}

    return {
        'raw_nl': nl,
        'template_nl': template_nl,
        'raw_code': code,
        'template_code': template_code,
        'slot_map': final_slot_map
    }