import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaPreTrainedModel
from config import MODEL_CONFIG

class SemanticAttention(nn.Module):
    """Lớp Chú ý Ngữ nghĩa để kết hợp thông tin từ nhiều lớp của CodeBERT."""
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, all_layer_outputs):
        # all_layer_outputs: [batch_size, num_layers, seq_len, hidden_size]
        u = self.tanh(self.fc(all_layer_outputs))
        
        # Tính attention score với vector trung bình của u làm context
        uj = u.mean(dim=1, keepdim=True)
        scores = torch.matmul(u, uj.transpose(-2, -1)).squeeze(-1)
        alphas = torch.softmax(scores, dim=1)
        
        # Tính weighted sum
        alphas_expanded = alphas.unsqueeze(-1)
        weighted_sum = torch.sum(alphas_expanded * all_layer_outputs, dim=1)
        return weighted_sum

class FusionLayer(nn.Module):
    """Lớp Hợp nhất để kết hợp thông tin từ hai encoder."""
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_final = nn.Linear(hidden_size * 3, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, C_raw, C_temp):
        # L2 Normalization
        C_raw_norm = torch.nn.functional.normalize(C_raw, p=2, dim=-1)
        C_temp_norm = torch.nn.functional.normalize(C_temp, p=2, dim=-1)

        G1 = self.tanh(self.fc1(torch.cat([C_raw_norm, C_temp_norm], dim=-1)))
        G2 = self.tanh(self.fc2(torch.cat([C_raw_norm, C_raw_norm - C_temp_norm], dim=-1)))
        G3 = self.tanh(self.fc3(torch.cat([C_raw_norm, C_raw_norm * C_temp_norm], dim=-1)))
        
        G_concat = torch.cat([G1, G2, G3], dim=-1)
        G = self.tanh(self.fc_final(G_concat))
        
        # Residual Connection
        C_bar = G + C_raw_norm
        return C_bar

class ExploitGen(RobertaPreTrainedModel):
    """Mô hình ExploitGen chính."""
    def __init__(self, config):
        super().__init__(config)
        # Tải FG-CodeBERT đã huấn luyện sẵn
        self.raw_encoder = RobertaModel(config)
        self.temp_encoder = RobertaModel(config)
        
        # Đảm bảo hai encoder không chia sẻ tham số
        for param in self.temp_encoder.parameters():
            param.requires_grad = True

        hidden_size = config.hidden_size
        self.semantic_attention = SemanticAttention(hidden_size)
        self.fusion_layer = FusionLayer(hidden_size)
        
        # Lớp Linear để dự đoán token (Language Modeling Head)
        self.lm_head = nn.Linear(hidden_size, config.vocab_size, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Khởi tạo trọng số
        self.post_init()

    def forward(
        self,
        raw_input_ids=None,
        temp_input_ids=None,
        attention_mask=None,
        labels=None,
    ):
        # 1. Encoder
        raw_outputs = self.raw_encoder(
            input_ids=raw_input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True
        )
        temp_outputs = self.temp_encoder(
            input_ids=temp_input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True
        )
        
        # Lấy output từ tất cả các lớp (trừ lớp embedding)
        C_raw_all = torch.stack(raw_outputs.hidden_states[1:], dim=1)
        C_temp_all = torch.stack(temp_outputs.hidden_states[1:], dim=1)
        
        # 2. Semantic Attention
        C_raw_att = self.semantic_attention(C_raw_all)
        C_temp_att = self.semantic_attention(C_temp_all)

        # 3. Fusion Layer
        C_bar = self.fusion_layer(C_raw_att, C_temp_att)
        
        # 4. Output Projection
        sequence_output = self.dropout(C_bar)
        logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        return {'loss': loss, 'logits': logits}

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # Cần chuẩn bị đầu vào cho hàm generate
        return {"input_ids": input_ids}