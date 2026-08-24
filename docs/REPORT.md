# Preference Alignment Experiment Report

**Student**: Tran Hieu  
**Date**: 2026-08-24  
**Repository**: K3-Track3-Day22-2a202602030-TranHieu

## 1. Dataset Analysis & Cleaning

### Data Loading Summary
- **Total examples loaded**: `24`
- **Validation issues found**: Line 1 chứa JSON không hợp lệ — dấu ngoặc kép chưa escape trong prompt `"self-attention"` (char 35), gây lỗi `JSONDecodeError: Expecting ',' delimiter`.
- **Cleaning steps taken**:
  1. Sửa line 1 trong `data/sample_preferences.jsonl`: thay `"self-attention"` (unescaped double quotes) bằng `'self-attention'` (single quotes) để tạo JSON hợp lệ.
  2. Implement `load_jsonl()` với line-numbered error messages — khi gặp lỗi JSON hoặc validation, raise `ValueError` kèm số dòng cụ thể (ví dụ: `Line 1: invalid JSON — ...`).
  3. Thêm duplicate prompt detection: dùng `dict` track prompt đã thấy, log `WARNING` nếu trùng.
  4. Cải thiện schema validation trong `schemas.py`: normalize `.strip().lower()` trước khi so sánh `chosen` vs `rejected`, thêm near-duplicate detection bằng `SequenceMatcher` (threshold > 0.95).

### Split Strategy
- **Train/Val Ratio**: `80/20` (20 train, 4 val từ 24 examples)
- **Leakage Prevention**: Group tất cả examples theo `prompt` field → shuffle danh sách unique prompts bằng `random.Random(42)` (deterministic) → phân bổ **toàn bộ group** vào train hoặc val. Đảm bảo không có prompt nào xuất hiện ở cả hai split. Đã verify bằng test `test_split_no_prompt_leakage` với `set.isdisjoint()`.

## 2. Implementation: DPO, ORPO & KTO

Cả ba phương pháp đều được implement.

### Objective Selection
- **Why DPO?**: DPO (Direct Preference Optimization) biến bài toán RLHF thành classification đơn giản, không cần train reward model riêng. Phù hợp cho lab vì công thức rõ ràng và dễ implement.
- **Why ORPO?**: ORPO (Odds Ratio Preference Optimization) kết hợp SFT loss với preference penalty trong một objective duy nhất, không cần reference model.
- **Why KTO?**: KTO (Kahneman-Tversky Optimization) tối ưu trực tiếp theo hàm giá trị lý thuyết triển vọng (Prospect Theory), không bắt buộc dữ liệu theo cặp chặt chẽ.
- **Key Hyperparameters**:
    - `beta`: `0.1` (DPO / KTO — controls deviation from reference policy)
    - `lambda_orpo`: `0.1` (ORPO — weight of odds-ratio penalty)
    - `desirable_weight`: `1.0`, `undesirable_weight`: `1.0` (KTO)

### Numerical Stability
- **Challenges**:
  - `log(σ(x))` (log-sigmoid) có thể gây underflow khi `x` rất âm, hoặc overflow khi tính `exp(-x)` với `x` rất âm
  - ORPO: `odds = p/(1-p)` khi `p → 1` gây division by near-zero; `log(odds)` khi `odds → 0` gây `-inf`
  - KTO: `1 - σ(x)` cần tính qua `σ(-x) = exp(-logaddexp(0, x))` để tránh tràn số
- **Solutions**:
  - **DPO**: Dùng `np.logaddexp(0, -logits)` thay vì naive `-np.log(1/(1+np.exp(-x)))`. `logaddexp` xử lý overflow/underflow tự động bằng log-sum-exp trick
  - **ORPO**: Thêm epsilon `1e-10` vào mẫu số `(1 - p + eps)` và trong `log(odds + eps)` để tránh division by zero và log(0)
  - **KTO**: Dùng hàm mũ âm của `logaddexp` để tính sigmoid value functions một cách ổn định

## 3. Evaluation Results

### Metrics
| Metric | Value |
|---|---|
| Pairwise Accuracy | `100%` (1.0) |
| DPO Mean Loss | `0.6747` |
| ORPO Mean Loss | `1.3710` |
| KTO Mean Loss | `0.9903` |
| Mock Mean Loss | `0.4567` |

### Qualitative Review

**Example 1:**
- **Prompt**: "Explain the concept of 'self-attention' in Transformers."
- **Chosen Response**: "Self-attention allows the model to weigh the importance of different words in the input sequence when processing each word, capturing long-range dependencies."
- **Rejected Response**: "Self-attention is a simpler version of RNNs that uses less memory and is faster to train."
- **Scorer Output**: Chosen `0.6745` > Rejected `0.6288`
- **Model Preference**: ✅ Correct

**Example 6:**
- **Prompt**: "What is the vanishing gradient problem?"
- **Chosen Response**: "The vanishing gradient problem occurs in deep neural networks where gradients become extremely small during backpropagation, making it difficult to train earlier layers."
- **Rejected Response**: "The vanishing gradient problem is caused by using too many layers in the network, which makes the model too complex."
- **Scorer Output**: Chosen `0.7300` > Rejected `0.6250`
- **Model Preference**: ✅ Correct

**Example 11:**
- **Prompt**: "What is the difference between a CNN and a Transformer?"
- **Chosen Response**: "CNNs use convolutional layers to process grid-like data like images, capturing local patterns, while Transformers use self-attention to process sequential data like text, capturing global dependencies."
- **Rejected Response**: "CNNs are used for text data, while Transformers are used for image data."
- **Scorer Output**: Chosen `0.6446` > Rejected `0.5146`
- **Model Preference**: ✅ Correct

## 4. Discussion & Failure Modes

- **What went well?**:
  - Deterministic text-quality scorer (`_deterministic_score`) đạt 100% pairwise accuracy trên dataset này. Scorer kết hợp word count (chuẩn hóa lên 50 từ) và vocabulary richness (unique words / total words), phản ánh thực tế là chosen responses thường dài hơn và đa dạng từ vựng hơn rejected responses.
  - Line-numbered error handling phát hiện ngay lỗi JSON ở line 1 khi load data.
  - Split by prompt hoạt động đúng — test `isdisjoint()` confirm không có data leakage.

- **Observed Bias**:
  - Scorer hiện tại **thiên về responses dài hơn** (length bias): `length_score = min(num_words / 50, 1.0)` chiếm 50% trọng số. Trong dataset này, chosen responses luôn dài hơn rejected nên đạt 100%, nhưng với dataset mà rejected dài hơn chosen thì scorer sẽ sai.
  - Scorer **không đánh giá nội dung semantic** — chỉ dựa trên surface-level features. Một response dài nhưng sai vẫn có thể score cao.

- **Safety (Regression Prompts)**:
  - Scorer hiện tại là deterministic text-based, **không phải language model**, nên không thể test trực tiếp với regression prompts (medical advice, uncertainty, etc.).
  - Để xử lý regression prompts đúng cách, cần thay scorer bằng actual LLM và kiểm tra: (1) model từ chối cho medical advice, (2) model thừa nhận uncertainty khi không chắc chắn, (3) model yêu cầu thêm context khi thiếu thông tin.
  - Đây là limitation chính của mock/CPU-only approach — cần GPU + real model cho safety evaluation đầy đủ.
