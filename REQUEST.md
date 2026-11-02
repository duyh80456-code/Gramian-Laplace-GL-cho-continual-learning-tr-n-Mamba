# PROMPT CHO CODING AGENT — Gramian-Laplace (GL) cho continual learning trên Mamba (2026-11-02)
Thay thế `SSMCL_agent_prompt_benchmark.md`. Prompt tự đủ: mọi công thức cần cài nằm ở §4.

## 0. Nguyên tắc
1. **Dựng trên code chính thức, không viết lại pipeline.** Clone `https://github.com/zugexiaodui/mamba-cl` (Mamba-CL, Xidian; PyTorch + timm), pin commit, ghi hash. Dùng nguyên data loader, backbone, lịch train, đánh giá, metric của repo. Phương pháp mới chỉ là **một số hạng loss cộng thêm** và **một bước tính thống kê cuối task**, cắm qua hook.
2. Inf-SSM (arXiv 2505.18604, CVPR 2026) không có code công khai ⇒ cài lại theo công thức ở §4.6, đánh dấu `reimplemented=true`.
3. Không thêm cơ chế nào ngoài prompt. Chỗ chưa quy định: chọn phương án đơn giản nhất, ghi `DECISIONS.md`.
4. **Test:** chỉ các test ở §6 (ngắn, float64, CPU, mô hình tí hon). **Không** chạy full pytest, **không** smoke test end-to-end.
5. Không đổi tiêu chí ở §7 sau khi thấy kết quả.

## 1. Cấu hình (theo paper/repo Mamba-CL, không có bước tái tạo riêng)
- Đọc README/config của repo; dùng đúng backbone, siêu tham số, protocol của repo. Theo paper: De-focus Mamba-Large pretrained ImageNet-21k (48 block); chỉ train A, W_B, W_C, W_δ của mỗi SSM và out-proj; còn lại đóng băng; Adam (0.9, 0.999), wd 5e-5; lr backbone 2e-4 (×0.1 ở epoch 25, 40), lr classifier 1e-2; 50 epoch; batch 200; ảnh 192×192 + AutoAugment; η = 0.90 (ImageNet-R 10-split).
- Run `MambaCL` của đợt chạy chính đồng thời là kiểm tái tạo: báo số của nó cạnh số paper (81.67% / forgetting 4.07), không dùng làm cổng dừng.
- Run dài hơn giới hạn phiên (Kaggle 12 giờ): checkpoint mỗi task + resume; không cắt protocol.

## 2. Ký hiệu khớp code Mamba
Mỗi block SSM (một hướng quét; nếu backbone quét nhiều hướng thì làm riêng mỗi hướng): đầu vào lõi x_t ∈ R^{D} (D = d_inner, sau in-proj/conv/SiLU), số trạng thái n (d_state).
- `x_proj`: x_t ↦ [u_t (r_dt chiều); B_t (n); C_t (n)], tức u_t = W_dt x_t, B_t = W_B x_t, C_t = W_C x_t (W_dt, W_B, W_C là các khối hàng của x_proj).
- `dt_proj`: δ_t = softplus(W_up u_t + b) ∈ R^D.
- A = −exp(A_log) ∈ R^{D×n}; Ā^d_t = exp(δ^d_t A^d); h^d_t = Ā^d_t ⊙ h^d_{t−1} + δ^d_t x^d_t B_t; y^d_t = C_tᵀh^d_t (+ D_skip x^d_t).
- Sau lõi: gate SiLU(z), out-proj.

## 3. Arm, setting, seed
**Setting chính:** 10-split ImageNet-R. **Setting phụ** (chỉ khi xong chính): 10-split CIFAR-100. 3 seed (0, 1, 2).
| Arm | Mô tả |
|---|---|
| Seq | repo, không bảo vệ |
| MambaCL | repo, nguyên bản |
| L2 | ½μ‖θ − θ_{k−1}‖² trên tham số được train (neo task trước; từ task 1 neo = pretrained) |
| EWC | Laplace chéo GGN (online, có suy giảm γ) |
| KFAC | K-FAC Laplace (Ritter 2018): mọi ma trận được train (W_dt, W_B, W_C, W_up, out-proj) dùng Kronecker (hiệp phương sai đầu vào ⊗ hiệp phương sai gradient đầu ra); A_log dùng chéo |
| InfSSM | §4.6, `reimplemented=true`; InfSSM+ thêm phạt B |
| **GL** | §4: Laplace với H = μI + Σ_k γ^{K−k} Ĥ_k (online, suy giảm γ) |
| **MambaCL+GL** | projector của MambaCL giữ nguyên + số hạng GL |
| GL(μ=0) | ablation: bỏ prior đẳng hướng |
| GL−A | ablation: khối A dùng chéo GGN thay §4.3 |
| GL−Δ | ablation: khối Δ dùng K-FAC thường thay §4.4 |
**Đợt 1 (chạy ngay, chỉ so baseline với phương pháp của mình):** `MambaCL` (baseline, repo nguyên bản) vs `GL` vs `MambaCL+GL`, setting chính, **chỉ seed 0**, protocol đầy đủ. Các arm còn lại (L2, EWC, KFAC, InfSSM, ablation) chỉ cài/chạy ở đợt 2 sau khi đợt 1 được duyệt; không cần cài chúng trước.

Chọn siêu tham số (đợt 1): `MambaCL` dùng η của paper. `GL`, `MambaCL+GL`: λ ∈ {3 giá trị}, μ, γ cố định hợp lý (ghi DECISIONS.md), chọn λ trên **validation** (10% train mỗi task) bằng 3 task đầu, rồi chạy đủ 10 task với λ đã chọn.

## 4. Công thức GL (cài đúng như dưới)
Thống kê tính ở cuối task k, tại tham số cuối task, trên ≤ 2000 mẫu train của task k (không lưu ảnh). Mọi tổng theo t là theo token của chuỗi; kỳ vọng theo mẫu. Phạt: L_GL = ½λ Σ_block δθᵀĤδθ, δθ = θ − θ_{k} (neo online).

### 4.1 Metric hạ nguồn
w^d_t = đường chéo của GᵀH_oG tại output lõi y^d_t, với G = ∂(logits)/∂y, H_o = diag(p) − ppᵀ. Ước lượng không thiên lệch bằng 1 mẫu Monte Carlo: lấy ŷ ~ p, w^d_t ≈ (∂ log p_ŷ/∂y^d_t)² (giống Fisher thực nghiệm theo nhãn lấy mẫu). Dùng w^d_t trong 4.2–4.4.

### 4.2 Khối W_C, W_B (Kronecker, trọng số output)
- W_C: Ĥ_C = c_C · Σ_x ⊗ P_w, Σ_x = E Σ_t x_t x_tᵀ, P_w = E Σ_t Σ_d w^d_t h^d_t h^{dᵀ}_t (n×n).
- W_B: Ĥ_B = c_B · Σ_z ⊗ Q_w, z_t = Σ_d δ^d_t x^d_t x_t (D), Σ_z = E Σ_t z_t z_tᵀ; Q_w: đệ quy lùi theo t cho mỗi kênh d: Q^d_s = w^d_s C_sC_sᵀ + diag(Ā^d_{s+1}) Q^d_{s+1} diag(Ā^d_{s+1}), Q_w = E Σ_{s,d} Q^d_s / (số token).
- Hiệu chỉnh vết: c_C = E Σ_t Σ_d w^d_t ‖x_t‖²‖h^d_t‖² / tr(Σ_x ⊗ P_w); c_B tương tự với z và tổng vết của Q.
- Phạt dạng vết: δW_C ∈ R^{n×D}: δθᵀĤδθ = c·tr(δW_C Σ_x δW_Cᵀ P_w) (không dựng Kronecker đầy đủ).

### 4.3 Khối A (Hadamard)
Tham số hóa A_log; δA = A ⊙ δA_log. Với mỗi kênh d: Ĥ_A^d = E Σ_s (q^d_s q^{dᵀ}_s) ⊙ Q^d_s, q^d_s = δ^d_s Ā^d_s ⊙ h^d_{s−1} (n). Tích lũy theo luồng (cần Q^d_s từ đệ quy lùi: chạy lùi trên cùng chuỗi). Phạt Σ_d δA^{dᵀ} Ĥ_A^d δA^d.

### 4.4 Khối Δ (dt hạng thấp — dạng chính xác, rẻ)
Vận tốc trạng thái v^d_t = A^d ⊙ Ā^d_t ⊙ h^d_{t−1} + x^d_t B_t; κ^d_t = v^{dᵀ}_t Q^d_t v^d_t; s^d_t = σ(pre^d_t)² κ^d_t (σ = sigmoid, đạo hàm softplus).
- W_up (D×r_dt), hàng d tác động lên u_t: Ĥ_up^d = E Σ_t s^d_t u_t u_tᵀ (r_dt×r_dt mỗi kênh; r_dt ≈ D/16 nên rẻ). Bias b: E Σ_t s^d_t.
- W_dt (r_dt×D): δδ^d_t = σ(·)(W_up δW_dt x_t)_d ⇒ Ĥ_dt = c · Σ_x ⊗ (W_upᵀ S̄ W_up), S̄ = diag(E Σ_t s^d_t)/(số token) (Kronecker; hiệu chỉnh vết như 4.2).

### 4.5 Phần còn lại
out-proj: K-FAC chuẩn (hiệp phương sai đầu vào ⊗ hiệp phương sai gradient đầu ra theo nhãn lấy mẫu), như arm KFAC. Bỏ số hạng chéo giữa các block (đã chứng minh: tổng các block chặn trên mức hỏng thật trong hệ số L).

### 4.6 Inf-SSM (cài lại)
Với A chéo: G(A,C; A′,C′) = CᵀC′ ⊙ 1/(1 − a a′ᵀ) (a = đường chéo Ā, sau Soft-Normalization của Ā, C theo paper; trạng thái lấy trung bình theo chiều ngoài như paper mô tả cho S6). Khoảng cách chordal d² = 2n − 2‖G₁^{-1/2} G₃ G₂^{-1/2}‖²_F với G₁ = G(old,old), G₂ = G(new,new), G₃ = G(old,new). Loss L = L_cls + λ Σ_block d². InfSSM+: thêm γ‖B − B_old‖²_F. Mọi chi tiết không rõ ⇒ ghi DECISIONS.md.

## 5. Chẩn đoán (rẻ, chỉ đo, không ảnh hưởng train)
Cuối task 1, seed 0: trên 64 mẫu tạm, 50 nhiễu ngẫu nhiên mỗi khối, Spearman giữa δθᵀĤδθ và GGN thật (JVP tới logits, metric H_o) cho GL, KFAC, EWC, và proxy Mamba-CL (chỉ thừa số đầu vào). Báo theo khối (C, B, A, up, dt, out-proj) và trung bình theo block sâu/nông.

## 6. Test (chỉ những test này)
1. Lõi SSM tí hon (D = 6, n = 4, r_dt = 2, T = 20, float64): δθᵀĤ_Cδθ (trước hiệu chỉnh vết, w ≡ 1) = ‖Δy‖² chính xác khi chỉ đổi W_C (sai số ≤ 1e-8).
2. Đệ quy Q khớp tổng tường minh Σ_{t≥s} ΦᵀCCᵀΦ.
3. Ĥ_up^d khớp ‖Δy‖² bậc một (JVP) khi chỉ đổi hàng d của W_up (sai số tương đối ≤ 1e-6, w ≡ 1, khi thay Q bằng tổng tường minh).
4. λ = 0 ⇒ trùng bitwise với arm nền tương ứng (Seq hoặc MambaCL) trong 20 bước.
5. InfSSM: d² = 0 khi old = new; d² bất biến khi hoán vị trạng thái.

## 7. Tiêu chí (đăng ký trước, ghi PREREGISTRATION.md trước khi chạy)
Δ ghép cặp theo seed trên độ chính xác cuối (AA). Đợt 1 (chỉ seed 0) là kết quả sơ bộ: báo số, chưa đánh giá tiêu chí ≥ 2/3 seed.
- **Claim chính (cộng thêm):** MambaCL+GL > MambaCL và GL > L2 ở ≥ 2/3 seed.
- **Claim phụ:** GL > InfSSM và GL > EWC ở ≥ 2/3 seed.
- **Đóng góp riêng SSM:** GL > KFAC ≥ 0.5 điểm ở ≥ 2/3 seed; báo GL−A, GL−Δ (khối A dự kiến ít tác dụng — chỉ báo cáo).
- Bộ nhớ thống kê (số float) và GPU-giờ của mọi arm: chỉ báo cáo, không claim trước.

## 8. Thứ tự chạy và báo cáo
1. Test §6 → đợt 1: ba arm, seed 0, protocol đầy đủ → dừng, báo.
2. Đợt 2 (chỉ khi được duyệt): seed 1–2 và các arm còn lại ở §3.
4. Setting phụ chỉ khi được duyệt.
Báo cáo: commit repo + commit code mới, DECISIONS.md, kết quả test §6, bảng arm × seed (AA, AIA, forgetting), chẩn đoán §5, bộ nhớ, GPU-giờ, mọi `reimplemented`. Dừng sau mỗi bước để chờ duyệt.