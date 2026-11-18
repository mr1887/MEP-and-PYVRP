# MEP_evolve_VRP

Nghiên cứu tiến hóa heuristic `select_parents` cho Vehicle Routing Problem bằng LLM (Gemini), theo chu trình Reason–Act–Reflect. **Candidate selector được đánh giá bằng HGS trong fork PyVRP của tác giả bài PyVRP+**, không còn dùng GA/OX tự viết làm evaluator.

## 1. Kiến trúc thực thi

1. MEP chọn hai chương trình heuristic cha mẹ.
2. Gemini sinh một chương trình `select_parents(population, rng, cost_evaluator, k=2)`.
3. Mã được kiểm tra cú pháp/chữ ký.
4. Evaluator ghi mã selector vào file tạm và gọi `pyvrp.solve(..., params=SolveParams(custom_selector_path=...))`.
5. HGS tạo và quản lý quần thể nghiệm; tại mỗi vòng, `Population.select()` gọi selector do LLM cung cấp.
6. HGS tự thực hiện crossover, local search, cập nhật penalty và quản lý nghiệm sống sót. Fitness là objective cost của nghiệm khả thi; thấp hơn là tốt hơn.
7. Kết quả qua nhiều seed/instance được trả về MEP để chọn survivor.

Hai cấp độ parent cần phân biệt:
- **Heuristic parents:** hai chương trình Python được MEP chọn làm đầu vào cho Gemini.
- **Solution parents:** hai nghiệm VRP do selector đang được đánh giá chọn bên trong HGS.

## 2. Cấu trúc chính

| File | Vai trò |
|---|---|
| `base_heuristic.py` | Selector Appendix C tái dựng và các baseline seed |
| `prompt_builder.py` | Prompt Reason–Act–Reflect cho `select_parents` |
| `llm_client.py` | Gọi Gemini và trích code |
| `code_sandbox.py` | Kiểm tra cú pháp/chữ ký và compile |
| `hgs_backend.py` | Adapter sửa loader selector trong fork để hỗ trợ built-ins/import phù hợp |
| `fitness.py` | Gọi HGS thực, truyền selector qua custom hook, tính mean objective |
| `population.py` | Quần thể các chương trình heuristic của MEP |
| `evolve.py` | Điều phối các thế hệ MEP |
| `requirements.txt` | Pin PyVRP fork theo commit |

## 3. Cài đặt

Fork của tác giả là PyVRP 0.11-era và có HGS/Population hook; không thay bằng `pip install pyvrp` bản mới. PyVRP chính thức từ 0.13 đã chuyển khỏi kiến trúc GA/HGS cũ, nên dự án pin thẳng commit fork trong `requirements.txt`.

Khuyến nghị Python 3.11 và môi trường ảo sạch:

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Nếu cài đặt fork thất bại ở bước build C++ extensions, cần cài công cụ build C++ tương thích với hệ điều hành (Windows: Visual Studio Build Tools với C++; Linux: compiler toolchain, CMake/Ninja). Không cài PyVRP 0.14 thay thế để né lỗi, vì API đó không có HGS hook mà dự án cần.

## 4. Chạy

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
python evolve.py
```

Tuỳ chỉnh ngân sách qua biến môi trường:

```powershell
$env:NUM_GENERATIONS="2"
$env:OFFSPRING_PER_GENERATION="2"
$env:NUM_EVAL_RUNS="2"
$env:HGS_TIME_LIMIT="3"
python evolve.py
```

Mặc định: 10 thế hệ, 10 offspring/thế hệ, quần thể heuristic tối đa 5; mỗi candidate được chạy 5 seed trên mỗi instance, mỗi lần HGS tối đa 10 giây. Đây là cấu hình mặc định của repo, không đồng nghĩa tái lập đầy đủ thí nghiệm của bài báo.

Có thể đặt danh sách instance bằng `VRP_INSTANCES`, phân tách bằng dấu phẩy, ví dụ `X-n101-k25.vrp,X-n106-k14.vrp`. Đảm bảo các file tồn tại trong thư mục `data/`.

## 5. Lưu ý về selector loader

Fork upstream hiện tại có hai hạn chế trong `Population._safe_import`: AST validator cấm toàn bộ import, và restricted builtins không có đủ hàm Python phổ biến mà selector Appendix C sử dụng. `hgs_backend.py` thay riêng phương thức nạp selector trước khi chạy HGS, kiểm tra import chỉ thuộc namespace PyVRP và cung cấp builtins Python tiêu chuẩn.

Đây **không phải sandbox bảo mật**. `exec()` chạy mã Python trong tiến trình chính; không dùng với mã không đáng tin cậy. Với nghiên cứu nghiêm túc hoặc chạy mã bên ngoài, cần cô lập bằng process/container, giới hạn CPU/RAM/thời gian và quyền truy cập file/network.

## 6. Phạm vi tái lập và giới hạn

- Đã chuyển evaluator sang HGS fork và dùng custom selector hook của `Population`; HGS vẫn giữ crossover, local search, penalty management và survivor management của fork.
- Dự án hiện chỉ tiến hóa `select_parents`, trong khi bài báo còn nghiên cứu `select_survivors` và `update_penalties`.
- Benchmark mặc định hiện chỉ có một instance CVRP. Cần bổ sung các biến thể/instances và tập test độc lập trước khi kết luận tổng quát.
- Ba seed ban đầu (Appendix-C reconstruction, tournament, random) là các baseline do repo chuẩn bị; không khẳng định là bản sao nguyên văn toàn bộ khởi tạo của tác giả.
- Fork HGS được pin ở một commit cụ thể để tránh thay đổi API; cần giữ nguyên commit và cấu hình khi so sánh các heuristic.
- Cần chạy end-to-end trên máy sạch sau khi cài dependencies. Unit tests của repo không thay thế được kiểm thử runtime/native extension trên máy của bạn.
- Kết quả hiện tại nên được mô tả là **tích hợp và đánh giá parent-selector trong HGS của fork PyVRP tác giả**, chưa phải tái lập toàn bộ benchmark và kết quả thực nghiệm của bài báo.

## 7. Đầu ra

- `generated_code/`: code từng offspring.
- `results/checkpoint.json`: checkpoint sau mỗi thế hệ.
- `results/best_select_parents.py`: selector có mean HGS objective tốt nhất.
- `results/final_population.json`: quần thể heuristic cuối.
