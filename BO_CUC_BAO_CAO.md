# BỐ CỤC BÁO CÁO THỰC NGHIỆM HPC

**Tên đề tài:** Nghiên cứu ứng dụng thuật toán K-means trong phân đoạn ảnh màu RGB: Cài đặt song song và đánh giá hiệu năng trên CPU và GPU.

**Môn học:** HPC — Tính toán hiệu năng cao.  
**Căn cứ:** `Huong_dan_bo_cuc_bao_cao_HPC_20261.docx` và thông tin trong `AGENTS.md`.

> Đây là đề cương để triển khai báo cáo, chưa phải báo cáo kết quả. Thông tin phần cứng, dữ liệu, tham số và số liệu cần được điền theo thực nghiệm thực tế. Chưa có phiếu giao đề tài chi tiết nên các quy mô ảnh và giá trị K bên dưới là đề xuất, cần đối chiếu lại với phiếu giao đề tài.

## Quy mô và phân bổ nội dung

Báo cáo gồm Mở đầu, 4 chương, Kết luận và hướng phát triển, Tài liệu tham khảo, Phụ lục. Nội dung chính dài **30–35 trang**, không tính trang đầu, tài liệu tham khảo và phụ lục; Chương 1 không quá 10 trang.

| Phần | Số trang dự kiến | Trọng tâm |
|---|---:|---|
| Mở đầu | 2 | Lý do chọn đề tài, mục tiêu, phương pháp |
| Chương 1 | 5 | Bài toán, K-means, khả năng ứng dụng HPC |
| Chương 2 | 10 | Thiết kế và cài đặt bốn phiên bản |
| Chương 3 | 4 | Nút thắt và tối ưu có kiểm chứng |
| Chương 4 | 9 | Thực nghiệm, hiệu năng, kiểm chứng đầu ra |
| Kết luận và hướng phát triển | 1 | Kết quả chính, hạn chế, hướng mở rộng |
| **Tổng** | **31** | Chương 2 và Chương 4 chiếm khoảng 61,3% nội dung chính |

Có thể điều chỉnh khi dàn trang, nhưng giữ tổng nội dung chính trong khoảng 30–35 trang và ưu tiên ít nhất 60% cho phần thiết kế–cài đặt và thực nghiệm–đánh giá theo khuyến nghị của mẫu.

## Phần đầu báo cáo

- Trang bìa: tên trường/khoa, môn học, tên đề tài, giảng viên, thành viên, lớp và thời gian thực hiện.
- Bảng phân công nhiệm vụ và đóng góp của thành viên nếu mẫu của lớp yêu cầu; có thể đặt chi tiết tại phụ lục.
- Mục lục; danh mục hình; danh mục bảng.
- Danh mục từ viết tắt: HPC, RGB, CPU, GPU, OpenMP, MPI, CUDA, SSE.

## MỞ ĐẦU

### 1. Lý do chọn đề tài

- Nhu cầu chia ảnh màu thành các vùng hoặc nhóm điểm ảnh có đặc trưng tương đồng.
- K-means là phương pháp phân cụm có thể áp dụng lên các vector màu RGB.
- Khi số điểm ảnh, số cụm và số vòng lặp tăng, chi phí tính toán tăng; từ đó đặt ra nhu cầu khai thác CPU nhiều lõi và GPU.
- Làm rõ phạm vi: phân cụm theo màu RGB, không khẳng định kết quả tương đương phân đoạn ngữ nghĩa hay nhận dạng đối tượng.

### 2. Mục tiêu nghiên cứu

- Xây dựng phiên bản K-means tuần tự C/C++ làm mốc đối chiếu.
- Xây dựng các phiên bản OpenMP, MPI và CUDA C++ theo cùng đặc tả thuật toán.
- Phân tích và áp dụng ít nhất một tối ưu HPC, đo hiệu quả trước và sau tối ưu.
- Kiểm chứng kết quả phân cụm và đánh giá thời gian, speedup, hiệu quả song song trên các quy mô ảnh.

### 3. Đối tượng, phạm vi và phương pháp

- Đối tượng: ảnh màu RGB và thuật toán K-means với khoảng cách Euclid trong không gian màu ba chiều.
- Phạm vi: biểu diễn mỗi điểm ảnh bằng ba thành phần R, G, B; chưa đưa tọa độ không gian vào đặc trưng.
- Phương pháp: nghiên cứu thuật toán, cài đặt, kiểm thử và thực nghiệm so sánh có kiểm soát.
- Nêu ngắn gọn cấu trúc bốn chương.

## CHƯƠNG 1. TỔNG QUAN VÀ CƠ SỞ LÝ THUYẾT

### 1.1. Giới thiệu và phát biểu bài toán

- Ứng dụng của phân đoạn theo màu: gom nhóm vùng màu và giảm số màu biểu diễn ảnh.
- Đầu vào: ảnh RGB kích thước W × H, số cụm K, seed hoặc tâm khởi tạo, số vòng lặp tối đa và ngưỡng dừng.
- Quy đổi ảnh thành N = W × H vector điểm ảnh xᵢ = (Rᵢ, Gᵢ, Bᵢ).
- Đầu ra: nhãn cụm của từng điểm ảnh, K tâm cụm, ảnh tái tạo bằng màu tâm cụm và thông tin thực thi.
- Yêu cầu: kết quả hợp lệ, có thể tái lập, xử lý được nhiều quy mô ảnh và so sánh công bằng giữa các phiên bản.

### 1.2. Cơ sở lý thuyết của thuật toán K-means

- Giải thích cụm, tâm cụm và số cụm K.
- Hàm mục tiêu tổng bình phương khoảng cách trong cụm:

  `J = Σᵢ₌₁ᴺ ||xᵢ − μ[labelᵢ]||²`.

- Bước gán cụm: `labelᵢ = argminₖ ||xᵢ − μₖ||²`; thống nhất cách xử lý khi bằng khoảng cách.
- Bước cập nhật: tâm cụm bằng trung bình các vector RGB thuộc cụm đó.
- Quy tắc khởi tạo: chọn một phương pháp chung; lưu seed và tốt nhất lưu trực tiếp bộ tâm ban đầu cho mọi phiên bản.
- Quy tắc cụm rỗng: nêu rõ và áp dụng thống nhất, ví dụ giữ nguyên tâm cũ.
- Điều kiện dừng: số vòng lặp tối đa hoặc độ dịch chuyển tâm cụm dưới ngưỡng đã chọn.
- Quy tắc chuyển tâm thực sang màu ảnh đầu ra: làm tròn và giới hạn miền giá trị màu.
- Sơ đồ khối hoặc giả mã: đọc ảnh → chuẩn bị dữ liệu → khởi tạo → gán cụm → cập nhật tâm → kiểm tra dừng → dựng ảnh → lưu kết quả.
- Ví dụ tính tay với một tập điểm RGB nhỏ và K = 2: trình bày khoảng cách, nhãn và tâm sau một vòng lặp.

### 1.3. Phân tích khả năng ứng dụng HPC

- Chi phí gán cụm theo số vòng lặp I: O(I × N × K × d), với d = 3.
- Chi phí cập nhật tâm và bộ nhớ cho điểm ảnh, nhãn, tâm cụm, bộ tổng cục bộ.
- Các điểm ảnh có thể được gán cụm độc lập khi dùng cùng bộ tâm của vòng lặp hiện tại.
- Tổng RGB và số phần tử mỗi cụm cần được tổng hợp trước khi cập nhật tâm cho vòng lặp tiếp theo.
- Phân tích phần tuần tự, đồng bộ giữa các vòng lặp và giới hạn tăng tốc.
- Thách thức riêng: tranh chấp cập nhật trên OpenMP, truyền thông MPI, truy cập bộ nhớ và truyền CPU–GPU trên CUDA.

### 1.4. Phạm vi và yêu cầu thực nghiệm

- Bốn phiên bản bắt buộc: C/C++ tuần tự, OpenMP, MPI và CUDA C++.
- Cùng dữ liệu, biểu diễn màu, tâm ban đầu, quy tắc cụm rỗng và tiêu chí dừng.
- Ít nhất ba quy mô dữ liệu; tối thiểu ba lượt đo cho mỗi cấu hình.
- Chỉ tiêu chính: tính đúng đắn, thời gian, speedup, hiệu quả song song CPU và hiệu quả tối ưu.
- Chỉ tiêu bổ sung phục vụ kiểm chứng: SSE, số vòng lặp, sai lệch tâm và ảnh đầu ra.
- Ghi lại yêu cầu cụ thể từ phiếu giao đề tài khi có thông tin.

## CHƯƠNG 2. THIẾT KẾ VÀ CÀI ĐẶT THUẬT TOÁN

### 2.1. Phiên bản tuần tự C/C++

- Thiết kế dữ liệu: mảng RGB, mảng nhãn, mảng tâm, tổng RGB và số phần tử từng cụm.
- Giải thích các bước đọc ảnh, khởi tạo, gán nhãn, cập nhật tâm, kiểm tra dừng và tái tạo ảnh.
- Trình bày sơ đồ hoặc giả mã, các hàm chính và đoạn mã then chốt.
- Xử lý dữ liệu không hợp lệ: ảnh không đọc được, K không hợp lệ, ngưỡng và số vòng lặp không hợp lệ.
- Kiểm thử bằng dữ liệu nhỏ biết trước kết quả, ảnh một màu và trường hợp có cụm rỗng.
- Xác định phạm vi thời gian baseline và công cụ đo thời gian thực trôi qua.

### 2.2. Phiên bản song song OpenMP

- Chia vòng lặp gán cụm theo chỉ số điểm ảnh cho các luồng CPU.
- Phân loại dữ liệu shared/private; giải thích các chỉ thị thực sự sử dụng.
- Trình bày cách tổng hợp tổng RGB và số phần tử mà không gây race condition.
- Nêu vị trí đồng bộ trước khi cập nhật hoặc sử dụng bộ tâm mới.
- Trình bày sơ đồ hoặc giả mã và đoạn mã song song quan trọng.
- Thực nghiệm với 1, 2, 4 luồng theo khả năng máy; ghi rõ cách đặt số luồng.

### 2.3. Phiên bản song song MPI

- Chia điểm ảnh cho các tiến trình; xử lý đúng khi N không chia hết cho số tiến trình.
- Tiến trình gốc đọc dữ liệu và khởi tạo tâm; mô tả việc phân phối điểm ảnh và tâm ban đầu.
- Mỗi tiến trình gán cụm và tính tổng RGB, số phần tử trên phần dữ liệu của mình.
- Tổng hợp thống kê để cập nhật cùng một bộ tâm và thống nhất quyết định dừng.
- Giải thích các lệnh phù hợp với cài đặt thực tế, chẳng hạn `MPI_Bcast`, `MPI_Scatterv`, `MPI_Allreduce`, `MPI_Gatherv`.
- Thu nhãn về tiến trình gốc và dựng ảnh đầu ra; kèm sơ đồ trao đổi dữ liệu hoặc giả mã.
- Đo thời gian bao phủ các tiến trình; dùng thời gian lớn nhất giữa các tiến trình khi tổng hợp thời gian tính toán.
- Nêu rõ chạy trên một máy hay nhiều nút, số tiến trình và chi phí truyền thông.

### 2.4. Phiên bản GPU CUDA C++

- Ánh xạ công việc lên thread, block, grid; ví dụ một thread xử lý một điểm ảnh khi gán cụm.
- Quản lý bộ nhớ cho ảnh, nhãn, tâm và thống kê cụm trên GPU.
- Thiết kế kernel gán nhãn, tổng hợp thống kê và bước cập nhật tâm theo cài đặt thực tế.
- Xử lý chỉ số vượt N, cụm rỗng, đồng bộ và kiểm tra lỗi CUDA.
- Sơ đồ luồng xử lý: chuẩn bị → truyền lên GPU → lặp xử lý → truyền kết quả về → dựng ảnh.
- Trình bày giả mã và đoạn mã then chốt; ghi cấu hình block/grid được sử dụng.
- Phân biệt thời gian các kernel trong toàn bộ quá trình lặp với thời gian tổng gồm truyền dữ liệu; ghi rõ nơi cấp phát và đồng bộ nằm trong phép đo.

### 2.5. So sánh thiết kế bốn phiên bản

| Nội dung | Tuần tự C/C++ | OpenMP | MPI | CUDA C++ |
|---|---|---|---|---|
| Đơn vị xử lý | Một luồng | Luồng CPU | Tiến trình | Thread GPU |
| Phân chia dữ liệu | Toàn bộ điểm ảnh | Các đoạn điểm ảnh | Các phần dữ liệu cục bộ | Các điểm ảnh trên grid |
| Tổng hợp tâm | Cộng tuần tự | Tổng hợp giữa luồng | Phép tổng hợp MPI | Tổng hợp trên GPU/CPU theo thiết kế |
| Bộ nhớ | RAM | RAM dùng chung | Không gian nhớ riêng mỗi tiến trình | RAM và bộ nhớ GPU |
| Chi phí cần phân tích | Các bước tuần tự | Tạo luồng, đồng bộ, tranh chấp | Phân phối và tổng hợp dữ liệu | Truyền dữ liệu, gọi kernel, đồng bộ |

Nhận xét ưu điểm, hạn chế và quy mô phù hợp dựa trên thiết kế; kết luận về tốc độ chỉ đưa ra sau thực nghiệm.

## CHƯƠNG 3. PHÂN TÍCH VÀ TỐI ƯU HÓA HIỆU NĂNG HPC

### 3.1. Xác định nút thắt hiệu năng

- Tách thời gian gán cụm, tổng hợp/cập nhật tâm và các chi phí đồng bộ hoặc truyền dữ liệu.
- OpenMP: kiểm tra tranh chấp khi nhiều luồng cập nhật cùng cụm, chi phí đồng bộ và băng thông bộ nhớ.
- MPI: kiểm tra chi phí tổng hợp thống kê mỗi vòng lặp và mất cân bằng tải.
- CUDA: kiểm tra chi phí truyền CPU–GPU, phép cộng đồng thời và cách truy cập bộ nhớ.
- Dùng số đo để xác định nút thắt; phân biệt quan sát đã đo với giả thuyết cần kiểm tra.

### 3.2. Lựa chọn giải pháp cải tiến

- Chọn ít nhất một cải tiến có thể triển khai và đo được.
- Đề xuất ưu tiên: dùng tổng RGB và bộ đếm riêng cho từng luồng OpenMP, sau đó gộp cuối vòng lặp để giảm tranh chấp cập nhật.
- Nếu chọn MPI: gộp thống kê vào ít lần tổng hợp hơn hoặc giảm truyền dữ liệu không cần thiết.
- Nếu chọn CUDA: giữ dữ liệu trên GPU qua các vòng lặp hoặc tổng hợp cục bộ theo block khi số đo cho thấy phù hợp.
- Giải thích nút thắt được xử lý, chi phí bổ sung và ảnh hưởng đến tính đúng đắn.
- Chỉ trình bày như kết quả hoàn thành đối với tối ưu đã thực sự cài đặt và đo đạc.

### 3.3. Cài đặt và đánh giá trước/sau tối ưu

- Mô tả thay đổi bằng giả mã hoặc đoạn mã ngắn.
- Dùng cùng dữ liệu, tâm ban đầu, tham số, tài nguyên và phạm vi thời gian.
- Kiểm tra kết quả đầu ra trước khi so sánh tốc độ.
- Bảng: cấu hình, thời gian trước, thời gian sau, tỷ số `T_trước / T_sau`, số vòng lặp và sai lệch kết quả.
- Đồ thị so sánh trước/sau theo quy mô ảnh.
- Giải thích cả trường hợp cải tiến không hiệu quả hoặc làm chậm chương trình.

## CHƯƠNG 4. THỰC NGHIỆM, ĐÁNH GIÁ VÀ KIỂM CHỨNG

### 4.1. Môi trường thực nghiệm

- CPU: tên, số lõi vật lý/luồng, RAM; hệ điều hành.
- Công cụ: trình biên dịch, cờ biên dịch/tối ưu, phiên bản hỗ trợ OpenMP và thư viện MPI.
- GPU: tên, dung lượng bộ nhớ, CUDA Toolkit và driver.
- Nếu dùng Google Colab: ghi loại GPU thực tế, môi trường và thời điểm chạy do tài nguyên có thể khác giữa các phiên.
- Ghi số luồng, số tiến trình, block/grid và cách giới hạn tài nguyên.
- Nếu CPU và GPU được đo trên các máy khác nhau, công khai cấu hình và giới hạn của phép so sánh speedup.

### 4.2. Bộ dữ liệu và kịch bản thử nghiệm

- Nguồn ảnh, quyền sử dụng, định dạng, kích thước và số điểm ảnh; minh họa ảnh gốc.
- Mô tả chuyển đổi về RGB, xử lý alpha nếu có, thay đổi kích thước và miền giá trị màu.
- Ít nhất ba quy mô; đề xuất ảnh nhỏ, trung bình, lớn, chẳng hạn 256 × 256, 1024 × 1024, 2048 × 2048, tùy phiếu giao đề tài và tài nguyên.
- Giá trị K đề xuất: 4, 8, 16; chốt bộ tham số phù hợp trước khi chạy và ghi rõ cấu hình nào đã đo.
- Lưu seed, bộ tâm ban đầu, ngưỡng dừng, số vòng lặp tối đa và quy tắc xử lý cụm rỗng.
- Mỗi cấu hình chạy tối thiểu ba lượt; báo cáo thời gian trung bình và độ phân tán, đồng thời giữ số liệu từng lượt.
- Nếu có lượt chạy khởi động, mô tả riêng và loại khỏi các lượt đo chính theo quy tắc thống nhất.
- Nêu phạm vi đo: tính toán, truyền dữ liệu, cấp phát, đọc/ghi ảnh có được tính hay không; các phép so sánh dùng phạm vi tương ứng.
- Báo cáo số vòng lặp thực tế; nếu khác nhau do sai số số thực, phân tích ảnh hưởng đến thời gian và kết quả.

### 4.3. Kết quả thời gian và hiệu năng

- Bảng thời gian tuần tự; OpenMP với 1/2/4 luồng; MPI với 1/2/4 tiến trình theo khả năng máy.
- CUDA báo cáo riêng tổng thời gian kernel và tổng thời gian có truyền dữ liệu; xác định rõ cách cộng thời gian kernel.
- Bảng và biểu đồ thời gian theo N và K; nhận xét cụ thể sau mỗi bảng/hình.
- Bảng ghi kết quả dự kiến có các cột sau, chỉ điền sau khi đo:

| Ảnh / N | K | Phiên bản | Luồng / tiến trình / block | Vòng lặp | Phạm vi đo | Thời gian TB (ms) | Độ lệch chuẩn (ms) |
|---|---|---|---|---|---|---|---|
| [Điền] | [Điền] | [Điền] | [Điền] | [Điền] | [Điền] | [Điền] | [Điền] |

### 4.4. Đánh giá speedup và hiệu quả song song

- OpenMP/MPI: `S(p) = T_s / T_p`; `E(p) = S(p) / p × 100%`, với p là số luồng hoặc tiến trình.
- CUDA: `S_GPU = T_s / T_GPU`; ghi rõ dùng thời gian kernel hay tổng thời gian và baseline tương ứng.
- Không tính hiệu quả song song GPU bằng cách chia speedup cho số CUDA threads.
- Vẽ đồ thị speedup theo số luồng/tiến trình và theo quy mô ảnh.
- Phân tích ảnh hưởng của N, K, số vòng lặp và tài nguyên xử lý.
- Giải thích trường hợp speedup nhỏ hơn 1 hoặc GPU chậm hơn CPU dựa trên số đo.

### 4.5. Kiểm chứng kết quả và đối chứng độc lập

- Dùng tập RGB nhỏ tính tay hoặc có kết quả tham chiếu để đối chiếu bốn phiên bản.
- Kiểm tra nhãn nằm trong miền hợp lệ, tổng số điểm ảnh các cụm bằng N, tâm hợp lệ và ảnh đầu ra đúng kích thước.
- So sánh SSE, tâm cụm, nhãn và số vòng lặp; công bố ngưỡng sai số trước khi đánh giá.
- Căn chỉnh hoán vị nhãn cụm trước khi so sánh; hai nhãn khác số chưa đủ để kết luận phân cụm khác nhau.
- Giải thích sai lệch số thực do thứ tự cộng/reduction và ảnh hưởng của điểm ảnh có khoảng cách gần bằng nhau.
- Theo mẫu: nếu đề tài thuộc ĐT01–ĐT04, đối chứng với **Weka SimpleKMeans**. Cần xác nhận mã đề tài từ phiếu giao; không tự bỏ yêu cầu Weka chỉ vì đầu vào là ảnh.
- Khi dùng Weka, chuyển tập RGB nhỏ thành các dòng R, G, B; ghi phiên bản, cấu hình khoảng cách, chuẩn hóa, khởi tạo và tiêu chí dừng. Đối chiếu khi các thiết lập tương thích và giải thích phần khác biệt còn lại.
- Minh họa ảnh gốc, ảnh phân đoạn và bảng màu tâm cụm; nhận xét giới hạn của phân cụm chỉ theo màu.
- Không báo cáo accuracy phân đoạn nếu không có nhãn chuẩn và cách ánh xạ phù hợp.

### 4.6. Nhận xét tổng hợp

- Tổng hợp mức độ đúng đắn, khả năng tái lập và giới hạn sai số.
- Đánh giá khả năng mở rộng khi tăng số điểm ảnh và tài nguyên xử lý.
- So sánh bốn phiên bản theo thời gian, chi phí bộ nhớ và chi phí trao đổi dữ liệu.
- Nêu điều kiện/quy mô mà mỗi phiên bản phù hợp dựa trên kết quả đã đo.
- Đánh giá mức độ hoàn thành mục tiêu và các giới hạn của môi trường thực nghiệm.

## KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

- Tóm tắt bài toán, bốn phiên bản đã xây dựng và tối ưu đã thực hiện.
- Nêu kết quả định lượng chính có dẫn chiếu bảng/hình; không khẳng định tăng tốc khi chưa có số đo.
- Hạn chế: nhạy với K và khởi tạo, giới hạn của đặc trưng RGB, chi phí đồng bộ/truyền dữ liệu và tài nguyên phần cứng.
- Hướng phát triển có thể cân nhắc: K-means++, đặc trưng màu kết hợp vị trí, không gian màu khác hoặc xử lý nhiều ảnh; phân biệt rõ với phạm vi đã thực hiện.

## TÀI LIỆU THAM KHẢO

- Tài liệu thuật toán K-means và phân đoạn ảnh thực sự được sử dụng.
- Tài liệu chính thức về OpenMP, MPI, CUDA và Weka phục vụ cài đặt/đối chứng.
- Nguồn bộ ảnh, thư viện đọc/ghi ảnh và công cụ đo nếu sử dụng.
- Trích dẫn nhất quán trong nội dung; bổ sung thông tin xuất bản, phiên bản và ngày truy cập theo yêu cầu của giảng viên.

## PHỤ LỤC

### Phụ lục A. Mã nguồn và hướng dẫn chạy

- Mã nguồn đầy đủ của bốn phiên bản `.cpp`/`.cu` và phần tối ưu.
- Lệnh biên dịch, yêu cầu môi trường, lệnh chạy và ý nghĩa tham số thực tế.
- Cấu trúc thư mục và cách tái lập một cấu hình thực nghiệm.

### Phụ lục B. Dữ liệu và kết quả chi tiết

- Ảnh đầu vào hoặc hướng dẫn tải; dữ liệu RGB nhỏ dùng kiểm chứng.
- Seed, bộ tâm ban đầu và bộ tham số của từng lần thử.
- CSV/Excel kết quả từng lượt, cách tổng hợp, đồ thị và nhật ký chạy.
- Kết quả Weka hoặc đối chứng theo đúng phiếu giao đề tài.

### Phụ lục C. Minh chứng và đóng góp

- Bằng chứng chạy chương trình trên CPU/GPU và thông tin cấu hình máy.
- Ảnh kết quả phân đoạn ở các cấu hình tiêu biểu.
- Bảng phân công và mức đóng góp của các thành viên.

## Đối chiếu thang điểm trong file mẫu

| Tiêu chí | Điểm | Vị trí trình bày |
|---|---:|---|
| Bài toán, dữ liệu và tái lập | 0,75 | Chương 1; Mục 4.2; Phụ lục B |
| Phiên bản C/C++ tuần tự | 0,75 | Mục 2.1 |
| Phiên bản OpenMP | 1,00 | Mục 2.2 |
| Phiên bản MPI | 1,00 | Mục 2.3 |
| Phiên bản CUDA C++ | 1,50 | Mục 2.4 |
| Phân tích và tối ưu HPC | 1,50 | Chương 3 |
| Thực nghiệm, speedup, kiểm chứng | 1,50 | Mục 4.3–4.6 |
| Weka hoặc kết quả tham chiếu theo phiếu giao | 1,00 | Mục 4.5; Phụ lục B |
| Báo cáo, dữ liệu, mã nguồn, vấn đáp | 1,00 | Toàn bộ và phụ lục |
| **Tổng** | **10,00** | |

## Yêu cầu trình bày và bộ hồ sơ nộp

- Mỗi phiên bản có sơ đồ hoặc giả mã và giải thích đoạn mã then chốt; mã đầy đủ đặt trong phụ lục hoặc nộp riêng.
- Mỗi bảng/hình có số thứ tự, tên, đơn vị nếu có và nhận xét trong nội dung.
- Có ít nhất một cải tiến HPC được đo trước/sau.
- Mỗi cấu hình được đo ít nhất ba lượt trong điều kiện đối chiếu nhất quán.
- Không bắt buộc CUDA nhanh hơn CPU; nhận định phải dựa trên số đo.
- Hồ sơ gồm báo cáo Word/PDF, mã nguồn, dữ liệu/seed, CSV/Excel kết quả, đồ thị và hướng dẫn chạy.
