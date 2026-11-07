# Chạy K-means phân đoạn ảnh RGB trên Windows và Linux

Mỗi thư mục `windows/` và `linux/` chứa bốn chương trình:

| File nguồn | Chương trình | Cách xử lý |
|---|---|---|
| `kmeans_sequential.cpp` | `kmeans_sequential` | C/C++ tuần tự |
| `kmeans_openmp.cpp` | `kmeans_openmp` | OpenMP, mặc định yêu cầu 4 luồng |
| `kmeans_mpi.cpp` | `kmeans_mpi` | MPI, số tiến trình do `mpiexec -n` xác định |
| `kmeans_cuda.cu` | `kmeans_cuda` | CUDA C++, chạy thuật toán trên GPU NVIDIA |

Trên Windows, các chương trình có đuôi `.exe`. File `kmeans_common.hpp` chứa phần đọc ảnh, tham số, khởi tạo, cập nhật tâm CPU và cửa sổ hiển thị dùng chung; cần giữ cạnh các file nguồn khi biên dịch. Thuật toán được tự cài đặt, không gọi `cv::kmeans`.

## Chạy từ terminal

Truyền đường dẫn ảnh làm tham số đầu tiên, sau đó là cấu hình. Cửa sổ chỉ hiển thị ảnh kết quả phân đoạn K-means. Bấm **Esc** trong cửa sổ hoặc đóng cửa sổ để thoát. Thêm `--no-gui` nếu chỉ muốn đo hoặc lưu kết quả.

### Linux

```bash
./linux/build/kmeans_sequential "/duong dan/anh.jpg" --k 8 --iterations 100
./linux/build/kmeans_openmp "/duong dan/anh.jpg" --k 8 --threads 4
mpiexec -n 4 ./linux/build/kmeans_mpi "/duong dan/anh.jpg" --k 8
./linux/build/kmeans_cuda "/duong dan/anh.jpg" --k 8 --block-size 256
```

### Windows (PowerShell)

```powershell
.\windows\build\Release\kmeans_sequential.exe "D:\Anh\anh mau.jpg" --k 8 --iterations 100
.\windows\build\Release\kmeans_openmp.exe "D:\Anh\anh mau.jpg" --k 8 --threads 4
mpiexec -n 4 .\windows\build\Release\kmeans_mpi.exe "D:\Anh\anh mau.jpg" --k 8
.\windows\build\Release\kmeans_cuda.exe "D:\Anh\anh mau.jpg" --k 8 --block-size 256
```

### Tham số cấu hình

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| Đường dẫn ảnh | Bắt buộc | Ảnh JPG/PNG/BMP hoặc định dạng OpenCV hỗ trợ |
| `--k` | 8 | Số cụm màu, nhận 1–256; giảm xuống số điểm ảnh nếu ảnh nhỏ hơn K |
| `--iterations` | 100 | Số vòng lặp tối đa, nhận 1–10000 |
| `--threads` | 4 | Số luồng OpenMP, nhận 1–1024; chỉ có tác dụng với bản OpenMP |
| `--block-size` | 256 | Số CUDA threads/block, nhận 32–1024 và phải phù hợp giới hạn GPU; chỉ tác động bản CUDA |
| `--no-gui` | Không bật | Không mở cửa sổ ảnh, phù hợp đo hiệu năng |
| `--output` | Không lưu | Lưu ảnh kết quả vào đường dẫn mới, đuôi ảnh xác định định dạng |
| `--help` | — | Hiển thị hướng dẫn |

Số tiến trình MPI được đặt bằng `mpiexec -n N`; chạy trực tiếp binary MPI chỉ có một tiến trình. Chỉ tiến trình gốc mở cửa sổ và ghi ảnh kết quả.

Ví dụ chạy không mở cửa sổ và lưu kết quả:

```bash
./linux/build/kmeans_cuda "anh.jpg" --k 16 --block-size 128 --no-gui --output "ket_qua_moi.png"
```

Cả bốn phiên bản dùng seed **42** và ngưỡng dịch chuyển tâm **0,1 đơn vị RGB**. Chỉ ảnh hiển thị được thu nhỏ cho vừa màn hình; thuật toán xử lý toàn bộ độ phân giải gốc. Ảnh được giải mã về RGB 8 bit, bỏ kênh alpha nếu có. Đường dẫn có khoảng trắng cần đặt trong dấu ngoặc kép; đường dẫn tiếng Việt trên Windows được đọc bằng Unicode.

Chương trình không tự ghi ảnh. `--output` từ chối ghi đè ảnh đầu vào hoặc file đã tồn tại; chọn tên mới cho mỗi lần lưu.

## Biên dịch trên Linux

Cần CMake ≥ 3.24, trình biên dịch C++17 có OpenMP, OpenCV có HighGUI, MPI và CUDA Toolkit cho bản GPU. Không cần GPU để biên dịch bản CPU. Có GPU/driver NVIDIA phù hợp mới chạy được bản CUDA.

Ví dụ cài phụ thuộc bằng trình quản lý gói của hệ điều hành:

```bash
# Ubuntu/Debian: nhóm thư viện CPU và MPI
sudo apt install build-essential cmake libopencv-dev libopenmpi-dev openmpi-bin

# Arch/CachyOS: bao gồm CUDA Toolkit
sudo pacman -S --needed base-devel cmake opencv openmpi cuda
```

Trên Ubuntu/Debian, cài thêm CUDA Toolkit theo hướng dẫn NVIDIA phù hợp phiên bản hệ điều hành/driver. CUDA Toolkit cần phiên bản host compiler được hỗ trợ; không dùng việc bỏ qua kiểm tra phiên bản như một giải pháp mặc định.

Biên dịch từ thư mục dự án:

```bash
./linux/build.sh
```

Các file thực thi nằm trong `linux/build/`. Nếu thiếu MPI hoặc CUDA Toolkit, CMake cảnh báo và chỉ tạo những bản có đủ công cụ; cần cài đủ rồi cấu hình lại để có cả bốn bản. Nếu CMake đã ghi nhớ CUDA không có, cấu hình lại với `cmake --fresh` hoặc chỉ định trực tiếp trình biên dịch:

```bash
./linux/build.sh -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc
```

Có thể đặt `-DCMAKE_CUDA_ARCHITECTURES=75` cho GTX 1650; với GPU khác, chọn kiến trúc tương ứng. Khi cần host compiler riêng, truyền `-DCMAKE_CUDA_HOST_COMPILER=/duong/dan/g++` ngay từ lần cấu hình đầu hoặc cấu hình lại bằng `cmake --fresh`.

### Bộ công cụ riêng đã chuẩn bị trên máy Linux hiện tại

Để kiểm tra đủ bốn phiên bản mà không dùng quyền quản trị, MPI/CUDA đã được cài vào `~/.local/share/bc-hpc-tools`, host compiler GCC 14 vào `~/.local/share/bc-hpc-compiler`. OpenCV dùng bản có sẵn của hệ thống. Các thư mục này ở ngoài dự án và không được mang theo khi sao chép mã nguồn sang máy khác.

Trong terminal mới, nạp đường dẫn công cụ trước khi dùng `mpiexec` hoặc cấu hình lại bản build:

```bash
source linux/env.sh
./linux/build.sh
mpiexec -n 4 ./linux/build/kmeans_mpi "anh.jpg" --k 8
```

`env.sh` chỉ dùng các công cụ riêng nếu chúng tồn tại. Trên máy khác, cài thư viện/toolkit như hướng dẫn phía trên. Các file nhị phân trong `linux/build/` đã được biên dịch cho môi trường hiện tại; khi chuyển máy nên biên dịch lại.

## Biên dịch trên Windows

Cài các thành phần 64 bit tương thích:

1. Visual Studio 2022, workload **Desktop development with C++**, CMake và Windows SDK.
2. OpenCV cho Windows, có thư viện C++ và HighGUI. Xác định thư mục chứa `OpenCVConfig.cmake`.
3. Microsoft MPI: cài cả **MS-MPI Runtime** để chạy và **MS-MPI SDK** để biên dịch. Đảm bảo `mpiexec.exe` có trong `PATH`.
4. CUDA Toolkit hỗ trợ phiên bản Visual Studio đang dùng, cùng driver NVIDIA phù hợp, để tạo và chạy bản GPU.

Mở **Developer PowerShell for VS 2022** tại thư mục dự án:

```powershell
# Thay đường dẫn bằng vị trí thực tế chứa OpenCVConfig.cmake.
.\windows\build.ps1 -OpenCVDir "C:\opencv\build"
```

Hoặc dùng CMake trực tiếp:

```powershell
cmake -S windows -B windows/build -A x64 -DOpenCV_DIR="C:/opencv/build"
cmake --build windows/build --config Release --parallel
```

Các file `.exe` nằm trong `windows/build/Release/`. Đặt các DLL OpenCV cần thiết cạnh file `.exe` hoặc thêm thư mục `bin` tương ứng của OpenCV vào `PATH`. Nếu bản OpenCV có các DLL phụ thuộc như TBB thì cũng cần cung cấp chúng; MS-MPI Runtime phải được cài trên máy chạy. Sau bước này chạy các lệnh PowerShell ở trên.

Mã nguồn Windows và Linux có cùng thuật toán; phải biên dịch trên đúng nền tảng. File nhị phân Linux không chạy trực tiếp trên Windows. Bộ mã không kèm sẵn bản `.exe` Windows đã được kiểm chứng.

## Kiểm chứng và ý nghĩa thời gian

Bộ kiểm tra dùng Python standard library, tự tạo ảnh thử và so sánh những bản đã biên dịch:

```bash
python tests/test_versions.py linux/build
```

Trên Windows:

```powershell
python tests/test_versions.py windows/build/Release
```

Kiểm tra bao gồm ảnh hai màu có kết quả biết trước, một màu/cụm rỗng, một điểm ảnh, trung bình khi K = 1, ảnh ngẫu nhiên, giới hạn một vòng lặp, đường dẫn tiếng Việt có khoảng trắng, đầu vào sai và bảo toàn ảnh gốc. OpenMP được so với 1/2/4 luồng; CUDA được so với block size 32/128/256/1024; nếu có `mpiexec` trong PATH thì kiểm tra thêm MPI với 1/2/4 tiến trình, kể cả tiến trình không nhận điểm ảnh. Bản chưa được biên dịch được ghi rõ là bỏ qua; nếu bản CUDA đã có nhưng không chạy được thì kiểm tra thất bại.

Cả bốn bản dùng cùng khởi tạo farthest-first với điểm đầu chọn bằng seed 42, cùng quy tắc phá hòa bằng cụm có chỉ số nhỏ nhất, giữ nguyên tâm cụm rỗng và cùng tiêu chí dừng. Tổng RGB và số phần tử dùng số nguyên 64 bit để tránh sai khác thứ tự cộng song song. Nhãn được tính lại theo tâm cuối cùng trước khi dựng ảnh.

Dòng kết quả trên terminal có `pixels`, `k`, `iterations`, `compute_ms`, `sse`; CUDA có thêm `kernel_ms`:

- `compute_ms` của tuần tự/OpenMP: vòng lặp K-means và bước gán nhãn cuối; không gồm đọc ảnh, khởi tạo tâm, dựng/lưu/hiển thị ảnh.
- `compute_ms` của MPI: phân phối điểm ảnh, các vòng lặp, tổng hợp thống kê và thu nhãn; lấy thời gian lớn nhất giữa các tiến trình. Không gồm khởi động `mpiexec`, đọc ảnh, khởi tạo/broadcast tâm ban đầu hay hiển thị.
- `compute_ms` của CUDA: cấp phát/giải phóng bộ nhớ GPU, truyền dữ liệu hai chiều, các vòng lặp, kiểm tra dừng và gán nhãn cuối. Khởi tạo CUDA context và tạo event nằm ngoài phép đo.
- `kernel_ms`: tổng thời gian đo bằng CUDA event quanh cặp kernel gán cụm/cập nhật tâm mỗi vòng lặp và kernel gán nhãn cuối; không tính truyền dữ liệu và các lệnh xóa bộ thống kê. Khoảng event có thể bao gồm khoảng trống giữa hai lần gọi kernel.

Chỉ so sánh speedup khi hiểu đúng các phạm vi trên. Đây là chương trình chạy từng ảnh để xem kết quả; muốn đưa số liệu vào báo cáo cần chạy tối thiểu ba lượt/cấu hình và lưu điều kiện đo. Phép đo không bao gồm toàn bộ thời gian từ lúc gọi chương trình đến khi mở cửa sổ.

OpenMP dùng tổng riêng từng luồng rồi gộp để tránh tranh chấp. CUDA giữ ảnh và tâm trên GPU trong quá trình lặp, nhưng sử dụng atomic toàn cục để tổng hợp cụm: đây là bản cơ sở đúng chức năng; ảnh lớn hoặc ít cụm có thể chịu tranh chấp và chưa đảm bảo nhanh hơn CPU. Cần đo trước khi quyết định tối ưu reduction theo block.

## Tài liệu API đã đối chiếu

- [OpenCV: đọc, hiển thị và ghi ảnh](https://opencv.org/blog/read-display-and-write-an-image-using-opencv/).
- [Open MPI: MPI_Scatterv](https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Scatterv.3.html).
- [NVIDIA CUDA C++ Programming Guide: atomic và mô hình thực thi](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html).
