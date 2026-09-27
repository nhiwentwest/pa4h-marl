# Dữ liệu dùng cho revision (2026-09-23)

Các file lớn và các run mới nằm trên Lightning server, không commit vào repo. Studio:
`/teamspace/studios/this_studio/pa4h_run` (code/run) và
`/teamspace/studios/this_studio/pa4h_data` (workload/power).

| Dữ liệu | Vị trí trên server | SHA-256 |
|---|---|---|
| [HeliosData Earth](https://github.com/S-Lab-System-Group/HeliosData) `cluster_log.csv` | `pa4h_data/helios_cluster_log.csv` | `bb62d7842ee50a093c1f872afa293a8e27500f38dfc5cd1dab3b10c53746b3db` |
| Helios CSV đã chuyển đổi (427.148 GPU job) | `pa4h_data/helios_earth_pod_hourly_jobs.csv` | `7bed08070ef3332d33081fd95327d9254cb1199683a17de2bb06b5c1ddbdc531` |
| [NLR GenAI power profiles, submission 312](https://data.nlr.gov/submissions/312), ZIP gốc | đã giải nén subset, xóa ZIP để tiết kiệm dung lượng | `dcad6de800fb565d850b163902e2eddae48aabd1ed1c7336f9a1cdaf3012f137` |
| NLR training metadata (41 profile) | `pa4h_data/extracted/01_aggregated_datasets/training/metadata.csv` | `385f20516edeef936115d3f4d9a874ac7ee2e36a69863be496e9f017df67b5a8` |
| [Alibaba GPU trace v2020](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020) `pai_job_table.tar.gz` | `pa4h_data/alibaba_v2020/pai_job_table.tar.gz` | `5aad7f7caac501136d14ed6a48e40546f825d7b0617a3a4f337e2348fe0a6cb0` |
| Alibaba v2020 `pai_task_table.tar.gz` | `pa4h_data/alibaba_v2020/pai_task_table.tar.gz` | `cd1d6dc3215d2a8607ccf6b6dd952b5db776df86926c73259fea7c1499ac40e5` |
| Alibaba v2020 `pai_group_tag_table.tar.gz` | `pa4h_data/alibaba_v2020/pai_group_tag_table.tar.gz` | `722fef30b7fb7aa50dabd79155614b5423a9d65cf45a9b26c590d57725423a14` |
| Alibaba v2020 CSV đã chuyển đổi (714.903 GPU job) | `pa4h_data/alibaba_v2020/pod_hourly_jobs.csv` | `97edc8da95c798df99f5c4559be60f5532a9b718174892ac36db7cc231c249c7` |

Helios CSV được tạo bằng `data/convert_helios_to_alibaba_schema.py`. `duration_hours`
được lấy từ cột `duration` (giây) của trace gốc; các job không có thời lượng dương
được loại. Đây là thay đổi quan trọng so với CSV lưu trước đây: adapter cũ đã
gán độ dài một chu kỳ power profile cho mọi job Helios khi không có duration.

Matcher dùng 41 power profile thuộc `01_aggregated_datasets/training`, replay
power 0,2 giây. Profile được ghép theo family và số node; `model_type=unknown`
trong Helios khiến chọn trong toàn bộ registry bằng hash cố định của job ID.
Đây là phép ghép suy diễn, không phải power telemetry đo trực tiếp của Helios.

Alibaba v2020 được chuyển bằng `data/convert_alibaba_v2020.py` với `--tags`: job submission
`pai_job_table.start_time`, GPU yêu cầu = tổng `inst_num * plan_gpu / 100` từ
`pai_task_table`, thời gian thực thi = task GPU kết thúc muộn nhất trừ task GPU
bắt đầu sớm nhất. Chỉ giữ job và task `Terminated` có GPU và thời lượng dương.
`model_type` lấy từ `pai_group_tag_table.workload` khi có; phần lớn job không
được gắn nhãn nên dùng `unknown`. Nhãn `training` trong queue là quy ước cho
GPU jobs được xét, không chứng minh tất cả job gốc là training.
Protocol lịch sử và protocol hiện tại cần được phân biệt. Bộ Alibaba năm seed
300 episode dùng cửa sổ train 1000–1010 ở 3%, 120 bước; năm cửa sổ validation
được ghi trong `reproduction/alibaba_seeds.json`. Helios pilot ban đầu dùng 5%
(38 job train), còn pilot actor đã sửa dùng 50% (358 job train), 180 bước.
Helios giữ validation chính 1010–1020 ở 5%, thêm stress ở 50%; cả hai decoder
được đánh giá. Holdout 1020–1030 vẫn được giữ riêng. Xem `RUN_STATUS.md`.
Các protocol revision không phải replay bit-exact của submission. Adapter ASI
cũ không được dùng cho Alibaba v2020.

Các checkpoint/CSV cũ trong `outputs/recovered_*` là artifact để audit lịch sử,
không phải kết quả của revision.
