# PA4H-MARL: Lập lịch Gang Job GPU nhận biết Công suất, SLA và Mạng

Hệ thống lập lịch **gang scheduling** cho cụm GPU bằng **Học tăng cường đa tác tử (MARL)** với 4 agent hoạt động theo chuỗi nhân quả cố định **A4 → A2 → A3 → A1**, huấn luyện theo mô hình CTDE (Centralized Training, Decentralized Execution) + PPO/GAE + counterfactual supervision.

- **A4 – Placement**: chọn anchor rack hoặc DEFER cho job đang chờ; rack action được hiện thực hoá **theo từng host** bởi một power-aware host picker (host chỉ được nhận nếu peak công suất dự phóng của rack vẫn trong budget).
- **A2 – Risk localization**: chọn rack có rủi ro vượt power budget.
- **A3 – Victim selection**: chọn job nạn nhân có projected relief dương trên rack bị gắn cờ.
- **A1 – Preemption control**: quyết định WAIT hay PREEMPT (checkpoint–restart).

A2 và A4 dùng **ST-GNN cấp rack** (dilated TCN trên 6 bước lịch sử + edge-aware GATv2 trên graph rack); A1, A3 dùng MLP. Workload lấy từ **Alibaba GPU Cluster Trace v2020** và **HeliosData Earth (SenseTime)**, ghép tất định với **power profile đo thật 0.2 giây (NREL Eagle GPU)**: mỗi decision step 300 giây replay 1.500 mẫu công suất per-node. Mô phỏng chạy trên **CloudSim Plus (Java)**, nối với Python qua **Py4J**; mạng ring all-reduce được mô phỏng trên topology fat-tree dùng chung.

## Cấu trúc repo

```
python/
  gang_env.py              # GangEnv: rack power replay, host-aware picker, SLA, network
  marl_gang_train.py       # Huấn luyện CTDE-PPO 4 agent (+ ST-GNN, counterfactual, dual-ascent SLA)
  eval_gang.py             # Đánh giá: RANDOM/SPREAD/EPOBF/RPA/SAFERES + MARL (argmax & sequence decoder)
  eval_baselines_helios.py # Entry point đánh giá trên trace Helios
  nrel_injection_bridge.py # Dựng job queue tất định từ Alibaba/Helios + NREL profile
  alibaba_nlr_matcher.py   # Match job -> NREL profile (model family + node count, hash tie-break)
  models.py                # Actor MLP, RackSTGNNActor (TCN + GATv2), CentralizedCritic
  resource_metrics.py      # Đo wall-clock/CPU/RAM (Python + JVM bridge)
  gang_observability.py    # TensorBoard + Prometheus (tùy chọn)
  baselines/               # Harness baseline phase-2 (EPOBF, RPA, Oracle/SafeRes look-ahead)
  eval_baselines.py, smoke_*.py, tests/
src/main/java/com/dacn/advanced/
  Py4jBridge.java          # CloudSim gang bridge: submitJob/preemptJob, rack power 0.2s, network
scripts/
  run_gang.sh              # Train Alibaba end-to-end (bridge + train + TensorBoard)
  launch_helios_train.sh   # Train Helios 300 episodes
  run_eval.sh, launch_seed1.sh
demo/                      # Demo terminal đọc artifacts của một run
data/convert_helios_to_alibaba_schema.py
tools/                     # Vẽ figure & diagram cho paper
```

## Chạy

```bash
# 1. Build phần Java (CloudSim Plus + Py4J)
mvn package dependency:copy-dependencies -DskipTests

# 2. Train trên Alibaba (tự khởi động bridge)
bash scripts/run_gang.sh                       # USE_STGNN=1 để bật ST-GNN

# 3. Train trên HeliosData Earth
bash scripts/launch_helios_train.sh

# 4. Đánh giá baselines + 2 decoder MARL trên cùng một replay tất định
PYTHONPATH=python python python/eval_gang.py
```

Dữ liệu không kèm theo repo (xem điều khoản của từng nguồn):
[Alibaba cluster-trace-gpu-v2020](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020) ·
[HeliosData](https://github.com/S-Lab-System-Group/HeliosData) ·
[NREL Eagle GPU Node Metrics](https://data.nrel.gov/submissions/301).
Đặt theo đường dẫn trong `nrel_injection_bridge.py` (`DACN_DATA`, `POD_HOURLY_JOBS`, `NREL_ROOT`).

## Ghi chú tái tạo

Các file `models.py`, `resource_metrics.py`, `gang_observability.py`, `alibaba_nlr_matcher.py`, `baselines/safe_res_la50.py` được **tái tạo lại từ call-site** (bản gốc nằm trên workspace huấn luyện Lightning AI đã hết hạn). Chúng tương thích interface với train/eval script; không đảm bảo trùng bit-exact với checkpoint đã train bằng bản gốc. Các file còn lại là bản gốc từ workspace huấn luyện.
