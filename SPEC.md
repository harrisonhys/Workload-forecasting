# Spesifikasi Teknis

Pasangan dari [PLAN.md](PLAN.md). Dokumen ini mengikat: kalau eksperimen menyimpang dari sini, perbarui dokumen ini dulu.
Bagian yang ditandai **[verifikasi]** adalah fakta yang harus dicek ulang saat setup (versi model, batas konteks, dan semacamnya).

---

## 1. Lingkungan

| Item | Nilai |
|---|---|
| Mesin utama | Apple M3 Pro, 11 core, 18 GB RAM, **CPU-only** (sesuai klaim "murah") |
| Disk | ≈45 GB bebas. Total data mentah yang diunduh ≈2 GB |
| Python | 3.12 via `uv` (Python sistem 3.9 tidak dipakai) |
| Data prep | `uv run scripts/prepare.py` (dependensi dideklarasikan inline, PEP 723) |
| Venv | root (`pyproject.toml`: statsforecast), `envs/fm` (Chronos, TimesFM, TiRex, TTM; torch 2.11), `envs/toto` (torch 2.7, numpy 1.26), `envs/moirai` (uni2ts 2.0, numpy 1.26), `envs/train` (LightGBM, neuralforecast). Pemisahan ini wajib karena pin torch/numpy/gluonts saling bentrok. Setiap run menulis forecast ke `.npz`, jadi venv tidak saling bergantung. |
| Reprodusibilitas | Revisi HF di-pin di `src/fm.py` (dict `R`), `SEED = 42`, `uv.lock` per venv. Antrean penuh: `bash scripts/run_all.sh` |

---

## 2. Dataset

### 2.1 Ringkasan

| ID | Sumber | Unduhan | Resolusi asli | Durasi | Target | Lisensi |
|---|---|---|---|---|---|---|
| `azure2019` | Azure Functions Trace 2019 (Shahrad et al., ATC'20) | 143 MB `.tar.xz` | hitungan invokasi per menit per fungsi | 14 hari | invokasi/menit per **app** | CC-BY |
| `azure2021` | Azure Functions Invocation Trace 2021 (Zhang et al., SOSP'21) | 18 MB `.rar` | per invokasi (end_timestamp, duration) | 14 hari | invokasi/menit per app | CC-BY |
| `alibaba2018` | Alibaba cluster-trace-v2018, `machine_usage` saja | 1,8 GB `.tar.gz` | sampel tak teratur (median 60 dtk, maks ±640 dtk) | 8 hari | rata-rata CPU % per **5 menit** per mesin + rata-rata cluster | lihat repo alibaba/clusterdata |
| `huawei` | Subset curated Toner et al. (sir-lab/time-series-fm-dataset) | 1,1 MB (4 zip) | 5 menit | 30 hari | jumlah request per channel | lihat repo |
| `google2019` *(opsional)* | Google ClusterData 2019 (Borg v3), cell `a` | BigQuery, ekspor ±beberapa MB | jendela 5 menit | 31 hari | total CPU per collection (job) | CC-BY |

Tidak dipakai: `container_usage` Alibaba (29 GB, disk tidak cukup), trace Google penuh (2,4 TiB), dan trace Huawei penuh 2023/2025 (cadangan kalau subset curated terlalu kecil).

### 2.2 Aturan pembentukan deret (diimplementasikan di `scripts/prepare.py`)

**azure2019**
- Agregasi fungsi → app (`HashApp`), dijumlah per menit, lalu hari d01–d14 disambung menjadi 20.160 langkah.
- Hanya app yang muncul di **ke-14 file harian** yang dipakai (app berumur panjang). App yang tidak muncul pada suatu hari dianggap tidak punya invokasi hari itu.
- Seleksi: 50 app dengan volume tertinggi ∪ sampel acak ≤125 app per kelas Syntetos–Boylan (smooth/erratic/intermittent/lumpy), seed 42 → ≤550 deret.
- Meta: `mean_duration_ms` (rata-rata tertimbang `Average × Count` dari file durasi), `trigger` dominan, statistik intermittency.

**azure2021**
- Waktu kedatangan = `end_timestamp − duration`, di-bin per menit, dihitung per `app`. Semua app dipakai.
- Meta: `mean_duration_ms`.

**alibaba2018**
- `machine_usage.tar.gz` di-stream (CSV tak terkompresi tidak pernah ditulis ke disk). Baris dengan `cpu` atau `mem` di luar [0, 100] dibuang.
- Rata-rata CPU per (mesin, bin 5 menit). Bin 1 menit **tidak dipakai** karena sampling tak teratur membuat ±50% bin kosong, dan interpolasi akan menghasilkan deret buatan yang terlalu mudah diprediksi.
- Outage di seluruh trace pada bin 470–534 (±5,4 jam, hari ke-2, <50% mesin melapor) diisi dengan nilai bin yang sama sehari sebelumnya (seasonal fill). Blok ini ada di split train.
- Mesin dengan cakupan <90% (tanpa menghitung blok outage) dibuang. Gap ≤3 bin diinterpolasi linear. Mesin dengan rata-rata CPU <1% (idle/mati) dibuang.
- Seleksi: 200 mesin acak (seed 42) + deret `cluster_mean` (rata-rata semua mesin yang lolos).

**huawei**: CSV curated dipakai apa adanya (sudah 5 menit), satu deret per channel.

**google2019**: hasil kueri §2.5 diubah ke format panjang, dengan bin kosong diisi 0.

### 2.3 Format keluaran

```
data/processed/<ds>.parquet       series_id: category, ts: datetime64[ns], y: float32
data/processed/<ds>_meta.parquet  index=series_id; total, zero_frac, adi, cv2, sbc, selected, freq, length, (+ kolom khusus dataset)
```
`ts` dimulai dari epoch sintetis 2000-01-01 karena semua trace dianonimkan. **Jangan** memakai fitur kalender absolut (hari dalam minggu, hari libur), karena hanya posisi relatif yang bermakna.

Kelas Syntetos–Boylan: ADI = n / n_nonzero, CV² dihitung dari nilai non-nol. Batas: ADI 1,32 dan CV² 0,49.

### 2.4 Langkah

```bash
bash scripts/download.sh            # semua kecuali google; boleh pilih: azure2019 huawei ...
uv run scripts/prepare.py selfcheck
uv run scripts/prepare.py           # azure2019 azure2021 alibaba2018 huawei
```

### 2.5 Google 2019 (opsional, BigQuery) **[verifikasi: belum pernah dijalankan]**

Butuh project GCP milik Anda. **Lakukan dry-run dulu** untuk melihat estimasi byte yang dipindai (free tier 1 TB/bulan). Simpan hasilnya sebagai CSV di `data/raw/google2019/`, lalu jalankan `uv run scripts/prepare.py google2019`.

```sql
-- Top-200 collection (job) dengan CPU terbesar yang aktif ≥90% durasi trace, cell a, bin 5 menit
WITH usage AS (
  SELECT collection_id,
         DIV(start_time - 600000000, 300000000) AS t5,   -- trace dimulai di t=600 s; satuan µs
         SUM(average_usage.cpus) AS cpu
  FROM `google.com:google-cluster-data`.clusterdata_2019_a.instance_usage
  WHERE start_time >= 600000000
  GROUP BY 1, 2
),
top AS (
  SELECT collection_id FROM usage
  GROUP BY 1
  HAVING COUNT(DISTINCT t5) >= 0.9 * (SELECT MAX(t5) + 1 FROM usage)
  ORDER BY SUM(cpu) DESC
  LIMIT 200
)
SELECT CAST(collection_id AS STRING) AS series_id, t5, cpu AS y
FROM usage JOIN top USING (collection_id)
ORDER BY 1, 2;
```

### 2.6 Status data (2026-10-21)

| Dataset | Kandidat | Dipilih | Panjang | Kelas SBC (dipilih) | Catatan |
|---|---|---|---|---|---|
| azure2019 | 13.057 app berumur panjang | 550 | 20.160 × 1 mnt | smooth 163, erratic 134, intermittent 127, lumpy 126 | 2 app tanpa `mean_duration_ms`; simulator melewatkan keduanya |
| azure2021 | 119 app | 119 | 20.160 × 1 mnt | intermittent 77, lumpy 42, **tanpa smooth** | arsip RAR7: ekstrak dengan unrar ≥7 yang di-build dari source RARLAB |
| huawei | 50 channel | 50 | 8.640 × 5 mnt | smooth 25, erratic 21, lumpy 4 | sama dengan setup Toner et al. |
| alibaba2018 | 3.787 mesin | 201 (200 + cluster_mean) | 2.304 × 5 mnt | — (utilisasi kontinu; rata-rata CPU 38%) | sha256 cocok dengan repo resmi; 4.023 mesin mentah |
| google2019 | — | — | — | — | belum dijalankan (butuh GCP) |

Implikasi: di resolusi 1 menit, azure2021 seluruhnya intermittent/lumpy. Ini menjadi uji utama H1b. Pada resolusi 5 menit sebagian akan menjadi lebih padat, jadi hitung ulang kelas SBC per resolusi saat evaluasi.

### 2.7 Kebocoran pretraining **[verifikasi per model]**

Korpus LOTSA (Moirai) memuat dataset CloudOps dari Woo et al. (2023), termasuk **Alibaba cluster trace 2018**, Azure VM 2017, dan Borg 2011. Untuk setiap model, catat apakah korpus pretraining-nya memuat salah satu dataset kita (`configs/leakage.csv`: model, dataset, status ∈ {ya, tidak, tidak diketahui}, sumber). Pasangan model–dataset yang "ya" tetap dilaporkan, tetapi diberi tanda di setiap tabel dan dikeluarkan dari klaim zero-shot.

---

## 3. Protokol evaluasi

### 3.1 Split kronologis (per dataset)

| Dataset | Train | Val | Test | Periode musiman *m* |
|---|---|---|---|---|
| azure2019 | hari 1–10 | 11–12 | 13–14 | 1440 (1 mnt) / 288 (5 mnt) |
| azure2021 | hari 1–10 | 11–12 | 13–14 | 1440 / 288 |
| alibaba2018 | hari 1–5 | 6 | 7–8 | 288 (5 mnt saja) |
| huawei | hari 1–24 | 25–26 | 27–30 | 288 |
| google2019 | hari 1–23 | 24–26 | 27–31 | 288 |

Train dan val hanya dipakai oleh baseline terlatih dan untuk kalibrasi conformal. Model zero-shot hanya melihat konteks sebelum setiap origin.

### 3.2 Rolling origin

- Origin pada periode test setiap **60 menit** (stride 60 langkah di 1 mnt, 12 langkah di 5 mnt). Contoh: azure2019 punya 48 origin per deret.
- Konteks diambil mundur dari origin dan boleh melintasi batas train/val. Ini sah karena datanya masa lalu.
- Granularitas: 1 menit (azure*) dan 5 menit (semua dataset; resample `sum` untuk hitungan, `mean` untuk utilisasi).

### 3.3 Konteks dan horizon

| | 1 menit | 5 menit |
|---|---|---|
| Konteks utama *C* | 2048 (≈34 jam, mencakup 1 siklus harian) | 2048 (≈7 hari) |
| Ablasi *C* | 512, 2048, maks. model (dipotong ke data yang tersedia) | 512, 2048 |
| Horizon *H* dievaluasi | 1, 5, 15, 60 langkah | 1, 3, 12 langkah (5, 15, 60 mnt) |
| Pengecualian alibaba2018 (8 hari = 2.304 langkah) | — | C utama 1440 (5 hari); ablasi 512, 1440 |
| Replikasi Toner et al. (huawei) | — | C = 520, H ∈ {30, 96, 336} |

Setiap model memprediksi `max(H)` sekali per origin, lalu prefiksnya dievaluasi. Model dengan horizon native lebih pendek memakai rollout autoregresif, dan hal ini dicatat.

### 3.4 Pasca-proses forecast

- Clip ke ≥0. Tidak ada pembulatan.
- Kuantil standar Q = {0.1, 0.2, …, 0.9}. Untuk model yang hanya memberi forecast titik (TTM, LightGBM, LSTM, N-BEATS, ETS, ARIMA, naive), kuantil dibentuk dengan **split conformal**: kuantil residual absolut per (dataset, model, h) dari periode val. Dengan cara ini semua model punya P90 untuk RQ3.

---

## 4. Model

### 4.1 Antarmuka seragam

```python
def predict(context: np.ndarray, h: int, quantiles: list[float]) -> np.ndarray:
    """context [B, C] float32 (NaN-free) → [B, h, len(quantiles)]"""
```
Satu fungsi per model di `src/models.py`, dipetakan lewat `MODELS: dict[str, Callable]`. Tidak ada class hierarchy. Antarmuka yang sama dipakai ulang oleh paper energy-aware autoscaling.

### 4.2 Foundation model zero-shot

| Nama | Checkpoint HF **[verifikasi revisi]** | Ukuran | Keluaran | Catatan |
|---|---|---|---|---|
| Chronos-Bolt-small | `amazon/chronos-bolt-small` | ≈48M | kuantil | konteks ≤2048 |
| Chronos-2 | `amazon/chronos-2` | ≈120M | kuantil | rilis Okt 2025, konteks panjang |
| TimesFM-2.5 | `google/timesfm-2.5-200m-pytorch` | 200M | kuantil | konteks hingga 16k |
| Moirai-2.0-small | `Salesforce/moirai-2.0-R-small` | kecil | kuantil | **cek kebocoran Alibaba (LOTSA)** |
| Toto | `Datadog/Toto-Open-Base-1.0` | 151M | sampel → kuantil | dilatih pada metrik observability |
| TiRex | `NX-AI/TiRex` | ≈35M | kuantil | **[verifikasi dukungan CPU tanpa kernel CUDA]** |
| TTM-r2 | `ibm-granite/granite-timeseries-ttm-r2` | ≈1–5M | titik | konteks 512/1024/1536 sesuai varian |
| *(opsional)* TimesFM-3 | jika checkpoint publik tersedia | 330M | kuantil | multivariat |

Inferensi selalu `torch.float32` di CPU, `torch.set_num_threads(4)`, batch 64 untuk run akurasi.

### 4.3 Baseline

| Nama | Implementasi | Mode latih |
|---|---|---|
| Naive, Seasonal Naive (*m* harian) | numpy | — |
| AutoETS | `statsforecast` | lokal; dilatih pada train+val lalu state di-update dengan data baru tanpa estimasi ulang |
| MSTL + AutoARIMA (musiman harian) | `statsforecast` | lokal, seperti di atas. ARIMA musiman langsung dengan *m*=1440 tidak praktis |
| CrostonOptimized, TSB | `statsforecast` | lokal; hanya deret intermittent/lumpy |
| Ridge online (replikasi Toner) | numpy | refit tiap 200 langkah, λ = 0.5 |
| LightGBM | `mlforecast` | **global per dataset × resolusi**; lag 1–60 + lag *m*; normalisasi per deret |
| LSTM | `neuralforecast` | global; input_size = min(C, 1440); early stopping di val |
| N-BEATS | `neuralforecast` | global; konfigurasi sama |

Hyperparameter: default library + **satu** pencarian kecil (≤10 konfigurasi, Optuna) di val untuk LightGBM/LSTM/N-BEATS. Catat waktu training wall-clock (bahan RQ2: "biaya yang dihilangkan zero-shot").

---

## 5. Metrik

### 5.1 Akurasi (RQ1)

- **MAE** per deret per horizon.
- **MASE**: skala = MAE seasonal-naive in-sample (lag *m*) pada split train. Jika skala 0, pakai naive lag-1. Jika tetap 0, deret dikeluarkan dari MASE (dihitung dan dilaporkan jumlahnya) tetapi tetap masuk MAE dan metrik keputusan.
- **WQL** atas Q, dan **coverage P90** (target 0,9).
- Agregasi utama: **geometric mean dari MASE relatif terhadap Seasonal Naive** per (dataset, resolusi, H), dengan bootstrap CI 95% (1000 resampel deret). Juga dipecah per kelas SBC.
- Uji: Friedman + post-hoc Nemenyi, critical-difference diagram per dataset (`autorank`).

### 5.2 Biaya (RQ2)

Benchmark terpisah (`src/latency.py`), bukan diambil dari run akurasi:
- Grid: batch {1, 64} × C {512, 2048} × H = 60. Pemanasan 5 kali, lalu 30 pengulangan terukur. Laporkan latensi p50/p95.
- Peak RSS (`resource.getrusage`), waktu load model, jumlah parameter, ukuran file bobot.
- Baseline: waktu training + latensi inferensi.
- Plot: Pareto MASE-relatif vs latensi p50 (batch 64, C = 2048).
- Opsional: ulangi di VM Linux x86 4 vCPU supaya mewakili node control-plane.

### 5.3 Keputusan scaling (RQ3), `src/simulate.py`

Langkah kontrol Δ = resolusi dataset (1 menit untuk azure*, 5 menit untuk alibaba/google). Pada waktu *t*, policy hanya melihat data ≤ *t − ℓ*, dengan ℓ = ⌈latensi inferensi / Δ⌉. Dengan cara ini latensi model (dari §5.2) **ikut masuk** ke simulasi.

**(a) Serverless** (azure2019, azure2021; huawei tidak dipakai karena tidak punya durasi)
- Instance yang dibutuhkan: `N_t = ceil(λ_t · d / 60 / k)`, dengan λ = invokasi/menit, d = `mean_duration_ms`/1000, k = konkurensi per instance ∈ {1, 4, 16} (default 1).
- Policy prediktif: pada *t*, siapkan `P_{t+L} = ceil(q̂_τ(λ_{t+L}) · d / 60 / k)`, dengan lead cold start L = 1 menit dan τ ∈ {0.5, 0.9, 0.99}.
- Pembanding: keep-alive tetap 10 menit (reaktif), **hybrid histogram policy** (Shahrad et al. 2020, versi disederhanakan), dan **oracle** (λ aktual).
- Metrik: rasio cold start `Σ(N−P)⁺ / ΣN`, instance-menit idle `Σ(P−N)⁺`, dan kurva trade-off keduanya saat τ divariasikan.

**(b) VM/container** (alibaba2018, google2019)
- Beban dalam satuan replika: `D_t = y_t / 100 · R_max` (R_max = 20). Kebutuhan `R_t = ceil(D_t / u*)`, target utilisasi u* = 0,7.
- Policy prediktif: `R̂_{t+L} = ceil(q̂_τ(D_{t+L}) / u*)`, dengan lead provisioning L ∈ {5, 10} menit (1–2 langkah).
- Pembanding: reaktif ala Kubernetes HPA (`ceil(R_cur · u_obs / u*)`, stabilisasi scale-down 5 menit), dan oracle.
- Metrik: % menit under-provisioned (proksi pelanggaran SLO), replika-menit berlebih, jumlah aksi scaling.

**Keluaran untuk paper energi:** `results/sim/<ds>/<policy>.parquet` berisi `series_id, t, demand, capacity, util`.

**Analisis H3a:** korelasi Spearman antara peringkat model berdasarkan MASE dan peringkat berdasarkan biaya keputusan, per dataset.

### 5.4 Karakteristik deret (RQ4)

Fitur per deret: zero_frac, ADI, CV², kekuatan musiman (STL, periode 288 pada 5 mnt), entropi spektral, indeks dispersi. Target: log MASE-relatif (FM terbaik vs Seasonal Naive). Analisis memakai korelasi Spearman dan pohon keputusan depth ≤3 sebagai aturan pemilihan yang bisa dibaca manusia.

---

## 6. Struktur repo

```
IDEAS 1/
├── PLAN.md  SPEC.md
├── scripts/
│   ├── download.sh          # ✅ ada
│   └── prepare.py           # ✅ ada (+ selfcheck)
├── data/raw/<ds>/  data/processed/
├── configs/
│   ├── exp.yaml             # model+revisi HF, C, H, stride, seed
│   └── leakage.csv
├── src/                     # fase berikut
│   ├── data.py              # load parquet, resample, split, iterator rolling window
│   ├── models.py            # MODELS = {name: predict_fn}
│   ├── run_forecast.py      # dataset × model × C → results/forecasts/
│   ├── metrics.py
│   ├── latency.py
│   ├── simulate.py
│   └── analysis.py          # tabel, CD diagram, Pareto, gambar paper
└── results/
    ├── forecasts/<ds>/<model>_C<C>_<res>.parquet   # series_id, origin, step, q0.1..q0.9
    ├── latency.parquet
    └── sim/
```

Forecast disimpan sekali, lalu metrik dan simulasi dihitung dari file tersebut. Dengan begitu menambah metrik tidak memerlukan inferensi ulang.

---

## 7. Anggaran komputasi (estimasi, validasi dengan §5.2 dulu)

| Dataset | Deret | Origin | Jendela per (model, C, res) |
|---|---|---|---|
| azure2019 | ≤550 | 48 | ≈26k |
| azure2021 | semua app | 48 | tergantung jumlah app |
| alibaba2018 | 201 | 48 | ≈9,6k |
| huawei | 50 | 96 (+ replikasi) | ≈4,8k |

Dengan batch 64 di CPU, perkiraan kasarnya adalah menit hingga ≈1 jam per model besar per konfigurasi. Kalau benchmark latensi menunjukkan angka jauh di atas itu, kurangi origin ke stride 120 menit sebelum mengurangi jumlah deret.

---

## 8. Definition of done per fase

1. **Data**: semua parquet `processed/` ada; selfcheck lulus; tabel statistik dataset (deret, panjang, distribusi kelas SBC) masuk ke paper.
2. **Baseline**: Seasonal Naive punya MASE-relatif = 1 secara definisi (sanity check). Baseline lain punya forecast lengkap untuk semua origin.
3. **FM**: semua model × dataset × C utama lengkap. Replikasi Huawei secara kualitatif cocok dengan Toner et al. (Seasonal Naive kuat), atau perbedaannya dijelaskan.
4. **Simulasi**: oracle mendominasi semua policy. Kurva trade-off τ monoton.
5. **Paper**: setiap angka di naskah dapat dibangkitkan ulang dari `results/` oleh `analysis.py`.

---

## 9. Catatan implementasi & penyimpangan dari rencana (2026-10-22)

Bagian ini mengalahkan bagian di atas bila keduanya bertentangan.

**Kode**

| File | Isi |
|---|---|
| `src/data.py` | `load(ds, res) -> Panel` (matriks [N, T], split, origin per jam); konfigurasi split per dataset |
| `src/baselines.py` | baseline statistik, signature `fn(ctx, h, season) -> mean` |
| `src/fm.py` | wrapper 8 FM + revisi HF yang di-pin; `predict(ctx, h) -> (mean, q | None)` |
| `src/trained.py` | LightGBM (rekursif, fitur lag ternormalisasi, ditulis langsung tanpa mlforecast), N-BEATS dan LSTM (neuralforecast) |
| `src/run_forecast.py`, `src/run_fm.py` | menulis `results/forecasts/<ds>_<res>min/<model>[_C<C>].npz` berisi `ids, origins, mean [N,O,H], q [N,O,H,9]?, seconds` |
| `src/metrics.py` | MASE, relMASE (geomean vs seasonal naive), WQL, coverage P90, conformal untuk model titik |
| `src/latency.py` | benchmark §5.2, satu model per proses |
| `tests/test_core.py` | 6 tes unit (`uv run --group dev pytest`) |

Format forecast adalah `.npz`, bukan parquet: array [N, O, H] langsung dipakai metrik dan simulator.

**Penyimpangan**

1. **MSTL-ETS dan MSTL-ARIMA hanya di 5 menit.** Di 1 menit, satu fit MSTL (konteks 7 hari) butuh 4–9 detik, sehingga sekitar 8 jam per model untuk azure2019. Baseline lokal di 1 menit: naive, seasonal naive, SWA-3 (rata-rata 3 hari terakhir pada jam yang sama), ETS non-musiman, Croston, dan TSB. Musiman di 1 menit ditangani LightGBM/N-BEATS/LSTM. MSTL-ARIMA juga lambat di 5 menit (>1 detik per fit untuk sebagian deret), jadi dijalankan paling akhir.
2. **Baseline statistik memakai konteks 7 hari** (SWA-3: 3 hari), bukan C = 2048. Model terlatih memakai input 1 hari (1 mnt) atau 3 hari (5 mnt). Artinya baseline melihat **lebih banyak** sejarah daripada FM. Ini menguntungkan baseline, jadi bias perbandingannya konservatif terhadap klaim FM.
3. **Toto memakai mode mean analitik** (`num_samples=None`), dengan kuantil dari conformal. Mode sampling membutuhkan 17,7 detik (20 sampel) atau 98 detik dan 7,6 GB RAM (100 sampel) per origin untuk 50 deret, dibanding 0,6 detik untuk mean. Varian `toto_s20` hanya dijalankan pada huawei dan alibaba2018. **Ini temuan RQ2**: FM probabilistik berbasis sampling ±30× lebih mahal di CPU.
4. **TTM-r2**: varian konteks disimpan di branch HF terpisah (`1536-96-r2`, dan seterusnya), jadi revisi tidak di-pin ke `main`. Varian yang terpilih tercatat di log.
5. **TimesFM-3** (`google/timesfm-3.0-pytorch`, 330M) ternyata tersedia publik dan masuk run utama.
6. **Moirai-2.0**: point forecast = median (model tidak punya head mean).
7. **TiRex** berjalan di CPU dengan `backend="torch"` (status [verifikasi] di §4.2 terjawab).

**Lisensi checkpoint** (wajib disebut di paper): Moirai-2.0-R-small **CC-BY-NC-4.0** (riset saja), TiRex lisensi "other" (NXAI community license, cek syaratnya), TimesFM-3 "other". Model lain Apache-2.0.

**Temuan awal (huawei, 5 mnt, baseline saja)**: di horizon 5–15 menit, **naive (nilai terakhir) mengalahkan seasonal naive** dengan relMASE 0,58 di H=1 dan 0,64 di H=3. ETS non-musiman sedikit lebih baik lagi (0,55). Toner et al. hanya membandingkan dengan seasonal naive di horizon 30–336. Paper ini harus melaporkan **"baseline sederhana terbaik"**, bukan hanya seasonal naive.

### Revisi untuk jurnal nasional (2026-10-25)

8. **N-BEATS memakai scaler `standard`.** Dengan scaler `robust`, normalisasi meledak pada deret yang jarang (IQR ≈ 0), sehingga relMASE validasi Azure 2019 5 mnt menjadi 1,40. Dengan `standard` turun ke 0,69, sedangkan `identity` 1,33. Pemilihan dilakukan **hanya pada periode validasi** Azure 2019 5 mnt, lalu diterapkan ke semua dataset. Hasil lama diarsipkan di `results/logs/nbeats_robust/`. LSTM tetap memakai `robust`, karena hasilnya normal.
9. **Kebijakan histogram hibrida Shahrad dkk.** (`src/simulate.py: hybrid_histogram`) ditambahkan sebagai pembanding serverless. Parameternya: rentang 4 jam, bin 1 menit, persentil ke-5/ke-99, margin 10%, CV ≥ 2, ≥ 10 sampel, dan fraksi di luar rentang ≤ 50%. Histogram dipelajari satu kali dari data sebelum periode uji. Aplikasi yang tidak representatif, termasuk kasus yang di versi aslinya ditangani ARIMA, kembali ke keep-alive 4 jam.
10. **Audit kebocoran pretraining (§2.7).** GiftEvalPretrain (subset LOTSA) memuat `alibaba_cluster_trace_2018`. Alibaba 2018 terkonfirmasi atau sangat mungkin ada di korpus TimesFM-2.5, TimesFM-3, Chronos-2, Toto 1.0, dan Moirai 2.0. TiRex hanya menyebut Azure VM dan Borg. Chronos-Bolt tidak memakai trace cloud. TTM-r2 tidak diketahui. Azure Functions dan Huawei tidak ditemukan di korpus mana pun. Tabel lengkapnya ada di naskah (Tabel 3).
11. **Definisi horizon.** relMASE "pada horizon H" adalah rata-rata galat langkah 1..H (`src/metrics.py`), bukan galat pada langkah ke-H saja.

### Revisi setelah review kedua (2026-10-26)

12. **relMASE memakai batas bawah rasio 0,01** (`src/metrics.py: RATIO_FLOOR`). Epsilon 1e-9 sebelumnya membuat deret dengan MAE model = 0 (8 dari 77 deret Azure 2021 untuk LSTM, LightGBM, dan naive) menarik rata-rata geometrik ke nol. Ini artefak yang membalik kesimpulan Azure 2021. Metrik pendamping yang ditambahkan: rasio jumlah galat (`rMAE_sum`). Uji Friedman dan Nemenyi memakai MAE mentah.
13. **Baseline `zero`** (selalu meramal 0) ditambahkan.
14. **Friedman dilaporkan apa adanya**: n, k, χ², dan log10 p (mpmath), tabel `paper/tables/friedman_nemenyi.csv`.
15. **Simulator serverless**: durasi rata-rata dari hari latih saja (`scripts/prepare.py: TRAIN_DAYS`). Ada ambang aktivasi θ, dan (τ, θ) dipilih pada periode validasi (semua pasangan Pareto-optimal di validasi, lalu dievaluasi di uji). Metrik risiko disebut **proksi** *cold start*, dengan definisi dan pseudocode di docstring `src/simulate.py`.
16. **Analisis Azure 2021 aktif vs tidak aktif** (`paper/tables/azure2021_active_silent.csv`).

### Revisi setelah review ketiga (2026-10-26, draf v0.5)

17. **Uji berpasangan** (`src/analysis.py: paired_table`, `paper/tables/paired_tests.csv`). TSFM terbaik dan pembanding terbaik dipilih dengan relMASE pada periode validasi, lalu selisih log rasio per deret di periode uji diuji dengan Wilcoxon peringkat bertanda, CI rasio geometrik dengan bootstrap berpasangan (1000 ulangan), dan nilai p dikoreksi Holm untuk enam kombinasi. Klaim signifikansi tidak lagi dibaca dari tumpang tindih CI.
18. **Sensitivitas perlakuan galat nol** (`floor_sensitivity_table`, `paper/tables/floor_sensitivity.csv`): batas bawah 1e-9, 0,001, 0,01, 0,1, rasio jumlah galat, dan median rasio.
19. **Keterulangan**: `scripts/export_configs.py` menulis `configs/splits.json` (batas split, origin, horizon, sha256 panel), `configs/series/*.txt`, dan `configs/models.json`. Langkah reproduksi ada di `README.md`.
