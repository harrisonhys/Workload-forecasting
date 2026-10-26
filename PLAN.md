# Rencana Riset: Zero-Shot Forecasting Beban Kerja untuk Autoscaling Serverless & Middleware

**Judul kerja:** *Do Time-Series Foundation Models Make Better Autoscalers? A Decision-Aware, CPU-Budgeted Benchmark on Public Cloud and Serverless Traces*

Dokumen dibuat 2026-10-21. Detail teknis ada di [SPEC.md](SPEC.md).

---

## 1. Posisi terhadap literatur

Prior work yang harus dihadapi langsung adalah **Toner et al., "Performance of Zero-Shot Time Series Foundation Models on Cloud Data"** (ICLR 2025 workshop *I Can't Believe It's Not Better*, arXiv:2502.12944). Ringkasan temuan mereka:

- Data: 4 subset curated dari trace serverless Huawei Cloud, resolusi 5 menit, 30 hari, 7–18 channel per subset.
- Model: VisionTS, TTM-r2, TimesFM, Chronos-**tiny**, Moirai-**small**, Mamba4Cast.
- Hasil: **seasonal naive mengalahkan semua foundation model** di semua horizon (MASE). TimesFM sering ~2× error seasonal naive.

Gap yang masih terbuka dan menjadi kontribusi paper ini:

| Keterbatasan Toner et al. | Yang kita lakukan |
|---|---|
| Satu provider (Huawei), channel dipilih manual | 4–5 trace publik: Azure Functions 2019 & 2021, Alibaba 2018, Huawei (replikasi), Google 2019 (opsional) |
| Checkpoint lama/kecil (Chronos-tiny, Moirai-small) | Generasi 2025–2026: Chronos-2, Chronos-Bolt, TimesFM-2.5, Moirai-2.0, Toto (dilatih pada data observability), TiRex, TTM-r2 |
| Satu panjang konteks (520) | Ablasi panjang konteks (512 → maksimum yang didukung model) |
| Hanya akurasi (MASE) | Akurasi **+ dampak ke keputusan scaling** lewat simulasi trace-driven |
| Latensi CPU hanya untuk TTM | Latensi CPU, memori, dan waktu load untuk **semua** model dan baseline, termasuk biaya *training* baseline |
| Tidak ada analisis kapan FM gagal | Stratifikasi per karakteristik deret (kelas Syntetos–Boylan, kekuatan musiman, burstiness) |

Rujukan pendukung: Toto/BOOM (Cohen et al., arXiv:2505.14766), foundation model yang dilatih khusus untuk metrik observability. Model ini kandidat kuat untuk mematahkan hasil negatif Toner et al.

**Sikap ilmiah:** paper ini **tidak mengasumsikan FM menang**. Hasil negatif yang direplikasi di beberapa provider, ditambah bukti bahwa error lebih kecil belum tentu menghasilkan keputusan scaling lebih baik, tetap merupakan kontribusi yang layak terbit.

---

## 2. Pertanyaan riset (RQ) dan hipotesis

**RQ1 — Akurasi.** Apakah TSFM zero-shot generasi terbaru menyamai atau mengungguli baseline terlatih (ARIMA/ETS, LightGBM, LSTM, N-BEATS) dan seasonal naive pada beban kerja cloud, lintas granularitas (1 dan 5 menit), horizon, dan kelas deret?
- *H1a:* Pada deret padat dan periodik (VM/mesin Alibaba, app Azure bervolume tinggi), FM terbaru ≤ seasonal naive dalam MASE.
- *H1b:* Pada deret intermittent/lumpy (mayoritas app Azure Functions), FM kalah dari baseline sederhana (naive, Croston/TSB).

**RQ2 — Biaya.** Bagaimana trade-off akurasi terhadap latensi inferensi CPU dan memori? Model mana yang muat di control loop autoscaler (interval 1 menit, ratusan hingga ribuan deret)?
- *H2:* Ada model FM kecil (TTM, TiRex, Chronos-Bolt-small) yang berada di Pareto front akurasi–latensi. Model >100M parameter tidak.

**RQ3 — Dampak keputusan.** Apakah akurasi forecast yang lebih baik menghasilkan keputusan scaling yang lebih baik (cold start, pelanggaran SLO, kapasitas terbuang, jumlah aksi scaling)?
- *H3a:* Korelasi peringkat antara MASE dan biaya keputusan **tidak sempurna** (Spearman ρ < 0.8). Akurasi saja tidak cukup untuk memilih forecaster autoscaling.
- *H3b:* Forecast kuantil (P90) dari FM probabilistik menurunkan pelanggaran SLO dibanding forecast titik pada tingkat over-provisioning yang sama.

**RQ4 — Kapan FM gagal (sekunder).** Fitur deret apa yang memprediksi performa relatif FM terhadap seasonal naive? Hasilnya bisa dipakai sebagai aturan pemilihan model yang murah.

---

## 3. Kontribusi yang dijanjikan

1. Benchmark zero-shot TSFM pertama yang mencakup **beberapa trace cloud/serverless publik**, dengan protokol dan kode terbuka.
2. Replikasi (dan pembaruan) hasil Toner et al. dengan checkpoint generasi 2025–2026.
3. **Evaluasi berbasis keputusan**: simulator autoscaling trace-driven untuk serverless (cold start vs. idle) dan VM/container (SLO vs. over-provisioning), dengan latensi inferensi ikut disimulasikan.
4. Profil biaya CPU yang lengkap: akurasi vs. latensi vs. biaya training.
5. Modul forecasting dengan antarmuka seragam yang dipakai ulang oleh **paper energy-aware autoscaling** (paper pendamping). Simulator mengeluarkan timeline kapasitas yang bisa langsung dimasukkan ke model daya.

---

## 4. Desain penelitian (ringkas)

| Komponen | Pilihan |
|---|---|
| Dataset | Azure Functions 2019 (1 menit, 14 hari), Azure Functions 2021 (per-invokasi → 1 menit), Alibaba 2018 machine_usage (→ 5 menit, 8 hari), Huawei curated (5 menit, 30 hari), Google 2019 (5 menit, opsional via BigQuery) |
| Model zero-shot | Chronos-Bolt (small), Chronos-2, TimesFM-2.5, Moirai-2.0 (small), Toto, TiRex, TTM-r2 |
| Baseline | Naive, Seasonal Naive, AutoETS, AutoARIMA, Croston/TSB (intermittent), LightGBM (global), LSTM (global), N-BEATS (global), ridge online (replikasi Toner) |
| Protokol | Split kronologis train/val/test; rolling-origin pada periode test; konteks dan horizon dibuat bervariasi |
| Metrik akurasi | MAE, MASE, WQL/CRPS, coverage P90; agregasi geometric mean relatif terhadap seasonal naive |
| Metrik biaya | Latensi p50/p95 (batch 1 & 64), peak RSS, waktu load; waktu training untuk baseline |
| Metrik keputusan | Serverless: rasio cold start, instance-menit idle. VM: % interval under-provisioned, core-menit terbuang, jumlah aksi scaling |
| Uji statistik | Friedman + Nemenyi (critical difference diagram), bootstrap CI 95% |

Detail di [SPEC.md](SPEC.md).

---

## 5. Linimasa (12 minggu, mulai 2026-10-26)

| Minggu | Aktivitas | Keluaran |
|---|---|---|
| 1 | Unduh & siapkan dataset, EDA, klasifikasi deret | `data/processed/*.parquet`, notebook EDA, tabel statistik dataset |
| 2 | Loader, rolling-window, baseline statistik (naive, seasonal, ETS, ARIMA, Croston) | Hasil baseline lokal |
| 3 | Baseline global (LightGBM, LSTM, N-BEATS) + pengukuran waktu training | Hasil baseline terlatih |
| 4–5 | Wrapper FM dengan antarmuka seragam, run zero-shot semua dataset | Tabel akurasi RQ1 |
| 5 | Replikasi Toner et al. di data Huawei (sanity check) | Tabel replikasi |
| 6 | Benchmark latensi/memori CPU | Pareto plot RQ2 |
| 7–8 | Simulator autoscaling (serverless + VM), integrasi latensi | Hasil RQ3 |
| 9 | Ablasi konteks, analisis fitur (RQ4), uji statistik | Gambar & tabel final |
| 10–11 | Penulisan paper | Draft lengkap |
| 12 | Review internal, rilis kode + artefak | Submission |

Google 2019 (BigQuery) dikerjakan paralel di minggu 1–3 **hanya jika** kuota/akses GCP tersedia. Dataset ini tidak menjadi syarat paper.

---

## 6. Risiko dan mitigasi

| Risiko | Dampak | Mitigasi |
|---|---|---|
| FM kalah total (seperti Toner et al.) | Narasi "FM unggul" gugur | Paper dibingkai sebagai evaluasi berbasis keputusan. Hasil negatif yang direplikasi lintas provider dan dikaitkan dengan biaya keputusan tetap merupakan kontribusi. |
| Deret Azure sangat sparse sehingga MASE tidak terdefinisi (skala nol) | Metrik bias | Laporkan per kelas Syntetos–Boylan. Deret dengan skala nol dikeluarkan dari MASE tetapi tetap masuk MAE dan metrik keputusan. |
| Disk lokal terbatas (≈45 GB bebas) | Tidak bisa memakai container_usage Alibaba (29 GB) atau Google penuh (2,4 TiB) | Pakai machine_usage (1,8 GB) yang di-stream langsung dari tar.gz. Google diagregasi di BigQuery. |
| Latensi FM besar di CPU | Run lama | Batasi jumlah deret dan origin (lihat SPEC §5). Batch inference. |
| Kebocoran data: FM mungkin pernah melihat trace publik saat pretraining | Hasil zero-shot terlalu optimistis | Korpus LOTSA (Moirai) memuat Alibaba 2018 lewat CloudOps TSF (Woo et al. 2023). Audit korpus tiap model (SPEC §2.7), tandai pasangan yang tumpang tindih, dan laporkan terpisah. |
| Versi checkpoint berubah | Tidak reprodusibel | Pin revisi Hugging Face (commit hash) di config. |

---

## 7. Kaitan dengan paper pendamping (energy-aware autoscaling)

- Antarmuka bersama: `predict(context[B, C], h, quantiles) -> [B, h, Q]` (SPEC §4.1).
- Simulator mengeluarkan `capacity_t` (instance/mesin aktif per menit) dan `util_t`. Paper energi tinggal menambahkan model daya, misalnya `P = P_idle + (P_max − P_idle)·u`.
- Hasil RQ2 (latensi CPU) menjadi input untuk anggaran overhead energi forecaster itu sendiri.

---

## 8. Target venue (cek tenggat terbaru sebelum memilih)

- Workshop: WoSC (Workshop on Serverless Computing, co-located Middleware), workshop NeurIPS/ICLR bertema time series.
- Konferensi: IEEE CLOUD, IEEE IC2E, ACM/SPEC ICPE, CCGrid, Middleware.
- Jurnal: Future Generation Computer Systems, IEEE Transactions on Cloud Computing, Journal of Systems and Software.

---

## Referensi kunci

- Toner, W., Lee, T. L., Joosen, A., Singh, R., & Asenov, M. (2025). *Performance of Zero-Shot Time Series Foundation Models on Cloud Data.* arXiv:2502.12944. Data: github.com/sir-lab/time-series-fm-dataset
- Cohen, B., et al. (2025). *This Time is Different: An Observability Perspective on Time Series Foundation Models* (Toto, BOOM). arXiv:2505.14766.
- Shahrad, M., et al. (2020). *Serverless in the Wild: Characterizing and Optimizing the Serverless Workload at a Large Cloud Provider.* USENIX ATC '20. (Azure Functions 2019 + kebijakan hybrid histogram sebagai baseline keep-alive)
- Zhang, Y., et al. (2021). *Faster and Cheaper Serverless Computing on Harvested Resources.* SOSP '21. (Azure Functions 2021)
- Woo, G., et al. (2023). *Pushing the Limits of Pre-training for Time Series Forecasting in the CloudOps Domain.* arXiv:2310.05063. (pretraining pada data cloud; sumber potensi kebocoran)
- Alibaba cluster-trace-v2018, github.com/alibaba/clusterdata
- Google ClusterData 2019 (Borg v3), github.com/google/cluster-data
