# Hasil

Status 2026-10-24: **semua run akurasi dan simulasi selesai** untuk huawei, alibaba2018, azure2021, dan azure2019 (google2019 belum, karena butuh GCP).
Angka bisa dibangkitkan ulang dengan `uv run python -m src.metrics <ds> <res>`, `uv run python -m src.simulate <ds> <res>`, dan `uv run python -m src.replicate_toner report`.

Definisi:
- **relMASE**: geometric mean dari MAE_model / MAE_seasonal-naive per deret (<1 berarti lebih baik dari seasonal naive). Deret dengan MAE seasonal naive = 0 dikeluarkan.
- **H**: horizon dalam langkah (5 mnt: 1/3/12 = 5/15/60 mnt; 1 mnt: 1/5/15/60).
- **Origin**: per jam di periode test.
- Semua model berjalan di CPU. Run N-BEATS dan LSTM pertama diam-diam memakai GPU Apple (MPS); hasil itu dibuang dan diulang di CPU.

> **REVISI v0.5 2026-10-26 (review ketiga):** uji berpasangan (pasangan dipilih di validasi, Wilcoxon + Holm): rasio TSFM/pembanding Huawei 0,890, Alibaba 0,933, Az19 5m 0,800, Az19 1m 0,758, Az21 5m 0,870 (p Holm 0,022), Az21 1m 0,743; keenamnya signifikan. Sensitivitas batas bawah: arah hasil sama untuk 0,001 sampai 0,1, rasio jumlah, dan median (Az21 median hampir seri: 0,500/0,502 dan 0,506/0,511); hanya 1e-9 membalik Az21. Ablasi konteks: 2048 lebih baik dari 512 di kedua resolusi; di atas 2048 perbaikan kecil (TimesFM-3 Huawei 0,471 -> 0,436 pada 6912). Lihat `paper/tables/paired_tests.csv`, `floor_sensitivity.csv`.

> **REVISI 2026-10-26 (review kedua; menggantikan angka Azure 2021 dan simulasi di bawah):**
> - Perlakuan galat nol diperbaiki (batas bawah rasio 1%). **Azure 2021 terbalik: TSFM kini terbaik** (Toto 0,313 / 0,330 vs LSTM 0,364 / 0,439; CI bersinggungan). Klaim lama "TSFM kalah 1,8 sampai 2,3 kali" adalah artefak epsilon.
> - Selisih TSFM vs pembanding terbaik: Huawei 11%, Alibaba 7%, Azure19 5m 20%, Azure19 1m 24%, Azure21 5m 14%, Azure21 1m 25%. Signifikan (CI tidak bersinggungan, Nemenyi) hanya di Azure 2019.
> - Simulasi serverless (durasi hanya dari data latih, θ dipilih di validasi): Azure 2019 pada proksi *cold start* 5%: Chronos-2 0,047, reaktif 0,150 (−69%), LightGBM 0,090 (−48%), N-BEATS 0,242. Azure 2021 tidak konsisten: hibrida TSFM −21% vs reaktif pada 10%, tetapi +12% pada 7,6%.
> - Klaim Pareto dikoreksi: TimesFM-3 punya galat agregat terendah (0,524). Chronos-2 (0,534) dan Toto (0,545) lebih cepat.
> - Naskah: `paper/draft_paper_id.docx` v0.4.

> **REVISI 2026-10-25 (berlaku atas angka di bawah):**
> - **N-BEATS dijalankan ulang dengan scaler `standard`** (lihat SPEC §9 butir 8). Hasil baru pada horizon terpanjang: Huawei 0,525, Alibaba 0,717, Azure19 5m 0,586, Azure19 1m 0,732, Azure21 5m 0,278, Azure21 1m 0,501. N-BEATS kini menjadi **pembanding terbaik** di Huawei dan Azure 2019. Selisih TSFM terbaik terhadap pembanding terbaik berubah: Huawei 28% → **11%** (CI bersinggungan), Azure19 5m 30% → **20%**, Azure19 1m 29% → **25%**. Alibaba tetap 7%.
> - **Kebijakan histogram hibrida Shahrad dkk.:** Azure 2019 *cold start* 4,7% / idle 0,206, dan TimesFM-3 pada *cold start* yang sama ±0,069 (−66%). Azure 2021 *cold start* 5,7% / idle 9,33 (buruk karena fallback 4 jam).
> - **Audit kebocoran:** korpus TimesFM-2.5/3, Chronos-2, Toto, dan Moirai-2 (kemungkinan besar) memuat Alibaba 2018. Azure dan Huawei bersih.
> - **H3a:** Spearman 0,88 / 0,99 / 0,83. N-BEATS paling akurat di antara pembanding di Azure 2019, tetapi keputusannya buruk (idle 0,259 pada *cold start* 5%, LightGBM 0,119).
> - Naskah terkini: `paper/draft_paper_id.docx` (v0.3 final).

---

## RQ1: Akurasi

### Ringkasan lintas dataset (relMASE pada horizon terpanjang: 60 menit)

| Model | huawei 5m | alibaba 5m | azure2019 5m | azure2019 1m | azure2021 5m | azure2021 1m |
|---|---|---|---|---|---|---|
| TimesFM-3 | 0.471 | **0.610** | **0.468** | 0.561 | 0.249 | 0.296 |
| Chronos-2 | **0.467** | 0.614 | 0.488 | 0.581 | 0.344 | 0.313 |
| Toto (mean) | 0.515 | 0.639 | 0.489 | **0.550** | 0.244 | 0.271 |
| TimesFM-2.5 | 0.499 | 0.613 | 0.522 | 0.610 | 0.288 | 0.340 |
| TiRex | 0.533 | 0.638 | 0.528 | 0.679 | 0.270 | 0.349 |
| Moirai-2.0-small ⚠️ | 0.530 | 0.641 | 0.576 | 0.796 | 0.311 | 0.378 |
| Chronos-Bolt-small | 0.602 | 0.659 | 0.745 | 1.079 | 0.408 | 0.527 |
| TTM-r2 | 0.755 | 0.673 | 1.151 | 1.543 | 0.932 | 1.121 |
| LightGBM | 0.669 | 0.695 | 0.703 | 0.832 | 0.142 | 0.157 |
| LSTM | 1.037 | 0.770 | 0.673 | 0.780 | **0.107** | **0.152** |
| N-BEATS | 0.707 | 0.710 | 1.227 | 1.237 | 0.307 | 0.537 |
| MSTL-ARIMA | 0.652 | 0.653 | 0.945 | – | 0.908 | – |
| MSTL-ETS | 0.647 | 0.698 | 0.963 | – | 0.969 | – |
| ETS | 0.886 | 0.780 | 0.945 | 1.236 | 0.476 | 0.693 |
| Naive | 0.928 | 0.844 | 0.846 | 1.146 | 0.133 | 0.185 |
| TSB | 1.286 | 0.750 | 0.900 | 1.140 | 0.358 | 0.232 |
| Croston | 1.024 | 0.730 | 0.919 | 1.282 | 1.574 | 2.240 |

⚠️ Moirai kemungkinan pernah melihat Alibaba 2018 saat pretraining (LOTSA).
MSTL hanya dijalankan di resolusi 5 menit (lihat SPEC §9).

### Horizon terpendek (5 menit / 1 menit)

| Dataset | FM terbaik | Baseline terbaik |
|---|---|---|
| huawei 5m (H=1) | 0.376 (TimesFM-3) | 0.484 (LightGBM) |
| alibaba 5m (H=1) | 0.540 (TimesFM-3) | 0.547 (MSTL-ARIMA) |
| azure2019 5m (H=1) | 0.396 (TimesFM-3) | 0.613 (LightGBM) |
| azure2019 1m (H=1) | 0.450 (TimesFM-3/Chronos-2) | 0.610 (LightGBM) |
| azure2021 5m (H=1) | 0.100 (Moirai) | **0.034 (naive)** |
| azure2021 1m (H=1) | 0.156 (TiRex) | **0.088 (naive)** |

### Per kelas pola (Syntetos–Boylan, dihitung pada split train di resolusi yang sama)

| Dataset, H | Kelas (n) | FM terbaik | Baseline terbaik |
|---|---|---|---|
| azure2019 5m, H=12 | smooth (209) | 0.451 TimesFM-3 | 0.640 LSTM |
| | erratic (140) | 0.455 TimesFM-3 | 0.593 LSTM |
| | lumpy (79) | 0.474 TimesFM-3 | 0.634 LightGBM |
| | intermittent (40) | 0.607 TimesFM-3 | 0.971 MSTL-ARIMA |
| azure2019 1m, H=60 | smooth (149) | 0.465 TimesFM-3 | 0.556 Croston |
| | erratic (131) | 0.483 Chronos-2 | 0.638 LSTM |
| | lumpy (115) | 0.524 Toto | 0.673 LightGBM |
| | intermittent (82) | 0.896 Toto | 1.099 SWA-3 |
| azure2021 1m, H=60 | intermittent (42) | 0.432 Toto | **0.402 LightGBM** |
| | lumpy (35) | 0.155 Toto | **0.045 LSTM** |

Catatan: di azure2021 1m, 42 dari 119 app dikeluarkan dari relMASE karena seasonal naive memiliki MAE 0 di test (app praktis mati).

### Replikasi Toner et al. (Huawei, C=520, MASE rata-rata)

| Model | H=30 | H=96 | H=336 |
|---|---|---|---|
| TimesFM-3 | **0.912** | 1.102 | 1.347 |
| Chronos-2 | 0.920 | **1.082** | **1.301** |
| TimesFM-2.5 | 0.996 | 1.216 | 1.453 |
| Toto | 1.004 | 1.347 | 1.671 |
| TiRex | 1.100 | 1.402 | 1.606 |
| Moirai-2.0-small | 1.124 | 1.558 | 2.096 |
| Chronos-Bolt-small | 1.222 | 1.468 | 1.653 |
| TTM-r2 | 1.290 | 1.600 | – |
| **Seasonal naive** | 1.324 | 1.358 | 1.459 |
| Naive | 1.936 | 3.413 | 4.054 |

### Kesimpulan RQ1

1. **H1a didukung.** Pada beban padat atau periodik (Huawei, Alibaba, Azure 2019), FM generasi 2025–2026 (TimesFM-3, Chronos-2, Toto) adalah yang terbaik di semua horizon. Keunggulannya atas baseline terbaik sekitar 25–35% (Azure 2019, Huawei) atau tipis (Alibaba: ±7% di H=12, ±1% di H=1 vs MSTL-ARIMA).
2. **H1b didukung sebagian.** Pada Azure 2021, yang seluruhnya intermittent/lumpy, FM **kalah telak** dari naive, LightGBM, dan LSTM: ±2–3× lebih buruk untuk deret lumpy. Tetapi di Azure 2019, FM juga unggul untuk kelas intermittent. Penentunya tampaknya tingkat kesparsean ekstrem (aplikasi yang hampir selalu nol), bukan label kelas SBC saja.
3. **Hasil negatif Toner et al. tidak berlaku untuk FM generasi baru.** Pada setting mereka sendiri, Chronos-2 dan TimesFM-3 mengalahkan seasonal naive di semua horizon. Model kecil atau lama tetap kalah di horizon ≥96, konsisten dengan temuan mereka.
4. **Ukuran model penting.** Model kecil (TTM-r2, Chronos-Bolt-small) sering kalah dari LightGBM. "Zero-shot" bukan jaminan, dan yang unggul hanya model terbesar dan terbaru.

---

### Uji statistik (Friedman + post-hoc Nemenyi, α = 0,05; `paper/tables/friedman_nemenyi.csv`, diagram CD di `paper/figures/cd_*.pdf`)

- Friedman menolak "semua model setara" di semua dataset dan horizon (p < 1e-300).
- **Model terbaik tidak pernah berbeda signifikan dari runner-up.** Di Azure 2019 1m H=60, peringkat rata-rata TimesFM-3 3,32 vs Toto 3,51 dengan CD 1,13. Klaim yang aman adalah "kelompok teratas = TimesFM-3/Chronos-2/Toto", bukan satu pemenang.
- Di Azure 2019, kelompok FM teratas berbeda signifikan dari baseline terbaik (LightGBM, peringkat 8,55; selisih >CD).
- **Peringkat ≠ rata-rata di Azure 2021.** Berdasarkan peringkat per deret, TiRex/Toto/TimesFM-3 paling sering menang. Berdasarkan geometric mean, LSTM dan LightGBM yang terbaik, karena mereka menang dengan selisih sangat besar di sebagian deret (aplikasi yang hampir selalu nol). CI bootstrap di Azure 2021 juga sangat lebar (mis. LSTM 0,107 [0,03; 0,28]). Paper harus melaporkan **keduanya**.
- Bootstrap CI 95% (1000 resampel deret) untuk semua sel ada di `paper/tables/accuracy.csv|.tex`. Di dataset padat, CI kelompok FM teratas tidak tumpang tindih dengan CI baseline terbaik (mis. Azure 2019 5m: TimesFM-3 [0,43; 0,51] vs LSTM [0,61; 0,75]; Alibaba: TimesFM-3 [0,59; 0,63] vs MSTL-ARIMA [0,64; 0,67]). CI hanya dihitung di horizon terpanjang. Di Alibaba H=1 selisihnya tipis (0,540 vs 0,547) dan belum diuji dengan CI.

### Toto probabilistik (20 sampel)

Lebih buruk daripada Toto mode mean (Huawei 0,595 vs 0,515; Alibaba 0,697 vs 0,639), padahal 35× lebih lambat. Rata-rata 20 sampel lebih berisik daripada mean analitik. Dengan anggaran CPU, mode sampling tidak sepadan.

---

### Ablasi panjang konteks (`paper/figures/context_ablation.pdf`, `paper/tables/context_ablation.csv`)

| C | Chronos-2 | TimesFM-3 | Toto | TiRex |
|---|---|---|---|---|
| *Huawei 5m, H=12* | | | | |
| 512 | 0.561 | 0.543 | 0.567 | 0.613 |
| 1024 | 0.470 | 0.472 | 0.521 | 0.544 |
| 2048 (utama) | 0.467 | 0.471 | 0.515 | 0.533 |
| 4096 | 0.458 | 0.454 | 0.523 | 0.533 |
| 6912 | 0.459 | **0.436** | – | 0.533 |
| *Azure 2019 1m, H=60* | | | | |
| 512 | **0.537** | 0.547 | 0.576 | – |
| 2048 (utama) | 0.581 | 0.561 | 0.550 | 0.679 |
| 8192 (Toto: 4096) | 0.596 | – | 0.546 | – |

- Di 5 menit, konteks panjang membantu sampai ±1024 langkah (3,5 hari). Hanya TimesFM-3 yang terus membaik sampai 24 hari.
- **Di 1 menit, konteks panjang memperburuk Chronos-2 dan TimesFM-3.** Chronos-2 C=8192 juga 4,4× lebih lambat (6.092 vs 1.394 dtk). Toto sedikit membaik.
- TiRex identik di C ≥ 2048, kemungkinan konteks dipotong internal ke 2048.
- Implikasi: C adalah hyperparameter yang bergantung resolusi. Angka utama (C=2048) sedikit **merugikan** Chronos-2 dan TimesFM-3 di Azure 2019 1m, sehingga kesimpulan RQ1 bersifat konservatif.

---

## RQ3: Keputusan autoscaling

Policy prediktif berupa rencana kapasitas per jam dari forecast (kenop = kuantil τ ∈ {0.5…0.9}). Policy hybrid = max(prediktif, reaktif). Pembandingnya:
- **Serverless:** keep-alive tetap, dengan kenop 1–60 menit.
- **VM:** HPA reaktif dengan jeda satu langkah, dengan kenop target utilisasi 0,4–0,8.

### Azure 2019 (serverless, 1 mnt, 548 app)

| Policy | cold start | idle instance-mnt / mnt-demand |
|---|---|---|
| Keep-alive 5 mnt | 3.9% | 0.26 |
| Keep-alive 10 mnt (default Azure) | 3.3% | 0.43 |
| Keep-alive 30 mnt | 0.7% | 0.99 |
| **Prediktif TimesFM-3 τ=0.9** | **1.9%** | **0.19** |
| Prediktif Chronos-2 τ=0.9 | 1.9% | 0.18 |
| Prediktif LightGBM τ=0.9 | 2.6% | 0.33 |
| Prediktif naive τ=0.9 | 3.0% | 0.99 |
| Hybrid TimesFM-3 τ=0.9 | 0.5% | 0.50 |
| Hybrid LightGBM τ=0.9 | 0.8% | 0.64 |

Pada cold start 5%, idle untuk TimesFM-3 adalah 0,064, untuk LightGBM 0,119, dan untuk keep-alive reaktif 0,166. **FM memangkas kapasitas idle ±60% dibanding reaktif dan ±45% dibanding LightGBM.** Pada cold start ±2%, FM butuh idle 0,18 sementara reaktif butuh 0,84, sekitar 4,5× lebih hemat.

### Azure 2021 (serverless, 1 mnt, 119 app, sangat sparse)

- **Rencana prediktif murni gagal total.** Cold start 69–89% untuk semua model, karena burst jarang tidak terprediksi oleh rencana per jam.
- Hybrid (keep-alive 10 mnt + forecast) menghasilkan cold start 5–7%. Untuk naive, LSTM, dan LightGBM, idle-nya 1,4–1,9, hampir sama dengan keep-alive saja (±1,4 pada cold start 7%). Untuk FM idle-nya 2,4–7,3, artinya FM **menambah** kapasitas terbuang karena terus memprediksi nilai kecil positif.
- Kesimpulan: untuk aplikasi serverless yang sangat sparse, forecasting tidak membantu, dan FM justru merugikan.

### Alibaba 2018 (VM, 5 mnt, 201 deret): replika berlebih per replika yang dibutuhkan, pada risiko under-provisioning yang sama

| Policy | risiko 5% | risiko 10% | risiko 20% |
|---|---|---|---|
| HPA reaktif | 0.242 | 0.187 | 0.105 |
| **Hybrid TimesFM-3** | **0.201** | 0.144 | – |
| Hybrid Chronos-2 | – | **0.141** | – |
| Hybrid MSTL-ARIMA | – | 0.142 | – |
| Hybrid LightGBM | – | 0.149 | – |
| Prediktif murni Chronos-2/TimesFM-3 | – | 0.168 | 0.106 |
| Prediktif murni MSTL-ARIMA | – | – | 0.114 |

- Rencana prediktif murni per jam **tidak lebih baik dari HPA reaktif** (0,106 vs 0,105 di risiko 20%).
- Hybrid menghemat ±24% dibanding reaktif di risiko 10%, tetapi **FM hampir setara dengan MSTL-ARIMA** (0,141 vs 0,142).

### H3a: apakah urutan akurasi = urutan kualitas keputusan?

Korelasi Spearman antara peringkat relMASE dan peringkat biaya pada risiko yang sama:

| Kasus | n model | ρ |
|---|---|---|
| Alibaba, hybrid, risiko 10% | 18 | 0.87 |
| Alibaba, prediktif, risiko 20% | 18 | 0.99 |
| Azure 2019, prediktif, cold 5% | 15 | 0.82 |

**H3a tidak didukung dalam bentuk aslinya**, karena urutan model tetap terjaga (ρ ≥ 0,82). Yang berubah adalah **besar selisihnya**. Di Alibaba-hybrid, keunggulan akurasi FM ±7–12% menyusut menjadi ±1–5% di biaya keputusan. Di Azure 2019, keunggulannya justru **membesar** (±30% akurasi → ±45% idle). Faktor yang menentukan adalah **cara memakai forecast** (prediktif vs hybrid, per jam vs reaktif) dan sifat beban (padat vs sparse), lebih daripada pilihan model.

---

## RQ2: Biaya CPU

Benchmark resmi: `src/latency.py` → `results/latency.csv` (30 pengulangan setelah 5 pemanasan, mesin idle).

Latensi p50 (ms) per panggilan, H=60, 4 thread, M3 Pro, CPU. RSS mencakup ±1,3 GB data yang dimuat harness.

| Model | b=1, C=512 | b=1, C=2048 | b=64, C=512 | b=64, C=2048 | peak RSS (MB) |
|---|---|---|---|---|---|
| Naive / seasonal naive | <0.1 | <0.1 | <0.1 | <0.1 | 1300 |
| TSB / Croston | 0.2 | 0.3 | 8.5 | 14 | 1400 |
| TTM-r2 | 1.3 | 1.7 | 6.7 | 15 | 1586 |
| Chronos-Bolt-small | 8.2 | 13.5 | 94 | 338 | 1943 |
| Moirai-2.0-small | 4.2 | 10.5 | 112 | 473 | 1667 |
| **Toto (mean)** | 27.7 | 29.2 | 184 | **657** | 2925 |
| **Chronos-2** | 27.5 | 49.5 | 477 | **1696** | 2481 |
| ETS (per deret) | 2.3 | 8.4 | 595 | 3409 | 1342 |
| TimesFM-3 | 74.8 | 128.7 | 1615 | 3338 | 5044 |
| TimesFM-2.5 ¹ | 836 | 3348 | 838 | 3604 | 4912 |
| TiRex | 223 | 223 | 9909 | 9898 | 1887 |
| Toto, 20 sampel | 111 | 283 | 4154 | **22994** | **9567** |

¹ Wrapper meng-compile TimesFM-2.5 dengan `per_core_batch_size=64`, jadi batch 1 tetap di-padding ke 64. Angka b=1 adalah artefak konfigurasi, bukan batas model.

Pareto akurasi–latensi (b=64, C=2048):
- **Chronos-2** (1,7 dtk, relMASE terbaik/kedua terbaik) dan **Toto mean** (0,66 dtk, top-3 di Azure) mendominasi TimesFM-3 (3,3 dtk, akurasi setara) dan TiRex (9,9 dtk, lebih buruk).
- TTM-r2 paling cepat (15 ms) tetapi akurasinya di bawah LightGBM.
- **Forecast probabilistik berbasis sampel** (Toto 20 sampel) 35× lebih lambat dan memakan 9,6 GB RAM.
- ETS lokal per deret pada C=2048 sudah lebih lambat daripada TimesFM-3. Biaya zero-shot FM besar tidak lebih mahal daripada fitting statistik per deret.

Waktu run penuh azure2019 1m (550 deret × 96 origin, 8 thread): TTM 15 dtk, Moirai 287 dtk, Chronos-Bolt 283 dtk, Toto 624 dtk, Chronos-2 1394 dtk, TimesFM-2.5 2990 dtk, TimesFM-3 3573 dtk, TiRex 9585 dtk.
Training baseline azure2019 1m (CPU): LightGBM 384 dtk, LSTM 1615 dtk, N-BEATS 2233 dtk. MSTL-ARIMA azure2019 5m: 9106 dtk (tanpa training global, tetapi fit ulang per origin).
Satu siklus forecast per jam untuk 550 app dengan TimesFM-3 butuh ±37 dtk di CPU laptop, jauh di bawah anggaran 60 menit. Bahkan TiRex (±100 dtk) masih layak.

---

## Narasi paper yang didukung data

> FM time-series generasi 2025–2026 memang mengalahkan baseline terlatih dan statistik dalam akurasi zero-shot pada beban cloud yang padat (25–35%), dan hasil negatif sebelumnya (Toner et al.) tidak bertahan untuk model baru. Namun (i) pada beban serverless yang sangat sparse mereka kalah dari naive/LSTM dan justru menambah kapasitas idle; (ii) dalam autoscaling, keuntungan akurasi hanya berbuah besar bila forecast dipakai sebagai rencana pre-warm serverless (Azure 2019: −45% idle vs LightGBM), sedangkan pada VM keuntungannya nyaris hilang setelah digabung dengan kontrol reaktif.

## Sisa pekerjaan

- Opsional: forecast tiap langkah kontrol untuk 2–3 model (keputusan user).
- Opsional: Google 2019 (butuh project GCP).

## Artefak paper (`uv run python -m src.analysis`)

| File | Isi |
|---|---|
| `paper/tables/datasets.tex` | jumlah deret dan kelas SBC per dataset |
| `paper/tables/accuracy.tex` | relMASE [CI 95%] semua model × dataset, terbaik ditebalkan |
| `paper/tables/friedman_nemenyi.csv` | p Friedman, CD, model terbaik dan runner-up per dataset/horizon |
| `paper/tables/relmase_ci_<ds>.csv` | angka mentah CI |
| `paper/figures/cd_<ds>.pdf` | diagram critical difference (6 buah) |
| `paper/figures/pareto_accuracy_latency.pdf` | akurasi vs latensi CPU |
| `paper/figures/tradeoff_autoscaling.pdf` | kurva trade-off autoscaling (3 panel) |
| `paper/figures/context_ablation.pdf` | relMASE vs panjang konteks |
