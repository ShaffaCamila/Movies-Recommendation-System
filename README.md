# Laporan Proyek Machine Learning - Naia Shaffa Camila

## Project Overview

Rekomendasi film merupakan fitur penting dalam platform streaming dan aplikasi hiburan untuk membantu pengguna menemukan film yang sesuai dengan preferensi mereka. Dengan banyaknya film yang tersedia, pengguna sering merasa bingung dalam memilih film yang ingin ditonton. Oleh karena itu, sistem rekomendasi yang efektif sangat dibutuhkan untuk meningkatkan pengalaman pengguna dan loyalitas pelanggan.

Dalam proyek ini, saya menggunakan dataset Movie Recommender Dataset dari Kaggle yang berisi informasi rating pengguna terhadap berbagai film. Dataset ini memungkinkan pengembangan sistem rekomendasi berbasis collaborative filtering dan content-based filtering yang dapat memberikan rekomendasi film secara personal.

## Business Understanding

Di era modern ini, produksi film terus meningkat secara signifikan setiap tahunnya. Dengan ribuan film baru yang dirilis dari berbagai genre dan negara, pilihan bagi penonton menjadi sangat banyak dan beragam. Namun, peningkatan kuantitas ini justru membuat pengguna semakin kesulitan menemukan film yang benar-benar sesuai dengan selera dan preferensi mereka. Proses pencarian yang manual dan tidak terarah sering kali membuat pengguna merasa bingung dan frustrasi. Oleh karena itu, diperlukan solusi yang mampu membantu pengguna menavigasi lautan konten film yang terus bertambah ini secara efektif dan efisien. Sistem rekomendasi film yang dapat memahami preferensi pengguna secara mendalam menjadi kunci untuk menghadirkan pengalaman menonton yang lebih personal dan memuaskan. Proyek ini bertujuan untuk mengembangkan sistem rekomendasi yang akurat dan adaptif sehingga pengguna dapat dengan mudah menemukan film yang relevan, sekaligus mendukung pertumbuhan platform streaming dan industri film secara keseluruhan.

### Problem Statements

- Ketidakefisienan dalam Pencarian Film: Dengan semakin banyaknya jumlah film yang diproduksi setiap tahun, pengguna sering kali menghabiskan waktu berjam-jam untuk mencari film yang sesuai dengan preferensi mereka. Hal ini menyebabkan kebingungan dan menurunnya kepuasan dalam pengalaman menonton.

- Keterbatasan dalam Rekomendasi yang Akurat: Algoritma rekomendasi yang kurang optimal sering memberikan saran film yang kurang relevan, sehingga pengguna kehilangan kesempatan untuk menemukan film baru yang sebenarnya cocok dengan minat mereka.

- Tantangan dalam Memahami Preferensi Pengguna: Data rating dan preferensi pengguna yang beragam dan tidak lengkap membuat sistem sulit untuk memberikan rekomendasi yang benar-benar personal dan tepat sasaran.

### Goals

- Menghadirkan Rekomendasi Film yang Relevan dan Personal: Proyek ini bertujuan untuk membangun sistem rekomendasi yang mampu memahami preferensi unik setiap pengguna melalui analisis data rating, sehingga dapat menyarankan film yang benar-benar sesuai dengan selera mereka.

- Meningkatkan Efisiensi dalam Menemukan Film: Dengan menerapkan algoritma yang tepat, sistem rekomendasi diharapkan dapat mengurangi waktu yang dibutuhkan pengguna untuk menemukan film yang diinginkan, sekaligus mengurangi kebingungan akibat banyaknya pilihan.

- Mendorong Eksplorasi Film Baru: Sistem ini juga bertujuan untuk memperkenalkan pengguna pada film-film berkualitas yang mungkin belum pernah mereka pertimbangkan sebelumnya, sehingga memperkaya pengalaman menonton dan meningkatkan kepuasan pengguna secara keseluruhan.

### Solution Statements

Untuk mencapai tujuan yang telah ditetapkan, proyek ini mengusulkan dua pendekatan utama dalam membangun sistem rekomendasi film:

1. **Content-Based Filtering (Menggunakan Cosine Similarity)**  
   Pendekatan content-based filtering ini merekomendasikan film berdasarkan kemiripan karakteristik konten antar film. Dalam proyek ini, sistem fokus pada **genre film** sebagai fitur utama untuk menilai kesamaan antar film. Setiap film direpresentasikan dalam bentuk vektor genre, lalu sistem menghitung **kemiripan antar film** menggunakan **Cosine Similarity**, yaitu pengukuran sudut antara dua vektor dalam ruang dimensi tinggi. Nilai kemiripan akan berada dalam rentang 0 hingga 1, di mana 1 berarti sangat mirip (arah vektor sama), dan 0 berarti tidak mirip (vektor tegak lurus).

2. **Collaborative Filtering (Menggunakan Algoritma NearestNeighbors)**  
   Pendekatan collaborative filtering ini menggunakan algoritma **`NearestNeighbors`** dari pustaka scikit-learn untuk mencari film-film yang mirip berdasarkan pola rating pengguna. Alih-alih fokus pada konten film, pendekatan ini menganalisis kesamaan perilaku pengguna dalam memberikan rating. Sistem akan membentuk **user-item matrix**, di mana setiap baris mewakili pengguna dan setiap kolom mewakili film. Setiap entri pada matriks berisi nilai rating yang diberikan oleh pengguna terhadap film tersebut.

## Data Understanding

Dataset yang digunakan adalah "Movie Recommender Dataset" dari Kaggle ([movie recommender dataset](https://www.kaggle.com/datasets/gargmanas/movierecommenderdataset)) yang berisi tiga file utama:

**movies.csv**

File ini memuat informasi tentang film, termasuk judul dan genre dari masing-masing film. Terdapat sebanyak **9742 entri film**, masing-masing dengan atribut sebagai berikut:

- `movieId` : ID unik yang merepresentasikan setiap film.
- `title` : Judul lengkap film beserta tahun rilisnya.
- `genres` : Genre film yang ditulis dalam format string dan dipisahkan dengan tanda `|` jika memiliki lebih dari satu genre.

**ratings.csv**

File ini berisi data interaksi pengguna dengan film, berupa rating yang diberikan. Terdapat **100,836 entri rating** dengan atribut sebagai berikut:

- `userId` : ID unik pengguna yang memberikan rating.
- `movieId` : ID film yang dirating oleh pengguna.
- `rating` : Nilai rating yang diberikan pengguna terhadap film, berada pada skala **0.5 hingga 5**.
- `timestamp` : Waktu ketika rating diberikan, dalam format UNIX timestamp.

---

### Visualisasi Awal: Distribusi Genre Film

Untuk memahami komposisi konten yang tersedia dalam dataset, langkah awal yang dilakukan adalah meninjau sebaran genre dari semua film. Genre merupakan fitur penting dalam pendekatan content-based filtering karena mencerminkan jenis atau tema dari film yang disukai pengguna. Oleh karena itu, visualisasi distribusi genre dapat memberikan gambaran umum mengenai preferensi konten serta membantu dalam proses feature engineering ke depan.

![alt text](images/genre.png)

### Visualisasi Tambahan: Distribusi Rating Pengguna

Selain memahami jenis konten melalui genre, penting juga untuk meninjau bagaimana pengguna memberikan rating terhadap film. Rating merupakan komponen utama dalam pendekatan collaborative filtering karena model ini mengandalkan pola dan hubungan antar rating pengguna untuk merekomendasikan film. Oleh karena itu, memahami distribusi rating sangat penting untuk mengetahui kecenderungan pengguna dalam menilai film — apakah condong memberikan rating tinggi, rendah, atau tersebar merata Visualisasi distribusi rating dapat memberikan wawasan penting seperti adanya potensi bias (misalnya mayoritas rating tinggi), distribusi tidak seimbang, atau kelangkaan rating ekstrem. Informasi ini akan sangat berguna saat menyusun strategi preprocessing dan dalam membangun user-item matrix yang optimal untuk collaborative filtering.

![alt text](images/rating.png)

## Data Preparation

### Content-Based Filtering

- Membuat salinan dataset movies

  ```python
  movies_df = movies.copy()
  ```

  Langkah ini dilakukan agar data asli tetap aman dan proses preprocessing tidak merusak dataset asli.

- Membuat fitur dummy untuk setiap genre

  ```python
  genre_dummies = movies_df['genres'].str.get_dummies(sep='|')
  ```

  Karena setiap film bisa memiliki lebih dari satu genre yang dipisahkan oleh '|', kita mengonversi kolom genre menjadi kolom-kolom biner (0/1) untuk masing-masing genre. Ini memungkinkan model untuk memahami fitur genre secara numerik.

- Menggabungkan data genre dummy dengan dataframe utama

  ```python
  movies_df = pd.concat([movies_df, genre_dummies], axis=1)
  ```

  Fitur genre hasil encoding ditambahkan ke dataframe utama sehingga dapat digunakan sebagai input untuk perhitungan similarity.

- Menghapus baris yang berisi genre ‘(no genres listed)’

  ```python
  movies_df = movies_df[~movies_df['genres'].str.contains(r'\(no genres listed\)')]
  ```

  Baris dengan genre ini dihapus karena tidak memberikan informasi yang berguna untuk analisis dan dapat mengganggu hasil similarity.

- Menghapus kolom dummy ‘(no genres listed)’ jika ada
  ```python
  movies_df.drop('(no genres listed)', axis=1, inplace=True)
  ```
  Setelah penghapusan baris, kolom genre ini juga dihapus agar tidak menjadi fitur yang tidak relevan dalam model.

### Collaborative Filtering - Data Preprocessing

- Menggabungkan dataset ratings dan movies

  ```python
  df = pd.merge(ratings, movies, on='movieId')

  ```

- Membentuk User-Item Matrix

  ```python
  user_item_matrix = df.pivot_table(index='userId', columns='movieId', values='rating')
  ```

  Membuat matriks di mana baris mewakili setiap pengguna (userId), kolom mewakili setiap film (movieId), dan nilai di dalamnya adalah rating yang diberikan pengguna terhadap film tersebut. Matriks ini menjadi dasar untuk model collaborative filtering.

- Mengisi nilai kosong dengan 0
  ```python
  user_item_matrix = user_item_matrix.fillna(0)
  ```
  Karena tidak semua pengguna memberi rating ke semua film, terdapat nilai kosong (NaN). Nilai ini diisi dengan 0 untuk menyatakan bahwa pengguna tersebut belum memberi rating film tersebut. Pengisian ini penting agar algoritma dapat memproses data tanpa error.

## Modeling

### Content-Based Filtering dengan Cosine Similarity

Cosine similarity digunakan dalam Content-Based Filtering untuk mengukur tingkat kemiripan antar film berdasarkan fitur konten, yaitu genre film. Dalam sistem rekomendasi ini, setiap film direpresentasikan sebagai vektor fitur yang menunjukkan kehadiran atau ketidakhadiran genre tertentu (one-hot encoding).

Cosine similarity mengukur seberapa mirip dua vektor dalam ruang multidimensi dengan menghitung cosinus sudut antara kedua vektor tersebut. Jika dua film memiliki genre yang sangat mirip, maka sudut antara vektor fitur mereka akan kecil sehingga nilai cosine similarity mendekati 1. Sebaliknya, jika genre film sangat berbeda, nilai similarity mendekati 0.

Rumus cosine similarity antara vektor A dan B adalah:

```python
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

Keterangan:

- `A · B` : _dot product_ (perkalian titik) antara vektor A dan B
- `||A||` dan `||B||` : magnitudo (norma) dari vektor A dan B

Penggunaan cosine similarity memungkinkan sistem rekomendasi untuk menghasilkan matriks kemiripan antar film berdasarkan genre, sehingga dapat memberikan rekomendasi film yang memiliki konten genre serupa dengan film yang disukai pengguna.

## Evaluation

### Content-Based Filtering

### 🎯 Objek Film Pencarian: _Toy Story (1995)_

**Genre Film Pencarian:** `Adventure | Animation | Children | Comedy | Fantasy`

---

| No. | Rekomendasi Film                               | Cosine Similarity | Genre Rekomendasi                               | Precision Genre |
| --- | ---------------------------------------------- | ----------------- | ----------------------------------------------- | --------------- |
| 1   | Antz (1998)                                    | 1.0               | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0             |
| 2   | Toy Story 2 (1999)                             | 1.0               | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0             |
| 3   | Adventures of Rocky and Bullwinkle, The (2000) | 1.0               | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0             |
| 4   | Emperor's New Groove, The (2000)               | 1.0               | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0             |
| 5   | Monsters, Inc. (2001)                          | 1.0               | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0             |

---

### Insight:

- **Cosine Similarity = 1.0**  
  Menunjukkan bahwa secara vektor genre, film-film rekomendasi **identik** dengan film pencarian.
- **Precision Genre = 1.0**  
  Menunjukkan bahwa semua genre pada film pencarian juga **sepenuhnya terdapat** pada setiap film rekomendasi.

---

### Kesimpulan:

Sistem rekomendasi berbasis **Content-Based Filtering dengan Cosine Similarity** bekerja **sangat baik** dalam kasus ini, karena mampu merekomendasikan film-film dengan **genre yang 100% cocok** dengan film pencarian. Hal ini menunjukkan bahwa metode ini efektif dalam menghasilkan rekomendasi berbasis kesamaan konten genre.

### Collaborative Filtering

### 🎯 Objek Film Pencarian

- **Judul** : _Toy Story (1995)_
- **Genre** : `Adventure | Animation | Children | Comedy | Fantasy`
- **Rating** : 3.92

---

### 🔍 10 Film yang Mirip:

| No. | Judul Film                                        | Cosine Similarity | Rata-Rata Rating |
| --- | ------------------------------------------------- | ----------------- | ---------------- |
| 1   | Toy Story 2 (1999)                                | 0.5726            | 3.86             |
| 2   | Jurassic Park (1993)                              | 0.5656            | 3.75             |
| 3   | Independence Day (a.k.a. ID4) (1996)              | 0.5643            | 3.45             |
| 4   | Star Wars: Episode IV - A New Hope (1977)         | 0.5574            | 4.23             |
| 5   | Forrest Gump (1994)                               | 0.5471            | 4.16             |
| 6   | The Lion King (1994)                              | 0.5411            | 3.94             |
| 7   | Star Wars: Episode VI - Return of the Jedi (1983) | 0.5411            | 4.14             |
| 8   | Mission: Impossible (1996)                        | 0.5389            | 3.54             |
| 9   | Groundhog Day (1993)                              | 0.5342            | 3.94             |
| 10  | Back to the Future (1985)                         | 0.5304            | 4.04             |

---

### Insight:

- Film-film rekomendasi memiliki **kemiripan perilaku penilaian (rating)** dengan _Toy Story (1995)_, meskipun tidak semua memiliki genre yang identik.
- **Similarity di kisaran ~0.53 - 0.57** menunjukkan bahwa film-film ini memiliki **pola rating dari pengguna yang cukup mirip** dengan _Toy Story_.
- Collaborative Filtering fokus pada **pola kesamaan antar pengguna**, bukan isi konten film.

---

### Kesimpulan:

Sistem **Collaborative Filtering** mampu memberikan rekomendasi berdasarkan **kesamaan preferensi pengguna**, bukan hanya konten film. Hal ini terlihat dari munculnya film-film populer lintas genre yang cenderung disukai oleh pengguna yang juga menyukai _Toy Story_. Meskipun beberapa film tidak satu genre, kemiripan rating oleh pengguna menjadi dasar utama sistem ini dalam memberikan rekomendasi.
