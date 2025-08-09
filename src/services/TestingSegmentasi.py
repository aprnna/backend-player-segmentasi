import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def create_bert_embeddings(csv_file_path, output_file_path):
    print("Loading data...")
    df = pd.read_csv(csv_file_path)
    required_columns = ['Game', 'cleaned_Reviews']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the dataset")
    print(f"Loaded {len(df)} reviews")
    # Remove duplicates based on cleaned_Reviews to ensure unique embeddings
    print("Removing duplicate reviews...")
    df_unique = df.drop_duplicates(subset=['cleaned_Reviews'], keep='first').reset_index(drop=True)
    print(f"After removing duplicates: {len(df_unique)} unique reviews")
    # Initialize BERT model for embeddings
    print("Loading BERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Creating embeddings...")
    # Prepare reviews for embedding
    reviews = df_unique['cleaned_Reviews'].fillna('').astype(str).tolist()
    # Create embeddings in batches to manage memory
    batch_size = 32
    embeddings = []
    for i in tqdm(range(0, len(reviews), batch_size), desc="Processing batches"):
        batch_reviews = reviews[i:i + batch_size]
        batch_embeddings = model.encode(batch_reviews, 
                                        convert_to_tensor=False,
                                        show_progress_bar=False,
                                        normalize_embeddings=True)
        embeddings.extend(batch_embeddings)
    embeddings = np.array(embeddings)
    print(f"Created embeddings with shape: {embeddings.shape}")
    # Create DataFrame with embeddings
    print("Preparing output DataFrame...")
    embedding_columns = [f'embedding_{i}' for i in range(embeddings.shape[1])]
    embedding_df = pd.DataFrame(embeddings, columns=embedding_columns)
    # Combine with original data
    result_df = pd.concat([df_unique[['Game', 'cleaned_Reviews']].reset_index(drop=True), embedding_df], axis=1)
    # Add additional useful columns
    result_df['num_words'] = result_df['cleaned_Reviews'].apply(lambda x: len(str(x).split()))
    result_df['review_length'] = result_df['cleaned_Reviews'].apply(lambda x: len(str(x)))
    # Reorder columns
    cols = ['Game', 'cleaned_Reviews', 'num_words', 'review_length'] + embedding_columns
    result_df = result_df[cols]
    # Save to CSV
    print(f"Saving embeddings to {output_file_path}...")
    result_df.to_csv(output_file_path, index=False)
    print("Embedding creation completed!")
    print(f"Final dataset shape: {result_df.shape}")
    print(f"Embedding dimensions: {len(embedding_columns)}")
    return result_df  # embedding_bertopic_v2.csv



# 1. Import library
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load data
df = pd.read_csv("embedding_bertopic_v2.csv")

# 3. Ambil hanya kolom embedding (diasumsikan dimulai dari kolom ke-4)
embedding_cols = [col for col in df.columns if col.startswith("embedding_")]
embeddings = df[embedding_cols].values

# 4. Reduksi dimensi dengan UMAP ke 2D
umap_model = umap.UMAP(n_components=2, random_state=42)
umap_embeddings = umap_model.fit_transform(embeddings)

# 5. Tambahkan hasil UMAP ke dataframe
df['UMAP_1'] = umap_embeddings[:, 0]
df['UMAP_2'] = umap_embeddings[:, 1]

df.to_csv("embedding_umap.csv", index=False)
# Baca file CSV
df = pd.read_csv("embedding_umap.csv")

# Hapus kolom dari embedding_1 hingga embedding_383
embedding_columns = [f"embedding_{i}" for i in range(1, 384)]
df = df.drop(columns=embedding_columns)

# Simpan ke file baru (atau timpa file lama jika mau)
df.to_csv("embedding_umap.csv", index=False)


import pandas as pd
from hdbscan import HDBSCAN
from sklearn.preprocessing import normalize

# Load data embedding
embedding_df = pd.read_csv("embedding_umap.csv")

# Ambil embedding kolom yang sesuai (bisa embedding_0, UMAP_1, UMAP_2, dst)
X = embedding_df[['UMAP_1', 'UMAP_2']].values
X = normalize(X)

# Clustering
clusterer = HDBSCAN(min_cluster_size=5)
embedding_df['cluster'] = clusterer.fit_predict(X)

# Simpan ulang hasil
embedding_df.to_csv("embedding_umap_clustered.csv", index=False)

import pandas as pd

# -----------------------------
# STEP 1: Load data
# -----------------------------
embedding_df = pd.read_csv("embedding_umap_clustered.csv")        # berisi cleaned_Reviews dan cluster
preprocessed_df = pd.read_csv("preprocessed_Reviews.csv")         # berisi Game dan cleaned_Reviews
topic_keywords_df = pd.read_csv("final_topic_keywordsv2.csv")       # berisi cluster dan keywords (list/str)

# Ubah kolom keywords ke list jika belum
def parse_keywords(x):
    if isinstance(x, str):
        return eval(x) if x.startswith("[") else x.split(",")
    return x

topic_keywords_df['keywords'] = topic_keywords_df['keywords'].apply(parse_keywords)

# Buat mapping cluster -> keywords
cluster_keywords_map = dict(zip(topic_keywords_df['cluster'], topic_keywords_df['keywords']))

# -----------------------------
# STEP 2: Gabungkan cluster ke preprocessed_df
# -----------------------------
merged_df = pd.merge(preprocessed_df, embedding_df[['cleaned_Reviews', 'cluster']], on='cleaned_Reviews', how='left')

# -----------------------------
# STEP 3: Buang noise (cluster = -1)
# -----------------------------
merged_df = merged_df[merged_df['cluster'] != -1]

# -----------------------------
# STEP 4: Hitung jumlah game per cluster (untuk cari cluster umum)
# -----------------------------
game_per_cluster = merged_df.groupby('cluster')['Game'].nunique()
threshold = 0.10 * merged_df['Game'].nunique()  # misal: >10% game
common_clusters = game_per_cluster[game_per_cluster > threshold].index.tolist()

# -----------------------------
# STEP 5: Hitung topik dominan per game (tanpa common clusters)
# -----------------------------
topic_per_game = merged_df[~merged_df['cluster'].isin(common_clusters)] \
    .groupby(['Game', 'cluster']) \
    .size().reset_index(name='count')

top_topics = topic_per_game.sort_values(['Game', 'count'], ascending=[True, False])
top_topics = top_topics.groupby('Game').head(3)

# Gabungkan list cluster per game
top_topics_agg = top_topics.groupby('Game')['cluster'].apply(list).reset_index()
top_topics_agg.columns = ['Game', 'Top_Clusters']

# -----------------------------
# STEP 6: Mapping ke keywords
# -----------------------------
def map_clusters_to_keywords(cluster_list, keyword_map):
    return [keyword_map[c] for c in cluster_list if c in keyword_map]

top_topics_agg['Top_Topic_Keywords'] = top_topics_agg['Top_Clusters'].apply(
    lambda cl: map_clusters_to_keywords(cl, cluster_keywords_map)
)

# -----------------------------
# STEP 7: Simpan ke CSV
# -----------------------------
top_topics_agg.to_csv("game_top_topics_keywords.csv", index=False)
print("✅ File berhasil disimpan sebagai 'game_top_topics_keywords.csv'")



import pandas as pd
import glob

# 1. Gabungkan semua data pemain dari dua folder
csv_files = glob.glob("Dataset/*.csv") + glob.glob("Dataset_2/*.csv")
df_players = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# 2. Load topik dari game
df_topics = pd.read_csv("game_top_topics_keywords.csv")

# 3. Normalisasi nama game untuk pencocokan
df_players['Game Name'] = df_players['Game Name'].str.lower().str.strip()
df_topics['Game'] = df_topics['Game'].str.lower().str.strip()

# 4. Gabungkan berdasarkan nama game
df_merged = df_players.merge(
    df_topics[['Game', 'Top_Clusters']],
    left_on='Game Name',
    right_on='Game',
    how='left'
)

# 5. Ubah Top_Clusters ke list Python jika masih string list
df_merged['Top_Clusters'] = df_merged['Top_Clusters'].apply(
    lambda x: eval(x) if isinstance(x, str) and x.startswith("[") else []
)

# 6. Ubah nama kolom menjadi Topic
df_merged = df_merged.rename(columns={'Top_Clusters': 'Topic'})

# 7. Pilih kolom yang diinginkan
df_result = df_merged[['Steam ID', 'App ID', 'Game Name', 'Playtime (hours)', 'Genres', 'Achievements', 'Topic']]

# 8. Konversi list ke string agar tidak pecah di CSV
df_result['Topic'] = df_result['Topic'].apply(lambda x: str(x))

# 9. Simpan ke file
df_result.to_csv("combined_player_game_with_topics.csv", index=False)



import pandas as pd

# Baca data hasil merge yang sudah memiliki kolom Dominant_Topic per game
df = pd.read_csv("combined_player_game_with_topics.csv")

# Tampilkan 5 baris pertama
print("Data gabungan combined_player_game_with_topics.csv:")
display(df.head())

# Jumlah baris awal
initial_rows = len(df)
print(f"Jumlah baris awal: {initial_rows}")

# Hapus baris dengan nilai NaN di kolom-kolom penting
required_columns = ["Steam ID", "App ID", "Game Name", "Playtime (hours)", "Genres", "Achievements"]
df = df.dropna(subset=required_columns)

# Jumlah baris setelah pembersihan
cleaned_rows = len(df)
print(f"Jumlah baris setelah menghapus NaN dan kolom 'Similarity_Score': {cleaned_rows}")


# Simpan jumlah baris sebelum penghapusan game nonaktif
before_removal = len(df)

# Filter: hanya ambil game dengan playtime >= 10 menit (yaitu >= 0.166 jam)
df = df[df["Playtime (hours)"] >= 0.166]

# Jumlah baris setelah penghapusan
after_removal = len(df)

print(f"Jumlah baris sebelum penghapusan game nonaktif: {before_removal}")
print(f"Jumlah baris setelah penghapusan game nonaktif: {after_removal}")


# Simpan jumlah baris sebelum penghapusan
before_removal = len(df)

# Hapus game dengan nama 'Unknown'
df = df[df["Game Name"].str.lower() != "unknown"]

# Simpan jumlah baris setelah penghapusan
after_removal = len(df)

print(f"Jumlah baris sebelum penghapusan game 'Unknown': {before_removal}")
print(f"Jumlah baris setelah penghapusan game 'Unknown': {after_removal}")


# Simpan jumlah baris sebelum penghapusan
before_dedup = len(df)

# Hapus data duplikat
df = df.drop_duplicates()

# Simpan jumlah baris setelah penghapusan
after_dedup = len(df)

print(f"Jumlah baris sebelum penghapusan duplikat: {before_dedup}")
print(f"Jumlah baris setelah penghapusan duplikat: {after_dedup}")


# Hitung jumlah game per SteamID
game_counts = df['Steam ID'].value_counts()

# Tambahkan kolom baru 'Game' berdasarkan jumlah game dari masing-masing SteamID
df['Game'] = df['Steam ID'].map(game_counts)

# Cek hasil
df[['Steam ID', 'Game']].drop_duplicates().head()

# Hitung Q1 (kuartil pertama) dari jumlah game
q1 = df['Game'].quantile(0.25)

# Tampilkan Q1 untuk informasi
print(f"Q1 Jumlah Game: {q1}")

# Filter data: hanya ambil pemain yang jumlah game-nya >= Q1
df = df[df['Game'] >= q1].reset_index(drop=True)

# Cek jumlah data setelah filter
df['Steam ID'].nunique(), df.shape


# 1. Pecah genre menjadi list
df['Genre_List'] = df['Genres'].fillna('').apply(lambda x: [g.strip() for g in x.split(',') if g.strip() != ''])

# 2. Ambil semua genre unik dari seluruh list
from itertools import chain

all_genres = set(chain.from_iterable(df['Genre_List']))
genre_mapping = {genre: i+1 for i, genre in enumerate(sorted(all_genres))}

# 3. Ganti list genre jadi list kode
df['Genre_Code_List'] = df['Genre_List'].apply(lambda genre_list: [genre_mapping[g] for g in genre_list if g in genre_mapping])

# 4. Simpan mapping ke CSV
genre_df = pd.DataFrame(list(genre_mapping.items()), columns=['Genre', 'Genre_Code'])
genre_df.to_csv('genre_code_mapping.csv', index=False)

# 5. Simpan data utama
df.to_csv('cleaned_with_genre_codes.csv', index=False)

import pandas as pd
import numpy as np

# 1. Load data
df_emb = pd.read_csv('embedding_bertopic_v2.csv')              # 'Game', 'cleaned_Reviews', embedding_*
df_asg = pd.read_csv('hasil_topic_assignment_automerged.csv')  # 'game', 'cleaned_reviews', 'topic', UMAP_*, dll.

# 2. Merge
df = pd.merge(
    df_emb,
    df_asg[['cleaned_Reviews', 'topic']],
    on='cleaned_Reviews',
    how='inner'
)
df = df.drop(columns=['cleaned_Reviews'])

# 3. Kolom embedding
emb_cols = [c for c in df.columns if c.startswith('embedding_')]

# 4. Hitung centroid per topic (kecuali -1)
centroids = df[df['topic'] != -1].groupby('topic')[emb_cols].mean()
centroids = centroids.div(np.linalg.norm(centroids, axis=1), axis=0)

# 5. Normalisasi semua embedding review
emb_norm = df[emb_cols].div(np.linalg.norm(df[emb_cols], axis=1), axis=0).values

# 6. Hitung similarity ke semua centroid
sim_matrix = np.dot(emb_norm, centroids.values.T)  # shape: (n_reviews, n_topics)
topic_labels = centroids.index.to_list()

# 7. Ambil topik dominan untuk setiap Game
df_sim = pd.DataFrame(sim_matrix, columns=topic_labels)
df_sim['Game'] = df['Game']

# 8. Ambil rata-rata similarity per game-topic
gt_sim = df_sim.melt(id_vars='Game', var_name='topic', value_name='similarity')
gt_mean = gt_sim.groupby(['Game', 'topic'])['similarity'].mean().reset_index()

# 9. Tambahkan penalti untuk topik yang terlalu umum (misal topik 0)
topic_counts = gt_mean['topic'].value_counts(normalize=True)
penalty_factor = 1 - topic_counts  # topik yang sering muncul → penalti lebih besar
gt_mean['penalized_sim'] = gt_mean.apply(
    lambda r: r['similarity'] * penalty_factor.get(r['topic'], 1),
    axis=1
)

# 10. Pilih topic dengan penalized_sim tertinggi
idx = gt_mean.groupby('Game')['penalized_sim'].idxmax()
dominant = gt_mean.loc[idx, ['Game', 'topic', 'similarity']].reset_index(drop=True)

# 11. Rename kolom
dominant = dominant.rename(columns={
    'topic': 'Dominant_Topic',
    'similarity': 'Similarity_Score'
})

# 12. Simpan hasil
dominant.to_csv('dominant_topic_per_game.csv', index=False)
print(dominant.head())


import pandas as pd
import numpy as np
from collections import Counter

# 1. Load data
df_dom = pd.read_csv("dominant_topic_per_game.csv")   # Game, Dominant_Topic, Similarity_Score
df_seg = pd.read_csv("cleaned_with_genre_codes.csv")     # Steam ID, Game Name, Playtime (hours), Genres, Achievements

# 2. Standardisasi nama game agar cocok
df_seg["Game Name"] = df_seg["Game Name"].str.strip().str.lower()
df_dom["Game"]       = df_dom["Game"].str.strip().str.lower()

# 3. Merge untuk mendapatkan Dominant_Topic per baris game–pemain
merged = pd.merge(
    df_seg, df_dom[["Game","Dominant_Topic"]],
    left_on="Game Name", right_on="Game",
    how="inner"
)

# 4. Filter invalid dan hanya pemain yang main ≥10 menit
merged = merged[merged["Game Name"] != "unknown"]
merged = merged[merged["Playtime (hours)"] >= 0.167]

# 5. Buang NaN & duplikat
merged.dropna(subset=[
    "Steam ID","Game Name","Playtime (hours)",
    "Genres","Achievements","Dominant_Topic"
], inplace=True)
merged.drop_duplicates(subset=["Steam ID","Game Name"], inplace=True)

# 6. Hanya pemain aktif (>= kuartil 1 total game)
game_counts = (
    merged
    .groupby("Steam ID")["Game Name"]
    .nunique()
    .reset_index(name="Total_Games")
)
q1 = game_counts["Total_Games"].quantile(0.25)
active_ids = game_counts[game_counts["Total_Games"] >= q1]["Steam ID"]
merged = merged[merged["Steam ID"].isin(active_ids)]

# 7. Hitung Total_Achievements & Avg_Playtime
achievement_sum = (
    merged
    .groupby("Steam ID")["Achievements"]
    .sum()
    .reset_index(name="Total_Achievements")
)
playtime_avg = (
    merged
    .groupby("Steam ID")["Playtime (hours)"]
    .mean()
    .reset_index(name="Avg_Playtime")
)

# 8. Mapping Genres → kode numerik
merged["Genres"] = merged["Genres"].astype(str)
genre_list = sorted({g for gs in merged["Genres"] for g in gs.split(", ")})
genre_map  = {g:i+1 for i,g in enumerate(genre_list)}
merged["Genre_Code"] = merged["Genres"].apply(
    lambda x: [genre_map[g] for g in x.split(", ") if g in genre_map]
)

# 9. Buat fungsi top_n dan ambil Top 3 Genre per pemain
def top_n(seq, n=3):
    c = Counter(seq)
    return [item for item, _ in c.most_common(n)]

top_genres = (
    merged
    .groupby("Steam ID")["Genre_Code"]
    .apply(lambda lists: top_n([g for sub in lists for g in sub], 3))
    .reset_index(name="Top_3_Genres")
)

# 10. Ambil Dominant_Topic bersih per pemain
# Skip topics -1 (noise) and 0
topic_lists = (
    merged
    .groupby("Steam ID")["Dominant_Topic"]
    .apply(lambda ts: [t for t,_ in Counter(ts).most_common()])
)
def choose_topic(lst):
    for t in lst:
        if t not in [-1, 0]:  # skip noise and topic 0
            return t
    return np.nan

dominant_topic = (
    topic_lists
    .apply(choose_topic)
    .reset_index(name="Topic Dominan")
)

# 11. Gabungkan semua ke summary
summary = (
    game_counts
    .merge(achievement_sum, on="Steam ID")
    .merge(playtime_avg,    on="Steam ID")
    .merge(dominant_topic,  on="Steam ID")
    .merge(top_genres,      on="Steam ID")
)

# 12. Explode Top_3_Genres → satu baris per genre dominan
exploded = (
    summary
    .explode("Top_3_Genres")
    .rename(columns={
        "Steam ID":         "Steam ID",
        "Total_Games":      "Total Game",
        "Avg_Playtime":     "Total Playtime",
        "Total_Achievements":"Total Achievement",
        "Top_3_Genres":     "Genre Dominan"
    })
)

# 13. Tambah kolom No
exploded.insert(0, "No", range(1, len(exploded) + 1))

# 14. Simpan hasil
exploded.to_csv("transformation_segmentation_v4.csv", index=False)
print("✅ Selesai. Hasil dengan Top 3 Genre explode tersimpan di 'transformation_segmentation_v4.csv'")


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1. Load dan bersihkan header
df = pd.read_csv('transformation_segmentation_v4.csv')
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(' ', '_')
)
# Sekarang df.columns → ['no', 'steam_id', 'total_game',
#                         'total_achievement', 'total_playtime',
#                         'topic_dominan', 'genre_dominan']

# 2. Pilih kolom dasar sebagai user_base
#    steam_id, total_game, total_achievement, total_playtime
user_base = (
    df[['steam_id','total_game','total_achievement','total_playtime']]
    .drop_duplicates(subset='steam_id')
    .set_index('steam_id')
)

# 3. Pivot frekuensi topik & genre per user
topic_pivot = (
    df.groupby(['steam_id','topic_dominan'])
      .size()
      .unstack(fill_value=0)
      .add_prefix('topic_count_')
)
genre_pivot = (
    df.groupby(['steam_id','genre_dominan'])
      .size()
      .unstack(fill_value=0)
      .add_prefix('genre_count_')
)

# 4. Gabungkan semua fitur ke satu DataFrame
user_features = (
    user_base
    .join(topic_pivot, how='left').fillna(0)
    .join(genre_pivot, how='left').fillna(0)
)

# 5. Hitung proporsi tiap count relatif ke total_game
count_cols = [c for c in user_features.columns if c.startswith(('topic_count_','genre_count_'))]
for c in count_cols:
    user_features[c + '_prop'] = user_features[c] / user_features['total_game']

# 6. Siapkan daftar kolom untuk normalisasi
norm_cols = (
    count_cols +
    [c + '_prop' for c in count_cols] +
    ['total_game','total_achievement','total_playtime']
)

# 7. Konversi ke float & normalisasi MinMax
user_features[norm_cols] = user_features[norm_cols].astype(float)
scaler = MinMaxScaler()
user_features[norm_cols] = scaler.fit_transform(user_features[norm_cols])

# 8. Reset index & simpan hasil
user_features = user_features.reset_index()
user_features.to_csv('user_features_for_AA_v2.csv', index=False)

print("✅ Selesai. user_features_for_AA_v2.csv shape:", user_features.shape)


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import nnls
from collections import Counter
import matplotlib.pyplot as plt
import umap
from kneed import KneeLocator

# ===================================
# 1) Load & Prepare Data for AA
# ===================================
df_feats = pd.read_csv("user_features_for_AA_v2.csv", dtype={"steam_id": str})
steam_ids = df_feats["steam_id"].tolist()

# Keep basics for later
total_games = df_feats.set_index("steam_id")["total_game"].to_dict()
total_playtime = df_feats.set_index("steam_id")["total_playtime"].to_dict()  # ✅ tambahan
total_achievements = df_feats.set_index("steam_id")["total_achievement"].to_dict()

# Build and scale feature matrix X
feature_cols = [c for c in df_feats.columns if c != "steam_id"]
X = df_feats[feature_cols].values
X_scaled = MinMaxScaler().fit_transform(X)

# ===================================
# 2) Load Raw Segmentation & Features
# ===================================
df_raw = pd.read_csv("transformation_segmentation_v4.csv", dtype={"Steam ID": str})
df_raw.columns = (
    df_raw.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
)
df_raw = df_raw[df_raw["steam_id"].isin(steam_ids)]

dominant_topic = df_raw.groupby("steam_id")["topic_dominan"].first().to_dict()

tmp = df_raw.groupby("steam_id")["genre_dominan"] \
            .apply(lambda xs: [g for g, _ in Counter(xs).most_common(3)])
top3_genres = {
    sid: tmp.get(sid, [])[:3] + [np.nan] * (3 - len(tmp.get(sid, [])))
    for sid in steam_ids
}

# ===================================
# 3) Archetypal Analysis Function
# ====================================
def compute_normed_A(X, K, max_iter=100, tol=1e-4):
    n, d = X.shape
    rng = np.random.RandomState(42)
    # Initialize archetypes randomly
    Z = X[rng.choice(n, K, replace=(K > n))].copy()

    for _ in range(max_iter):
        # Solve for weights A via non-negative least squares
        A = np.zeros((n, K))
        for i in range(n):
            A[i], _ = nnls(Z.T, X[i])

        # Update archetypes Z
        Z_new = np.zeros_like(Z)
        for j in range(K):
            w = A[:, j][:, None]
            if w.sum() > 0:
                Z_new[j] = (w * X).sum(axis=0) / w.sum()
            else:
                Z_new[j] = Z[j]

        # Check convergence
        if np.linalg.norm(Z - Z_new) < tol:
            break
        Z = Z_new

    # Normalize rows of A to sum to 1
    A_norm = A / A.sum(axis=1, keepdims=True)
    return A_norm, Z

# ===================================
# 4) Determine Best K via Elbow Method
# ====================================
min_K, max_K = 2, 12
Ks = list(range(min_K, max_K + 1))
errors = []

for K in Ks:
    A_norm_k, Z_k = compute_normed_A(X_scaled, K)
    # Reconstruction error (Frobenius norm)
    X_recon = A_norm_k.dot(Z_k)
    errors.append(np.linalg.norm(X_scaled - X_recon, 'fro'))

# Use KneeLocator to find elbow point
kl = KneeLocator(Ks, errors, curve='convex', direction='decreasing')
best_K = kl.elbow or Ks[np.argmin(errors)]
print(f"Selected best K: {best_K}")

# Plot the error curve with elbow marked
plt.figure(figsize=(6,4))
plt.plot(Ks, errors, 'o-', label='Reconstruction Error')
if kl.elbow:
    plt.axvline(kl.elbow, color='red', linestyle='--', label=f'Elbow at K={kl.elbow}')
plt.xlabel('K (Number of Archetypes)')
plt.ylabel('Frobenius Norm Error')
plt.title('Elbow Method for Archetypal Analysis')
plt.legend()
plt.tight_layout()
plt.show()

# ===================================
# 5) Compute Final AA with Best K
# ====================================
A_norm, Z = compute_normed_A(X_scaled, best_K)

# ===================================
# 6) Save Archetype Membership (Dominant Only)
# ====================================
dominant_archetypes = np.argmax(A_norm, axis=1) + 1  # 1-based index

# Build dictionary archetype -> list of steam_ids
dominant_membership = {f"archetype_{k+1}": [] for k in range(best_K)}
for sid, arc in zip(steam_ids, dominant_archetypes):
    dominant_membership[f"archetype_{arc}"].append(sid)

# Pad lists to equal length
max_len = max(len(v) for v in dominant_membership.values())
for k in dominant_membership:
    dominant_membership[k] += [np.nan] * (max_len - len(dominant_membership[k]))

# Save to CSV
pd.DataFrame(dominant_membership).to_csv("archetype_membership.csv", index=False)

# ===================================
# 7) Save Segmentation per User
# ====================================
rows = []
for idx, sid in enumerate(steam_ids):
    t1, t2, t3 = top3_genres[sid]
    row = {
        "steam_id": sid,
        "total_game": total_games[sid],
        "total_playtime": total_playtime[sid],  # ✅ baris tambahan
        "total_achievement": total_achievements[sid],
        "dominant_topic": dominant_topic.get(sid, np.nan),
        "top_1_genre": t1,
        "top_2_genre": t2,
        "top_3_genre": t3
    }
    # Add archetype weights
    for k in range(best_K):
        row[f"archetype_{k+1}_weight"] = A_norm[idx, k]
    rows.append(row)

# Create DataFrame and save
pd.DataFrame(rows).to_csv("archetypal_segments_elbowK_top3.csv", index=False)

# ===================================
# 8) UMAP Visualization of Archetype Segments
# ====================================
reducer = umap.UMAP(n_components=2, random_state=42)
coords_umap = reducer.fit_transform(A_norm)
dominant_arc = np.argmax(A_norm, axis=1) + 1

plt.figure(figsize=(8,6))
for k in range(1, best_K + 1):
    mask = dominant_arc == k
    plt.scatter(
        coords_umap[mask, 0],
        coords_umap[mask, 1],
        label=f"Archetype {k}",
        alpha=0.7
    )
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("User Segments by Archetype (UMAP Projection)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
