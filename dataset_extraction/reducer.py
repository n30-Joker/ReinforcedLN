import pandas as pd

# Load dataset
file_path = "light_novels.csv"
df = pd.read_csv(file_path)

# Step 1: Remove duplicate entries based on 'ID'
df = df.drop_duplicates(subset=["ID"], keep="first")

# Step 2: Ensure fair representation across genres
df["Genres"] = df["Genres"].fillna("")  # Handle missing genres

# Identify unique genres
unique_genres = set(g for sublist in df["Genres"].str.split(", ") for g in sublist if g)

# Define target dataset size
target_size = 2000
synopsis_ratio = 0.8  # 80% with synopsis, 20% without
with_synopsis_target = int(target_size * synopsis_ratio)
without_synopsis_target = target_size - with_synopsis_target

# Split dataset into those with and without synopses
df_with_synopsis = df[df["Synopsis"].notna() & df["Synopsis"].str.strip().ne("")]
df_without_synopsis = df[df["Synopsis"].isna() | df["Synopsis"].str.strip().eq("")]

# Initialize final dataset
final_df = pd.DataFrame()

# Step 3: Distribute samples fairly across genres
for genre in unique_genres:
    genre_df = df[df["Genres"].str.contains(genre, na=False, regex=False)]
    
    # Separate those with and without synopsis
    genre_with_synopsis = genre_df[genre_df["Synopsis"].notna() & genre_df["Synopsis"].str.strip().ne("")]
    genre_without_synopsis = genre_df[genre_df["Synopsis"].isna() | genre_df["Synopsis"].str.strip().eq("")]
    
    # Determine fair allocation per genre
    per_genre_limit = target_size // len(unique_genres)
    with_synopsis_limit = int(per_genre_limit * synopsis_ratio)
    without_synopsis_limit = per_genre_limit - with_synopsis_limit
    
    # Sample based on limits (avoid exceeding available count)
    sampled_with_synopsis = genre_with_synopsis.sample(n=min(len(genre_with_synopsis), with_synopsis_limit), random_state=42)
    sampled_without_synopsis = genre_without_synopsis.sample(n=min(len(genre_without_synopsis), without_synopsis_limit), random_state=42)
    
    # Add samples to the final dataset
    final_df = pd.concat([final_df, sampled_with_synopsis, sampled_without_synopsis])

# Step 4: Fill remaining slots to get as close to 2000 as possible
remaining_slots = target_size - len(final_df)
if remaining_slots > 0:
    additional_samples = df_with_synopsis.sample(n=min(len(df_with_synopsis), remaining_slots), random_state=42)
    final_df = pd.concat([final_df, additional_samples])

# Ensure final dataset is within target size
final_df = final_df.sample(n=min(len(final_df), target_size), random_state=42).reset_index(drop=True)

# Save to CSV
final_df.to_csv("reduced_light_novels.csv", index=False)

# Output final dataset size
print(f"Final dataset size: {len(final_df)} (Target: {target_size})")
