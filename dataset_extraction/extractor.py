import requests
import csv

def fetch_light_novels():
    client_id = "07b887779ad5700c393a8599d7c94457"
    base_url = "https://api.myanimelist.net/v2/manga/ranking"
    headers = {"X-MAL-CLIENT-ID": client_id}
    offset = 0
    limit = 500  # Max limit per request
    all_data = []
    
    while True:
        params = {
            "ranking_type": "lightnovels",
            "fields": "id,title,mean,rank,popularity,genres,synopsis,main_picture",
            "limit": limit,
            "offset": offset,
            "nsfw": "true"
        }
        
        response = requests.get(base_url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            break
        
        data = response.json()
        
        if "data" not in data or not data["data"]:
            break  # Exit if no more data available
        
        for item in data["data"]:
            manga = item["node"]
            genres = ", ".join([genre["name"] for genre in manga.get("genres", [])])
            
            all_data.append([
                manga.get("id", "N/A"),
                manga.get("title", "N/A"),
                manga.get("main_picture", {}).get("medium", "N/A"),
                manga.get("main_picture", {}).get("large", "N/A"),
                manga.get("mean", "N/A"),
                manga.get("rank", "N/A"),
                manga.get("popularity", "N/A"),
                genres,
                manga.get("synopsis", "N/A").replace("\n", " ")
            ])
        
        offset += limit
    
    return all_data

def save_to_csv(data, filename="light_novels.csv"):
    headers = ["ID", "Title", "Medium_Picture", "Large_Picture", "Mean", "Rank", "Popularity", "Genres", "Synopsis"]
    
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)
    
    print(f"Data successfully saved to {filename}")

if __name__ == "__main__":
    light_novels = fetch_light_novels()
    if light_novels:
        save_to_csv(light_novels)
    else:
        print("No data retrieved.")
