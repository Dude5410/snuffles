import chromadb
import numpy as np
import umap
import hdbscan
import datamapplot
import matplotlib.pyplot as plt

def visualize_chromadb():
    # 1. Anslut till din befintliga ChromaDB
    print("Hämtar data från ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="snuffles_docs")
    
    # Hämta dokument, metadata och inbäddningar (embeddings)
    data = collection.get(include=["embeddings", "documents", "metadatas"])
    
    if data["embeddings"] is None or len(data["embeddings"]) == 0:
        print("Databasen är tom! Kör vectorizing_folder.py först.")
        return
        
    embeddings = np.array(data["embeddings"])
    documents = data["documents"]
    
    print(f"Hittade {len(embeddings)} text-stycken. Reducerar dimensioner med UMAP...")

    # 2. Minska dimensionerna från modellens storlek ner till 2D (X, Y)
    umap_model = umap.UMAP(
        n_neighbors=15,      # Justera uppåt om kartan blir för "utsmetad"
        n_components=2, 
        min_dist=0.1,        # Hur nära punkterna får ligga varandra
        metric='cosine',     # Cosine är bäst för textinbäddningar
        random_state=42
    )
    data_2d = umap_model.fit_transform(embeddings)

    print("Klustergrupperar data med HDBSCAN...")

    # 3. Hitta kluster (grupper av semantiskt liknande textstycken)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,  # Minsta antal textstycken för att bilda ett kluster
        min_samples=2,
        metric='euclidean'
    )
    cluster_labels = clusterer.fit_predict(data_2d)

    # Generera grundläggande etiketter för varje kluster
    labels = np.array([f"Ämne {c}" if c != -1 else "Brus (Oklassificerat)" for c in cluster_labels])

    print("Skapar visualiseringen med DataMapPlot...")

    # 4. Rita ut kartan med DataMapPlot (med "dark mode" för rätt känsla)
    fig, ax = datamapplot.create_plot(
        data_2d,
        labels,
        title="Snuffles Dokumentkarta",
        sub_title="2D-visualisering av vektoriserade PDF:er och filer",
        darkmode=True,      # Ger den mörka bakgrunden och glöden
        label_font_size=9,
        use_medoids=True    # Centrerar etiketterna snyggare
    )

    # Visa grafen på skärmen
    plt.show()
    
    # Valfritt: Spara bilden som en högupplöst fil
    # fig.savefig("snuffles_map.png", dpi=300, bbox_inches='tight')
    print("Visualisering klar!")

if __name__ == "__main__":
    visualize_chromadb()