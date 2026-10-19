import chromadb
import numpy as np
import umap
import hdbscan
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer

def get_cluster_keywords(documents, cluster_labels):
    """Hittar de mest beskrivande orden för varje kluster."""
    df = pd.DataFrame({'doc': documents, 'cluster': cluster_labels})
    cluster_names = {}
    
    # Filtrerar bort vanliga engelska utfyllnadsord (använd 'swedish' om din data är på svenska)
    vectorizer = TfidfVectorizer(max_features=3, stop_words='english')
    
    for c in set(cluster_labels):
        if c == -1:
            cluster_names[c] = "Brus (Oklassificerat)"
            continue
            
        cluster_docs = df[df['cluster'] == c]['doc'].tolist()
        
        try:
            # Räkna ut vilka ord som är mest unika för just detta kluster
            vectorizer.fit_transform(cluster_docs)
            words = vectorizer.get_feature_names_out()
            cluster_names[c] = " + ".join(words).title()
        except Exception:
            cluster_names[c] = f"Kategori {c}"
            
    return cluster_names

def visualize_interactive():
    print("Hämtar data från ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="snuffles_docs")
    
    data = collection.get(include=["embeddings", "documents", "metadatas"])
    
    if data["embeddings"] is None or len(data["embeddings"]) == 0:
        print("Databasen är tom!")
        return
        
    embeddings = np.array(data["embeddings"])
    documents = data["documents"]
    filenames = [meta.get("filename", "Okänd fil") for meta in data["metadatas"]]
    
    print("Reducerar dimensioner med UMAP...")
    umap_model = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric='cosine', random_state=42)
    data_2d = umap_model.fit_transform(embeddings)

    print("Klustrar data med HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2, metric='euclidean')
    cluster_labels = clusterer.fit_predict(data_2d)
    
    print("Extraherar kategorinamn baserat på innehåll...")
    cluster_name_map = get_cluster_keywords(documents, cluster_labels)
    
    # Ersätt de gamla gruppnamnen med de faktiska nyckelorden
    labels = [cluster_name_map[c] for c in cluster_labels]
    short_docs = [doc[:100] + "..." if len(doc) > 100 else doc for doc in documents]

    print("Skapar interaktiv vy...")
    df = pd.DataFrame({
        'X': data_2d[:, 0],
        'Y': data_2d[:, 1],
        'Kategori': labels,
        'Fil': filenames,
        'Text': short_docs
    })

    # Skapa grafen och dölj X/Y-koordinaterna i hover-rutan för ett renare utseende
    fig = px.scatter(
        df, x='X', y='Y', color='Kategori',
        hover_data={'X': False, 'Y': False, 'Fil': True, 'Kategori': False, 'Text': True},
        title='Interaktiv Dokumentkarta med Kategorier'
    )
    
    fig.update_layout(template="plotly_dark")
    fig.show()

if __name__ == "__main__":
    visualize_interactive()