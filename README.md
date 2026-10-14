# Snuffles: Local Multimodal RAG System 🐶

![Snuffles hunden](images/snuffles_image.png)

Snuffles är en helt lokal, privat och säker Retrieval-Augmented Generation (RAG) motor byggd i Python. Projektet syftar till att analysera dokument (PDF) och bilder genom att integrera lokala AI-modeller och vektordatabaser, med ett strikt fokus på dataintegritet.

## 🚀 Arkitektur och Teknikstack
- **100% Lokalt:** All data, indexering och vektorisering hanteras lokalt på maskinen utan externa molnanrop.
- **Vektordatabas:** [ChromaDB](https://www.trychroma.com/) för persistent lokal lagring av dokument-embeddings.
- **Modellserver:** [oMLX](https://github.com/omlx/omlx) körs lokalt och agerar inferensmotor.
- **Modeller:**
  - `embeddinggemma-2` för att omvandla text och dokument till sökbara vektorer.
  - `gemma-4` för slutgiltig textgenerering och multimodal analys.