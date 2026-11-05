import streamlit as st
import snufflesAi
import io
from contextlib import redirect_stdout



# Ställ in flikens namn och titel på sidan
st.set_page_config(page_title="Snuffles AI", page_icon="🤖")
st.title("Snuffles")

# Skapa ett minne för chattmeddelandena om det inte redan finns
if "messages" not in st.session_state:
    st.session_state.messages = []

if "terminal_log" not in st.session_state:
    st.session_state.terminal_log = ""

st.sidebar.title("Terminal")
# Skapa en tom behållare i sidopanelen som vi kan skriva till
terminal_container = st.sidebar.empty()
# Rita ut den historiska loggen formaterad som kod
terminal_container.code(st.session_state.terminal_log, language="bash")

# --- Systeminfo i sidopanelen ---
st.sidebar.markdown("---") # Skapar en visuell avdelare i sidopanelen
st.sidebar.subheader("OMLX Info")
token_chat_ui = st.sidebar.empty()
token_rerank_ui = st.sidebar.empty()
token_embed_ui = st.sidebar.empty()

def update_token_metrics():
    """Uppdaterar sidopanelen med de senaste värdena från snufflesAi"""
    token_chat_ui.markdown(f"**Chatt:** `{snufflesAi.token_usage['chat']}` tok")
    token_rerank_ui.markdown(f"**Reranker:** `{snufflesAi.token_usage['reranker']}` tok")
    token_embed_ui.markdown(f"**Embedding:** `{snufflesAi.token_usage['embedding']}` tok")

# Rita ut mätarna med nuvarande värden
update_token_metrics()
# Gå igenom och visa alla sparade meddelanden
for message in st.session_state.messages:
    # Välj ikon baserat på vem som pratar (byt ut mot emojis om du saknar bildfiler)
    avatar_icon = "images/morty_avatar.png" if message["role"] == "user" else "images/snuffles_image.png"
    
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])

    # Skapa inmatningsfältet
if prompt := st.chat_input("I think he's saying i love lasagna..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user", avatar="images/morty_avatar.png"):
        st.markdown(prompt)
        
    # Anropa Snuffles och aktivera GUI-flaggan
    with st.chat_message("assistant", avatar="images/snuffles_image.png"):

        # Skapa en virtuell fil för att fånga upp alla print()
        terminal_output = io.StringIO()
        
        # Fånga alla utskrifter exakt när funktionen körs
        with redirect_stdout(terminal_output):
            stream_data, sources = snufflesAi.chat_with_gemma(prompt, is_gui=True)
            
        # Hämta den fångade texten och lägg till den i minnet
        ny_logg = terminal_output.getvalue()
        if ny_logg:
            st.session_state.terminal_log += ny_logg
            # Uppdatera rutan i sidopanelen direkt
            terminal_container.code(st.session_state.terminal_log, language="bash")

        
        stream_data, sources = snufflesAi.chat_with_gemma(prompt, is_gui=True)
        
        if isinstance(stream_data, str): 
            # Hanterar felmeddelandet om servern är nere
            st.error(stream_data)
            response_text = stream_data
        else:
            # Låter Streamlit strömma svaret i realtid
            response_text = st.write_stream(stream_data)
            
            # Skriv ut RAG-källorna om de fanns
            if sources:
                st.markdown("\n\nSources:")
                for src in sources:
                    st.markdown(f"- {src}")
        
        # Spara Snuffles slutgiltiga svar i webbminnet
        st.session_state.messages.append({"role": "assistant", "content": response_text})

