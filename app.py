import streamlit as st
import torch
from pathlib import Path
from typing import List, Dict
import base64
from PIL import Image
import io

# Import the OptimizedVectorDB class from vector_db.py
from vector_db import OptimizedVectorDB

def create_search_interface():
    """
    Creates and manages the main Streamlit interface for the document search system.
    
    This function sets up:
    1. Page configuration and styling
    2. Document upload interface
    3. Search interface with query input and type selection
    4. Results display with expandable sections
    
    The interface is designed to be intuitive and user-friendly while providing
    detailed information about search results and document processing status.
    """
    
    # Configure the Streamlit page with a dark theme and wide layout
    st.set_page_config(
        page_title="LOCAL DOCUMENT RAG BY TALIB SAYYED",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply custom CSS styling for dark theme and consistent UI elements
    st.markdown("""
        <style>
        /* Dark theme background and text colors */
        .stApp {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        /* Style input fields for better visibility */
        .stTextInput > div > div > input {
            background-color: #2D2D2D;
            color: #FFFFFF;
        }
        /* Custom button styling */
        .stButton > button {
            background-color: #4A4A4A;
            color: #FFFFFF;
            width: 100%;
        }
        /* Results container styling */
        .search-results {
            background-color: #2D2D2D;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Display main title
    st.title("LOCAL DOCUMENT RAG BY TALIB SAYYED")
    
    # Initialize vector database with GPU support if available
    @st.cache_resource
    def initialize_vector_db():
        """
        Initializes the vector database with appropriate hardware acceleration.
        The database is cached to prevent reinitialization on page rerun.
        """
        use_gpu = torch.cuda.is_available()
        return OptimizedVectorDB(
            persist_directory="vector_db",
            batch_size=32,
            use_gpu=use_gpu
        )
    
    vector_db = initialize_vector_db()

    # Create sidebar for document upload functionality
    with st.sidebar:
        st.header("Document Upload")
        uploaded_files = st.file_uploader(
            "Upload documents (PDF/Images)", 
            accept_multiple_files=True,
            type=['pdf', 'png', 'jpg', 'jpeg', 'tiff']
        )
        
        # Handle document indexing when files are uploaded
        if uploaded_files:
            if st.button("Index Documents"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Create temporary storage for uploaded files
                upload_dir = Path("temp_uploads")
                upload_dir.mkdir(exist_ok=True)
                
                # Process each uploaded file
                for i, file in enumerate(uploaded_files):
                    progress = (i + 1) / len(uploaded_files)
                    file_path = upload_dir / file.name
                    
                    # Save uploaded file
                    with open(file_path, "wb") as f:
                        f.write(file.getvalue())
                    
                    # Update progress indicators
                    status_text.text(f"Processing {file.name}...")
                    vector_db.index_documents(str(upload_dir))
                    progress_bar.progress(progress)
                
                status_text.text("Indexing complete!")

    # Create main search interface with two columns
    search_col, results_col = st.columns([1, 2])

    # Search input column
    with search_col:
        st.subheader("Search Query")
        query = st.text_input("Enter your search query:", key="search_query")
        search_type = st.radio(
            "Search Type:",
            ["Semantic Search", "Exact Match"],
            horizontal=True,
            help="Semantic Search finds related content, Exact Match finds specific phrases"
        )
        
        # Handle search execution
        if st.button("Search"):
            if query:
                exact_match = search_type == "Exact Match"
                results = vector_db.search(query, exact_match_only=exact_match)
                
                # Store results in session state for display
                st.session_state.search_results = results
                st.session_state.search_status = "Search completed"
            else:
                st.warning("Please enter a search query")

    # Results display column
    with results_col:
        st.subheader("Search Status")
        status = st.empty()
        
        # Show search status if available
        if 'search_status' in st.session_state:
            status.text(st.session_state.search_status)
        
        # Display search results in expandable sections
        st.subheader("Retrieved Text with Similarity Scores")
        if 'search_results' in st.session_state and st.session_state.search_results:
            for result in st.session_state.search_results:
                with st.expander(f"Result from {result['file_name']} (Page {result['page_number']})"):
                    st.write(f"**Relevance Score:** {result['relevance_score']:.3f}")
                    st.write(f"**OCR Confidence:** {result['confidence']:.1f}%")
                    st.write("**Text:**")
                    st.write(result['text'])

if __name__ == "__main__":
    create_search_interface()