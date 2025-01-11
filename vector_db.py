import os
from pathlib import Path
import logging
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from PIL import Image
import pytesseract
import cv2
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

@dataclass
class TextChunk:
    text: str
    file_name: str
    page_number: int
    chunk_index: int
    confidence: float = 0.0

class OptimizedVectorDB:
    def __init__(self, 
                 chunk_size: int = 512,
                 persist_directory: str = "vector_db",
                 batch_size: int = 32,
                 use_gpu: bool = None):
        logging.info("Initializing Optimized Vector DB...")
        
        # Auto-detect GPU if not specified
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        self.use_gpu = use_gpu
        self.device = 'cuda' if use_gpu else 'cpu'
        
        # Initialize with faster model and larger batch size
        self.encoder = SentenceTransformer('paraphrase-MiniLM-L3-v2', device=self.device)
        self.batch_size = batch_size
        
        # Initialize ChromaDB with optimized settings
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Configure embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name='paraphrase-MiniLM-L3-v2',
            device=self.device
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="document_chunks",
            embedding_function=self.embedding_function
        )
        
        self.chunk_size = chunk_size
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        self.document_extensions = {'.pdf'}
        self.processed_files = set()
        
        # Optimize PyTesseract configuration
        self.tesseract_config = '--oem 3 --psm 3 -c tessedit_create_hocr=1 textord_heavy_nr=1 textord_min_linesize=3'

    def enhance_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Optimized image enhancement"""
        if image.mode != 'L':
            image = image.convert('L')
        
        img_array = np.array(image)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img_array)
        
        # Denoise
        enhanced = cv2.fastNlMeansDenoising(enhanced)
        
        # Adaptive thresholding
        enhanced = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        return Image.fromarray(enhanced)

    def extract_text_from_image(self, image_path: str) -> Tuple[str, float]:
        """Optimized OCR processing"""
        try:
            image = Image.open(image_path)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            # Enhance image
            enhanced_image = self.enhance_image_for_ocr(image)
            
            # Single optimized OCR pass
            ocr_data = pytesseract.image_to_data(
                enhanced_image,
                output_type=pytesseract.Output.DICT,
                config=self.tesseract_config
            )
            
            # Process OCR results
            text_parts = []
            total_conf = 0
            word_count = 0
            
            for i, conf in enumerate(ocr_data['conf']):
                if conf > 0:  # Valid confidence score
                    word = ocr_data['text'][i].strip()
                    if word:
                        text_parts.append(word)
                        total_conf += conf
                        word_count += 1
            
            text = ' '.join(text_parts)
            avg_conf = total_conf / word_count if word_count > 0 else 0
            
            return text, avg_conf / 100.0
            
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {str(e)}")
            return "", 0.0

    def extract_text_from_pdf(self, pdf_path: str) -> List[Tuple[str, float]]:
        """Optimized PDF text extraction"""
        try:
            doc = fitz.open(pdf_path)
            results = []
            
            for page in doc:
                # Get text with formatting
                text = page.get_text("text", sort=True)
                blocks = page.get_text("blocks")
                
                # Calculate confidence based on text density and structure
                word_count = len(text.split())
                block_count = len(blocks)
                conf = min(1.0, word_count / 500) if block_count > 0 else 0.5
                
                results.append((text, conf))
            
            doc.close()
            return results
        except Exception as e:
            logging.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return []

    def chunk_text(self, text: str, file_name: str, page_number: int, confidence: float = 1.0) -> List[Dict]:
        """Optimized text chunking"""
        text = ' '.join(text.split())
        if not text:
            return []

        # Use sentence splitting for more natural chunks
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunk_id = f"{file_name}_p{page_number}_c{chunk_index}"
                    chunks.append({
                        "id": chunk_id,
                        "text": chunk_text,
                        "metadata": {
                            "file_name": file_name,
                            "page_number": page_number,
                            "chunk_index": chunk_index,
                            "confidence": confidence
                        }
                    })
                    chunk_index += 1
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length + 1

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_id = f"{file_name}_p{page_number}_c{chunk_index}"
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "file_name": file_name,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "confidence": confidence
                }
            })

        return chunks

    def process_files_batch(self, file_paths: List[str]) -> List[Dict]:
        """Process multiple files in parallel"""
        chunks = []
        with ThreadPoolExecutor(max_workers=min(8, len(file_paths))) as executor:
            future_to_file = {executor.submit(self.process_file, file_path): file_path 
                            for file_path in file_paths}
            
            for future in as_completed(future_to_file):
                file_chunks = future.result()
                if file_chunks:
                    chunks.extend(file_chunks)
        
        return chunks

    def process_file(self, file_path: str) -> List[Dict]:
        """Process individual file"""
        file_extension = Path(file_path).suffix.lower()
        chunks = []

        try:
            if file_extension in self.image_extensions:
                text, confidence = self.extract_text_from_image(file_path)
                if text.strip():
                    chunks.extend(self.chunk_text(text, file_path, 1, confidence))
            elif file_extension in self.document_extensions:
                pages = self.extract_text_from_pdf(file_path)
                for page_num, (page_text, confidence) in enumerate(pages, 1):
                    if page_text.strip():
                        chunks.extend(self.chunk_text(page_text, file_path, page_num, confidence))
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")

        return chunks

    def index_documents(self, folder_path: str) -> None:
        """Optimized document indexing"""
        logging.info(f"Starting optimized document indexing from {folder_path}")
        supported_extensions = self.image_extensions.union(self.document_extensions)
        files = [str(file) for ext in supported_extensions 
                for file in Path(folder_path).rglob(f'*{ext}')]
        
        # Process files in batches
        for i in range(0, len(files), self.batch_size):
            batch_files = files[i:i + self.batch_size]
            unprocessed_files = [f for f in batch_files if f not in self.processed_files]
            
            if not unprocessed_files:
                continue
                
            chunks = self.process_files_batch(unprocessed_files)
            
            if chunks:
                # Batch add to ChromaDB
                self.collection.add(
                    ids=[chunk["id"] for chunk in chunks],
                    documents=[chunk["text"] for chunk in chunks],
                    metadatas=[chunk["metadata"] for chunk in chunks]
                )
                
                self.processed_files.update(unprocessed_files)
                logging.info(f"Indexed batch of {len(unprocessed_files)} files")

    def search(self, query: str, top_k: int = 5, exact_match_only: bool = False) -> List[Dict]:
        """Optimized search implementation"""
        if exact_match_only:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where_document={"$contains": query}
            )
        else:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )

        formatted_results = []
        if results['ids'][0]:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'file_name': results['metadatas'][0][i]['file_name'],
                    'page_number': results['metadatas'][0][i]['page_number'],
                    'relevance_score': float(results['distances'][0][i]) if 'distances' in results else 1.0,
                    'is_exact_match': exact_match_only,
                    'confidence': results['metadatas'][0][i]['confidence']
                })

        return formatted_results

def main():
    logging.basicConfig(level=logging.INFO)
    
    use_gpu = torch.cuda.is_available()
    print(f"Using GPU: {use_gpu}")
    
    indexer = OptimizedVectorDB(
        persist_directory="vector_db",
        batch_size=32,
        use_gpu=use_gpu
    )
    
    indexer.index_documents("MVF-Test-Files")

    print("\nIndexing complete. Ready to accept queries.")
    print("You can input one query at a time. Type 'quit' to exit.")

    while True:
        try:
            query = input("\nEnter your search query (or 'quit' to exit): ").strip()
            if query.lower() == 'quit':
                print("\nExiting the search utility. Goodbye!")
                break

            search_type = input("Search type (1: Exact only, 2: Both exact and semantic): ").strip()
            exact_match_only = search_type == "1"

            results = indexer.search(query, exact_match_only=exact_match_only)

            if not results:
                print("\nNo results found.")
                continue

            print("\nSearch Results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. File: {result['file_name']}")
                print(f"   Page: {result['page_number']}")
                print(f"   Text: {result['text'][:200]}...")
                print(f"   Match Type: {'Exact' if result.get('is_exact_match') else 'Semantic'}")
                print(f"   Relevance: {result['relevance_score']:.3f}")
                print(f"   OCR Confidence: {result['confidence']:.1f}%")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting the search utility. Goodbye!")
            break

if __name__ == "__main__":
    main()