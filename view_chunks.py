# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 19:02:52 2026

@author: Pedro
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
META_PATH = BASE_DIR / "index" / "internal_meta.jsonl"

def load_all_chunks():
    """Load all chunks into memory"""
    chunks = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def main():
    chunks = load_all_chunks()
    print(f"Loaded {len(chunks)} chunks from metadata\n")
    
    while True:
        print("\n" + "="*60)
        print("CHUNK VIEWER")
        print("="*60)
        print("Options:")
        print("  1. View chunk by filename and number (e.g., filename1.htm 83)")
        print("  2. Search by ticker (e.g., ITUB4)")
        print("  3. Search by filename (e.g., d213207df1.htm)")
        print("  4. View chunk by index (0 to {})".format(len(chunks)-1))
        print("  quit - Exit")
        print("="*60)
        
        choice = input("\nYour choice> ").strip().lower()
        
        if choice in {"quit", "exit", "q"}:
            break
        
        if choice == "1":
            filename = input("Filename: ").strip()
            chunk_num = int(input("Chunk number: ").strip())
            
            found = False
            for c in chunks:
                if Path(c["source"]).name == filename and c["chunk"] == chunk_num:
                    display_chunk(c)
                    found = True
                    break
            
            if not found:
                print(f"\nChunk not found: {filename} chunk {chunk_num}")
        
        elif choice == "2":
            ticker = input("Ticker: ").strip().upper()
            matches = [c for c in chunks if c["ticker"] == ticker]
            print(f"\nFound {len(matches)} chunks for {ticker}")
            
            if matches:
                for i, c in enumerate(matches[:10]):  # Show first 10
                    print(f"{i+1}. {Path(c['source']).name} chunk {c['chunk']}")
                
                if len(matches) > 10:
                    print(f"... and {len(matches)-10} more")
                
                idx = input("\nEnter number to view (or press Enter to skip): ").strip()
                if idx.isdigit():
                    display_chunk(matches[int(idx)-1])
        
        elif choice == "3":
            filename = input("Filename: ").strip()
            matches = [c for c in chunks if Path(c["source"]).name == filename]
            print(f"\nFound {len(matches)} chunks from {filename}")
            
            for i, c in enumerate(matches):
                preview = c["text"][:60].replace("\n", " ")
                print(f"{i+1}. Chunk {c['chunk']:3d} - {preview}...")
            
            idx = input("\nEnter number to view (or press Enter to skip): ").strip()
            if idx.isdigit():
                display_chunk(matches[int(idx)-1])
        
        elif choice == "4":
            idx = int(input("Chunk index: ").strip())
            if 0 <= idx < len(chunks):
                display_chunk(chunks[idx])
            else:
                print(f"\nInvalid index. Must be 0-{len(chunks)-1}")

def display_chunk(chunk):
    """Pretty print a chunk"""
    print("\n" + "="*60)
    print(f"Ticker: {chunk['ticker']}")
    print(f"Source: {chunk['source']}")
    print(f"Type: {chunk['type']}")
    print(f"Chunk: {chunk['chunk']}")
    print("="*60)
    print(chunk["text"])
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
