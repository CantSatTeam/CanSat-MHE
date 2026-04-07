#!/usr/bin/env python3
"""
Script to move files listed in a text file from source to destination directory.
"""

import argparse
import os
import shutil
from pathlib import Path


def move_files_from_list(list_file, source_dir, dest_dir):
    """
    Move files listed in a text file from source to destination directory.
    
    Args:
        list_file: Path to text file containing file names (one per line)
        source_dir: Source directory containing the files
        dest_dir: Destination directory to move files to
    """
    # Ensure source and destination are Path objects
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    # Verify source directory exists
    if not source_path.exists():
        print(f"Error: Source directory does not exist: {source_path}")
        return
    
    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    print(f"Destination directory: {dest_path}")
    
    # Read file names from list
    with open(list_file, 'r') as f:
        file_names = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(file_names)} files in list")
    
    # Move files
    moved_count = 0
    not_found_count = 0
    error_count = 0
    
    for file_name in file_names:
        source_file = source_path / file_name
        dest_file = dest_path / file_name
        
        if not source_file.exists():
            print(f"Warning: File not found: {source_file}")
            not_found_count += 1
            continue
        
        try:
            # Check if destination file already exists
            if dest_file.exists():
                print(f"Warning: Destination file already exists, overwriting: {dest_file}")
            
            shutil.move(str(source_file), str(dest_file))
            moved_count += 1
            print(f"Moved: {file_name}")
            
        except Exception as e:
            print(f"Error moving {file_name}: {e}")
            error_count += 1
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  Successfully moved: {moved_count}")
    print(f"  Not found: {not_found_count}")
    print(f"  Errors: {error_count}")
    print(f"{'='*50}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Move files listed in a text file from source to destination directory'
    )
    parser.add_argument('list_file', type=str,
                        help='Path to text file containing file names (one per line)')
    parser.add_argument('source_dir', type=str,
                        help='Source directory containing the files')
    parser.add_argument('dest_dir', type=str,
                        help='Destination directory to move files to')
    
    args = parser.parse_args()
    
    move_files_from_list(args.list_file, args.source_dir, args.dest_dir)
