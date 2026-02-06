import os
import argparse
from pathlib import Path

# Config: Add any other folders you want to skip
EXCLUDE_DIRS = {
    '__pycache__', '.git', '.idea', 'venv', 'env', 
    'runs', 'data', 'logs', 'node_modules', 'dist', 'build', '.vscode'
}
# Config: Add extensions you want to capture
# INCLUDE_EXTS = {'.py', '.yaml', '.yml', '.json', '.md', '.txt', '.toml', '.ini'}
INCLUDE_EXTS = {'.py', '.yaml', '.yml', '.json', '.txt', '.toml', '.ini'}

def pack_project(root_dir="."):
    root_path = Path(root_dir).resolve()
    
    if not root_path.exists():
        print(f"❌ Error: Directory '{root_path}' does not exist.")
        return

    # --- DYNAMIC FILENAME LOGIC ---
    # Gets the parent directory name, replaces spaces with underscores
    parent_name = root_path.name.replace(" ", "_")
    output_filename = f"{parent_name}.codebase.md"
    
    # Generate the content in the target directory
    output_path = root_path / output_filename
    # ------------------------------

    print(f"📦 Packing project from: {root_path}")
    print(f"📝 Output file: {output_path}")

    try:
        with open(output_path, "w", encoding="utf-8") as outfile:
            outfile.write(f"# Project Context: {root_path.name}\n")
            outfile.write(f"# Path: {root_path}\n")
            outfile.write("# Generated for AI Review\n\n")
            
            file_count = 0
            
            for root, dirs, files in os.walk(root_path):
                # Skip excluded folders AND anything ending in .egg-info
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.endswith('.egg-info')]
                
                for file in files:
                    file_path = Path(root) / file
                    
                    # Filter by extension
                    if file_path.suffix not in INCLUDE_EXTS:
                        continue
                    
                    # Skip the output file itself and common packer script names
                    if file_path == output_path or file in ["pack_dir.py", "pack_code.py", "package-lock.json"]:
                        continue

                    try:
                        content = file_path.read_text(encoding="utf-8")
                        rel_path = file_path.relative_to(root_path)
                        
                        outfile.write(f"\n{'='*50}\n")
                        outfile.write(f"FILE: {rel_path}\n")
                        outfile.write(f"{'='*50}\n\n")
                        outfile.write(content)
                        outfile.write("\n")
                        
                        print(f"  + Added: {rel_path}")
                        file_count += 1
                    except Exception as e:
                        print(f"  ! Skipped {file_path.name}: {e}")

        print(f"\n✅ Done! Packed {file_count} files into '{output_filename}'.")

    except Exception as e:
        print(f"❌ Critical Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pack a codebase into a named markdown file.")
    parser.add_argument(
        "path", 
        nargs="?", 
        default=".", 
        help="The path to the project directory (default: current directory)"
    )
    
    args = parser.parse_args()
    pack_project(root_dir=args.path)