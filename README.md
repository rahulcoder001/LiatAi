# Player Re-Identification System

##  Download Required Files

Before running, place the following files in the `data/` folder:

- [15sec_input_720p.mp4](<your-drive-link>)
- [best.pt](<your-drive-link>)

Folder structure:
data/
├── 15sec_input_720p.mp4
└── best.pt

## Setup
1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

## run 
python main.py