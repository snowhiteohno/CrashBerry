"""Root-level launcher for the Gradio demo.
Run from the project root: `python run_demo.py`
Because this file lives at the project root, Python automatically
finds the `env`, `agent`, and `tools` packages without any sys.path
manipulation.
"""
from demo.app import demo

if __name__ == "__main__":
    demo.launch(share=True)
