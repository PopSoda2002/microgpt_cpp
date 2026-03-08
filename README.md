# Micro GPT

A miniature character-level language model for coding practice, with both Python and C++ implementations.

## Project Structure

- `microgpt.py` - Python implementation (with autograd)
- `microgpt.cpp` - C++ implementation (Autograd Value + Transformer forward pass)
- `input.txt` - Training data (auto-downloaded on first Python run)

## How to Run

### Python
```bash
python microgpt.py
```

### C++
```bash
g++ -std=c++17 -O2 -o gpt microgpt.cpp
./gpt
```

## Dependencies

- Python: No external deps, standard library only
- C++: Compiler with C++17 support (g++/clang++)
