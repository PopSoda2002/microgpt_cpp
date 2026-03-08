# Micro GPT

AI 编程时代的编码练习项目：一个微型的字符级语言模型，包含 Python 和 C++ 实现。

## 项目结构

- `microgpt.py` - Python 实现（带自动微分）
- `microgpt.cpp` - C++ 实现（含 Autograd Value 和 Transformer 前向传播）
- `input.txt` - 训练数据（首次运行 Python 版本会自动下载）

## 运行方式

### Python
```bash
python microgpt.py
```

### C++
```bash
g++ -std=c++17 -O2 -o gpt microgpt.cpp
./gpt
```

## 依赖

- Python: 无外部依赖，仅标准库
- C++: 需支持 C++17 的编译器（g++/clang++）
