# TinyGPT: A Minimalist Implementation of a Transformer-based Language Model

TinyGPT is a small-scale implementation of a transformer-based language model, built from scratch using PyTorch. This educational project demonstrates the core components of modern language models in a concise and understandable way.

## 🚀 Features

- **Pure PyTorch Implementation**: Built from the ground up with minimal dependencies
- **Educational Focus**: Clean, well-commented code for learning purposes
- **Modular Design**: Easy to understand and extend
- **Small Footprint**: Designed to run on consumer hardware

## 🏗️ Architecture

TinyGPT follows a simplified transformer decoder-only architecture:

```
TinyGPT(
 ├─ TokenEmbedding
 ├─ PositionalEmbedding
 ├─ DecoderBlock × N
 │   ├─ LayerNorm
 │   ├─ MultiHeadSelfAttention
 │   ├─ Linear Projection
 │   └─ FeedForward Network (2 layers, GELU)
 └─ Output Projection
)
```

## 📦 Dependencies

- Python 3.6+
- PyTorch
- NumPy

## 📚 File Structure

- `llm_scratch/`
  - `modules_llm.py`: Core transformer components and utilities
  - `final_model.py`: Main model implementation and training loop
  - `my_first_model.py`: Initial implementation (for reference)
  - `test.py`: Testing and evaluation scripts
  - `architecture.txt`: Model architecture overview

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the original [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper
- Built for educational purposes

