# Neural Networks: Playground

A course that guides you through building neural networks from the ground up. Each lecture introduces progressively more complex models, starting from basic autograd engines to transformer-based architectures. Below is an overview of the models developed throughout the course.

---

## Lecture Overview

### **Lecture 1: Introduction to Neural Networks and Backpropagation**
- **Model**: A minimal autograd engine (`micrograd`) implementing automatic differentiation.
- **Details**: Built from scratch in Python, this engine computes gradients for scalar-valued functions, forming the foundation for training neural networks via backpropagation. Demonstrates the core mechanics of gradient computation in deep learning.

### **Lecture 2: Bigram Character-Level Language Model**
- **Model**: Statistical bigram model for character-level language modeling.
- **Details**: Predicts the next character in a sequence using frequency counts. Introduces the framework of language modeling, including training, sampling, and evaluation using negative log likelihood.

### **Lecture 3: Multilayer Perceptron (MLP) for Language Modeling**
- **Model**: A two-layer MLP with an embedding layer, hidden layer, and ReLU activation.
- **Details**: Extends the bigram model by learning distributed representations of characters. Covers key ML concepts like hyperparameter tuning, train/dev/test splits, and overfitting.

### **Lecture 4: MLP with Batch Normalization**
- **Model**: Enhanced MLP incorporating batch normalization.
- **Details**: Stabilizes training by normalizing layer activations. Diagnoses gradient/activation scaling issues and introduces tools to monitor network health during training.

### **Lecture 5: Manual Backpropagation in MLPs**
- **Model**: Step-by-step backpropagation through the MLP without autograd.
- **Details**: Manually computes gradients for tensors in the cross-entropy loss, linear layers, activation functions, and embeddings. Reinforces intuition for gradient flow in computational graphs.

### **Lecture 6: WaveNet-Style Convolutional Network**
- **Model**: Hierarchical convolutional neural network (CNN) inspired by WaveNet.
- **Details**: Processes character sequences with dilated convolutions to capture long-range dependencies. Introduces PyTorch's `nn.Module` and tensor shape management for deeper networks.

### **Lecture 7: Transformer-Based GPT Model**
- **Model**: Generative Pretrained Transformer (GPT) implementing self-attention.
- **Details**: Follows the "Attention is All You Need" architecture. Trains an autoregressive language model using multi-head attention, positional embeddings, and transformer blocks. Scales to modern LLM frameworks.

### **Lecture 8: Byte Pair Encoding (BPE) Tokenizer**
- **Model**: Subword tokenizer using BPE, as used in GPT models.
- **Details**: Implements encoding/decoding between text and tokens. Explores tokenization challenges (e.g., multilingual support, edge cases) and its impact on model behavior.

---

## Course Progression
The course begins with foundational concepts (autograd, MLPs) and incrementally introduces modern techniques (batch norm, CNNs, transformers). Each model addresses limitations of prior approaches, culminating in a full GPT implementation with a custom tokenizer.

---

## License
This project is licensed under the [MIT License](LICENSE).