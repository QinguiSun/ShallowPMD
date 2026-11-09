# ShallowPMD

> **ShallowPMD: A flat, transparent, and pure PyTorch reimplementation of DeePMD-kit, built for effortless learning and hacking.**

---

## 🌟 Introduction

While Deepmodeling's **DeePMD-kit** has heavily revolutionized the field of machine learning molecular dynamics, its source code has grown remarkably complex over years of industrial iteration. To maintain cross-platform multi-framework ecosystems, the original codebase is layered with heavy abstractions supporting TensorFlow, PyTorch, JAX, and Paddle, alongside tight C++/CUDA dual-bindings for LAMMPS interfaces. 

This inherent complexity often creates a "black-box" barrier, hindering students and researchers from grasping the core physical principles and data flows.

**ShallowPMD** is born to address this. As the name implies, we flatten the code architecture from "Deep" to "Shallow". By stripping away heavy engineering armor and legacy framework shims, we return the machine learning potential model to its purest mathematical and physical essence.

## 🛠️ Core Simplifications & Features

*   **Flattened Class Hierarchy:** Completely abandons nested polymorphic abstractions and complex base routing classes. All structural components (Descriptors, Fitting Networks) directly inherit from native `torch.nn.Module`, making data propagation completely transparent.
*   **Pure PyTorch Ecosystem Binding:** Strips away all boilerplate code, runtime conditional checks, and legacy wrappers meant for TensorFlow, JAX, or Paddle. It offers an idiomatic PyTorch environment tailored for standard AI workflows.
*   **Comprehensive Line-by-Line Commentary:** Features thorough annotations dissecting the mathematical logic behind rotationally and permutationally invariant coordinate matrices (such as `torch.einsum` mappings) and smooth cut-off functions.
*   **Zero LAMMPS or C++ Compilation Dependency:** Severed from the traditional LAMMPS C++ plugin bindings. No more local compilation nightmares or configuration hell—just pure Python and PyTorch that you can run out-of-the-box.

Whether you want to dissect the raw underlying mathematics of Deep Potentials, or benchmark your own innovative architectural tweaks, **ShallowPMD** serves as your optimal white-box sandbox.

## 🚀 Getting Started

*(Stay tuned. We are currently utilizing REA (Reverse Engineer Anything) to systematically reconstruct and open-source the core physics engines here layer by layer!)*

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.
