# Decoding the AI That Solved Biology's 50-Year Grand Challenge

## An Open-Source Deep Dive into AlphaFold, AlphaFold3, and Boltz — Algorithm by Algorithm

---

*How we built the most comprehensive educational resource for understanding protein structure prediction AI — with 85+ interactive notebooks covering every algorithm.*

---

When DeepMind's AlphaFold solved the protein folding problem in 2020, it didn't just win CASP — it fundamentally changed computational biology. Yet for most researchers and engineers, the elegant 600-page supplementary material remained a black box.

**Until now.**

We've created **AlphaFold Codec** — an open-source repository that systematically deconstructs every algorithm from AlphaFold2, AlphaFold3, and the Boltz family. Not just explanations. Not just pseudocode. But **85+ executable Jupyter notebooks** with NumPy implementations you can run, modify, and learn from.

---

## The Problem with Understanding AlphaFold

Let's be honest: reading the AlphaFold papers is humbling. You encounter concepts like:

- **Evoformer** with its intricate MSA and pair representations
- **Invariant Point Attention (IPA)** operating in SE(3) equivariant space
- **Triangle Attention** — wait, triangles attending to what exactly?
- And now in AF3: **Diffusion Transformers** predicting atom coordinates through iterative denoising

The original code is written for production in JAX/Haiku, optimized for TPUs. Reading it to understand the *concepts* is like learning to cook by studying an industrial kitchen's automation system.

We asked: **What if you could see each algorithm isolated, implemented in pure NumPy, with test cases proving it works?**

---

## What We Built

### 🧬 AlphaFold2: All 32 Algorithms, Decoded

Every algorithm from the supplementary material, implemented and tested:

| Category | What You'll Learn |
|----------|-------------------|
| **Evoformer Stack** | How MSA and pair representations communicate through attention |
| **Triangle Operations** | The geometric intuition behind triangle multiplication and attention |
| **Invariant Point Attention** | SE(3) equivariance without the mathematical intimidation |
| **FAPE Loss** | Why frame-aligned point error is the secret sauce |
| **Recycling** | How iterative refinement bootstraps predictions |

Each notebook follows the same structure:
1. **Algorithm pseudocode** (directly from the paper)
2. **Source code location** in the official repository
3. **NumPy implementation** you can read in 5 minutes
4. **Working test cases** with expected outputs

---

### 🔮 AlphaFold3: The Diffusion Revolution

AF3 represents a fundamental architectural shift. Gone is the structure module with IPA. In its place: a **diffusion-based approach** that generates atom coordinates through iterative denoising.

We cover **23 key algorithms** including:

```
┌─────────────────────────────────────────────────────────────┐
│                    AlphaFold3 Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│  MSA Features → Pairformer (48 blocks) → Diffusion Module   │
│                                              ↓               │
│                                    200 denoising steps       │
│                                              ↓               │
│                              Confidence Head (pLDDT, pAE)    │
└─────────────────────────────────────────────────────────────┘
```

**Key AF3 notebooks you won't find elsewhere:**
- **Adaptive LayerNorm**: How noise level conditions the entire network
- **Atom Cross Attention**: Processing atom-level features
- **Diffusion Loss**: The training objective for structure generation

---

### ⚡ Boltz: Open-Source AF3 Alternative

Boltz is the first fully open-source model approaching AlphaFold3 accuracy. We've documented both versions:

**Boltz-1** (20 notebooks):
- Complete pipeline from input to confidence
- Pairformer, diffusion transformer, confidence module
- MIT licensed, production ready

**Boltz-2** (10 notebooks) — the game changer:
- **Binding affinity prediction** approaching FEP accuracy
- **1000x faster** than traditional free energy perturbation
- Contact conditioning for guided predictions
- First DL model predicting IC50 values accurately

```yaml
# Boltz-2: Predict binding affinity in seconds
sequences:
  - protein:
      id: A
      sequence: MVLSPADKTN...
  - ligand:
      id: B  
      smiles: CC(=O)NC1=CC=C(O)C=C1
affinity:
  predict: true
```

---

## Who This Is For

### Researchers
Implementing a new attention mechanism? See exactly how Triangle Attention is structured. Designing a new confidence metric? Understand how pLDDT is computed, mathematically and programmatically.

### Students
Learning geometric deep learning? Our notebooks provide gentle introductions to SE(3) equivariance, quaternion operations, and frame-based representations — without drowning in production code complexity.

### Engineers
Building on top of structure prediction? The NumPy implementations serve as clear specifications. Port to PyTorch, adapt for your use case, or simply understand the API contracts.

### The Curious
Want to understand the Nobel Prize-winning AI? Start with our inference pipeline notebook and work backwards.

---

## A Taste: Understanding Triangle Multiplication

Here's why our notebooks are different. Triangle multiplication is notoriously confusing. Let's demystify it:

**The intuition**: In a protein, residue i's relationship to residue j should be informed by their mutual relationships with every other residue k. It's like asking: "If A knows B, and A knows C, and B knows C, what does that tell us about A-B?"

**The implementation** (from our notebooks):

```python
def triangle_multiplication_outgoing(z, c=32):
    """
    z: pair representation [N, N, c_z]
    
    For each (i,j), aggregate information from edges (i,k) and (j,k)
    """
    # Project to gates and values
    a = linear(z, c)  # [N, N, c] — left projection  
    b = linear(z, c)  # [N, N, c] — right projection
    
    # The key operation: combine edges sharing an endpoint
    # a[i,k] * b[j,k] summed over k
    out = np.einsum('ikc,jkc->ijc', sigmoid(gate_a) * a, 
                                     sigmoid(gate_b) * b)
    
    return layer_norm(z + linear(out, c_z))
```

The `einsum` is doing the heavy lifting: for every pair (i,j), it looks at all triangles (i,j,k) and aggregates the information. **That's it.** No magic, just elegant tensor operations.

---

## The Numbers

| Component | Notebooks | Status |
|-----------|-----------|--------|
| AlphaFold2 Algorithms | 32 | ✅ Complete |
| AlphaFold3 Algorithms | 23 | ✅ Complete |
| Boltz-1 Algorithms | 20 | ✅ Complete |
| Boltz-2 Algorithms | 10 | ✅ Complete |
| **Total** | **85+** | **Ready to explore** |

Plus:
- Complete AF2 source code with annotations
- 128+ reference paper summaries
- Architecture diagrams
- Application examples (peptide docking, MD integration)

---

## Getting Started

```bash
git clone https://github.com/chenxingqiang/ref-Alphafold-Code.git
cd ref-Alphafold-Code

# Start with the AlphaFold2 index
jupyter notebook AF2-NoteBooks/ALGORITHM_INDEX.md

# Or jump straight to the iconic IPA
jupyter notebook AF2-NoteBooks/algorithm-22-InvariantPointAttention.ipynb
```

### Recommended Learning Path

1. **Start**: AlphaFold2 Algorithm 2 (Inference) — understand the full pipeline
2. **Core**: Algorithms 6-15 (Evoformer) — the representation learning heart
3. **Structure**: Algorithms 20-25 — from representations to 3D coordinates
4. **Compare**: AF3 Algorithm 15-17 (Diffusion) — see the paradigm shift
5. **Frontier**: Boltz-2 Algorithm 1-4 — binding affinity prediction

---

## Why We Built This

Protein structure prediction isn't just an academic exercise. It's enabling:

- **Drug discovery**: Understanding binding sites and designing inhibitors
- **Protein engineering**: Creating novel enzymes for sustainability
- **Disease research**: Modeling mutation effects and misfolding
- **Synthetic biology**: Designing proteins that don't exist in nature

But the barrier to entry has been too high. You shouldn't need to read 600 pages of supplementary material to understand how attention operates on MSA rows.

**We believe the best way to learn AI is to implement it.** And we've done the implementing so you can focus on understanding.

---

## What's Next

This is a living project. We're actively adding:

- **ESMFold integration**: Protein language model approaches
- **Chai-1 notebooks**: The next-gen competitor
- **Training tutorials**: How these models actually learn
- **Application notebooks**: From prediction to biological insight

---

## Join Us

The repository is fully open source. We welcome:

- **Bug reports**: Found an issue in our implementations?
- **Documentation**: Clearer explanations for complex concepts
- **New algorithms**: Coverage of emerging methods
- **Translations**: Making this accessible globally

**Repository**: [github.com/chenxingqiang/ref-Alphafold-Code](https://github.com/chenxingqiang/ref-Alphafold-Code)

---

## Final Thoughts

When John Jumper and Demis Hassabis won the Nobel Prize in Chemistry, they were recognized for solving a problem that stumped biologists for 50 years. But the real impact of AlphaFold isn't the prize — it's the 200+ million protein structures now available to every researcher on Earth.

Understanding *how* it works shouldn't be reserved for a select few.

**We've opened the black box. Come look inside.**

---

*If this helped you understand protein structure prediction, give us a ⭐ on GitHub. It helps others find this resource.*

*Have questions or want to contribute? Open an issue or reach out. We're building this together.*

---

### Tags
`#MachineLearning` `#DeepLearning` `#ComputationalBiology` `#AlphaFold` `#ProteinFolding` `#OpenSource` `#AI` `#Bioinformatics` `#DrugDiscovery`

---

*Author: Xingqiang Chen and Contributors*  
*Last updated: January 2026*
