# Decoding the AI That Solved Biology's 50-Year Grand Challenge

## An Open-Source Deep Dive into AlphaFold, AlphaFold3, and Boltz — Algorithm by Algorithm

---

*How we built the most comprehensive educational resource for understanding protein structure prediction AI — with 85+ interactive notebooks and a complete fine-tuning framework for real-world applications.*

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
- **NEW: Complete fine-tuning framework with 50+ task types**
- **NEW: Production-ready heads for drug discovery, antibody design, enzyme engineering, and more**

---

## 🔧 Fine-tuning: From Understanding to Application

Understanding algorithms is powerful. But what if you want to *adapt* these models for your specific research?

**We've built a complete fine-tuning framework** that lets you customize AlphaFold2, AlphaFold3, Boltz-1, and Boltz-2 for downstream tasks — without needing a cluster of TPUs.

### The Challenge of Fine-tuning Structure Prediction Models

These models have hundreds of millions of parameters. Fine-tuning them naively requires:
- Massive GPU memory (80GB+ A100s)
- Large datasets (tens of thousands of structures)
- Weeks of training time

Most researchers don't have these resources. So we implemented **parameter-efficient fine-tuning** techniques that make adaptation accessible.

### LoRA: Fine-tune with 0.1% of the Parameters

**LoRA (Low-Rank Adaptation)** decomposes weight updates into low-rank matrices. Instead of updating a 384×384 matrix (147,456 parameters), you update two small matrices: 384×8 and 8×384 (6,144 parameters).

```python
from finetuning import FineTuningConfig
from finetuning.modules import LoRAModule

# Apply LoRA to attention layers
lora_model = LoRAModule(
    model,
    rank=8,           # Low-rank dimension
    alpha=16.0,       # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# Only 0.1% of parameters are trainable!
trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in lora_model.parameters())
print(f"Trainable: {trainable/total:.4f}")  # ~0.001
```

**Result**: Fine-tune on a single 24GB GPU. Train in hours, not weeks.

### Supported Fine-tuning Tasks: 50+ Task Types

Inspired by production platforms like **ProteinBase.com**, we've built comprehensive support for real-world protein analysis applications:

#### 💊 Drug Discovery

| Task | Predictions | Use Cases |
|------|-------------|-----------|
| **Binding Affinity** | pKd, pIC50, ΔG, Ki | Lead optimization, SAR analysis |
| **Virtual Screening** | Hit probability, enrichment | High-throughput screening |
| **ADMET** | Absorption, metabolism, toxicity | Compound prioritization |

#### 🔬 Protein Engineering

| Task | Predictions | Use Cases |
|------|-------------|-----------|
| **Stability** | ΔΔG, Tm shift, aggregation | Thermostability optimization |
| **Solubility** | Expression scores, aggregation risk | Biomanufacturing |
| **Mutation Effects** | ΔΔG, fitness, pathogenicity | Variant analysis, mutagenesis |

#### 🧫 Antibody Design

| Task | Predictions | Use Cases |
|------|-------------|-----------|
| **Affinity Maturation** | CDR binding, mutant ranking | Therapeutic optimization |
| **Humanization** | Humanness scores, deimmunization | Drug development |
| **Developability** | Aggregation, viscosity, expression | Manufacturing readiness |

#### ⚗️ Enzyme Engineering

| Task | Predictions | Use Cases |
|------|-------------|-----------|
| **Activity** | kcat, Km, kcat/Km | Catalyst optimization |
| **Specificity** | Substrate profiles, selectivity | Industrial applications |
| **Directed Evolution** | Fitness landscapes, hot spots | Protein engineering |

#### 🔗 Protein-Protein Interactions

| Task | Predictions | Use Cases |
|------|-------------|-----------|
| **PPI Binding** | Kd, interface stability | Complex analysis |
| **Interface Prediction** | Contact residues, buried area | Structure analysis |
| **Hot Spot Detection** | ΔΔG per residue, druggability | PPI drug targets |

#### 🧬 Function Prediction

| Task | Predictions | Use Cases |
|------|-------------|-----------|
| **GO Terms** | Molecular function, biological process | Annotation |
| **EC Numbers** | Enzyme classification | Function discovery |
| **Localization** | Subcellular compartment | Systems biology |

#### 🛡️ Immunology

| Task | Predictions | Use Cases |
|------|-------------|-----------|
| **B-cell Epitopes** | Linear/conformational epitopes | Vaccine design |
| **T-cell Epitopes** | MHC-I/II binding, presentation | Immunotherapy |
| **Immunogenicity** | Therapeutic immunogenicity | Drug safety |

#### 📊 Structure Quality

| Task | Predictions | Use Cases |
|------|-------------|-----------|
| **Confidence Metrics** | pLDDT, pAE, pTM, lDDT | Model validation |
| **Disorder Prediction** | Intrinsically disordered regions | Structure analysis |
| **Contact/Distance** | Residue-residue contacts, distance maps | Structure validation |

```python
# Example: Quick access to any task configuration
from finetuning.configs import get_task_config, list_tasks_by_category

# See all available tasks
categories = list_tasks_by_category()
print(categories["antibody"])  # ['affinity_maturation', 'humanization', 'developability']

# Get optimized config for any task
config = get_task_config("enzyme_activity")
print(config.output_dim)  # 3 (kcat, Km, kcat_over_Km)
```

### Real-World Example: Drug Binding Affinity

Want to predict how strongly a drug candidate binds to its target protein? Here's how:

```python
from finetuning import FineTuningConfig, Trainer
from finetuning.heads import AffinityHead, AffinityHeadConfig
from finetuning.data import AffinityDataset

# 1. Configure for binding affinity prediction
config = FineTuningConfig(
    strategy="lora",
    task="binding_affinity",
    lora_rank=8,
    training=TrainingConfig(
        learning_rate=5e-5,
        max_steps=10000,
    ),
)

# 2. Add the affinity prediction head (Boltz-2 architecture)
head_config = AffinityHeadConfig(
    use_gaussian_smearing=True,  # Distance features
    use_attention_pooling=True,  # Aggregate over residues
)
affinity_head = AffinityHead(head_config)

# 3. Load your data (PDBbind, BindingDB, etc.)
train_data = AffinityDataset("./pdbbind/train", affinity_file="affinities.csv")

# 4. Train
trainer = Trainer(model, config, train_loader)
trainer.train()

# 5. Predict on new complexes
predictions = model.predict(protein_ligand_complex)
print(f"Predicted pIC50: {predictions['affinity_pred_value']:.2f}")
```

### Fine-tuning Strategies Compared

| Strategy | Trainable Params | Memory | Best For |
|----------|-----------------|--------|----------|
| **LoRA** | ~0.1% | Low | Small datasets (<1K samples) |
| **Adapter** | ~1% | Low | Multi-task learning |
| **Head-only** | ~5% | Medium | New prediction tasks |
| **Full** | 100% | High | Large datasets (>10K samples) |

### What Makes Our Framework Different

1. **Works with both PyTorch and JAX**: Boltz uses PyTorch, AlphaFold uses JAX. We support both.

2. **Production-ready training**: Gradient accumulation, mixed precision, distributed training, W&B logging.

3. **Task-specific heads**: Not just generic classifiers — architectures designed for structural biology (Gaussian smearing for distances, attention pooling for variable-length proteins).

4. **Educational implementations**: Every module has a NumPy reference implementation so you understand what's happening.

```python
# NumPy reference — see exactly what LoRA does
class LoRALinearNumPy:
    def forward(self, x):
        # Original: x @ W
        # LoRA: x @ W + x @ A @ B * scaling
        original = x @ self.W.T
        lora_contribution = x @ self.lora_A.T @ self.lora_B.T * self.scaling
        return original + lora_contribution
```

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
6. **Apply**: Fine-tuning framework — adapt models for your research

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

## Real-World Examples: Beyond Binding Affinity

### Example 2: Antibody Affinity Maturation

```python
from finetuning.heads import AntibodyAffinityHead, AntibodyHeadConfig
from finetuning.data import AntibodyDataset

# Configure for CDR-focused predictions
config = AntibodyHeadConfig(
    cdr_regions=["CDR-H3", "CDR-L3"],  # Focus on key binding regions
    use_paratope_attention=True,
    predict_developability=True,  # Also predict manufacturability
)

head = AntibodyAffinityHead(config)
dataset = AntibodyDataset("./sabdab", heavy_chain_col="VH", light_chain_col="VL")

# Train and predict affinity changes for CDR mutations
predictions = model.predict(antibody_antigen_complex)
print(f"Predicted ΔΔG: {predictions['ddg']:.2f} kcal/mol")
print(f"Developability score: {predictions['developability']:.2f}")
```

### Example 3: Enzyme Activity Prediction

```python
from finetuning.heads import EnzymeActivityHead, EnzymeHeadConfig
from finetuning.data import EnzymeDataset

# Predict full kinetic parameters
config = EnzymeHeadConfig(
    output_dim=3,  # kcat, Km, kcat/Km
    active_site_radius=8.0,  # Angstroms
    use_substrate_features=True,
)

head = EnzymeActivityHead(config)
dataset = EnzymeDataset("./brenda", activity_columns=["kcat", "Km"])

# Predict for enzyme-substrate pair
predictions = model.predict(enzyme_substrate_complex)
print(f"Predicted kcat: {10**predictions['kcat']:.1f} s⁻¹")
print(f"Predicted Km: {10**predictions['Km']:.1f} μM")
```

### Example 4: B-cell Epitope Prediction

```python
from finetuning.heads import BcellEpitopeHead, EpitopeHeadConfig
from finetuning.data import EpitopeDataset

# Predict conformational epitopes
config = EpitopeHeadConfig(
    epitope_type="conformational",
    surface_threshold=25.0,  # SASA threshold
    spatial_window=10.0,  # Angstroms
)

head = BcellEpitopeHead(config)
# Per-residue epitope probability
predictions = model.predict(antigen_structure)
print(f"Top epitope residues: {predictions['epitope_residues']}")
```

---

## What's Next

This is a living project. We're actively adding:

- **ESMFold integration**: Protein language model approaches
- **Chai-1 notebooks**: The next-gen competitor
- **Pre-trained LoRA weights**: Domain-specific adapters (antibodies, enzymes, GPCRs)
- **Colab notebooks**: Run fine-tuning in the cloud for free
- **Benchmark datasets**: Curated datasets for each task type
- **Model zoo**: Pre-fine-tuned models for common applications

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
`#MachineLearning` `#DeepLearning` `#ComputationalBiology` `#AlphaFold` `#ProteinFolding` `#OpenSource` `#AI` `#Bioinformatics` `#DrugDiscovery` `#FineTuning` `#LoRA` `#BindingAffinity` `#AntibodyDesign` `#EnzymeEngineering` `#ProteinEngineering` `#VaccineDesign` `#PPI`

---

*Author: Xingqiang Chen and Contributors*  
*Last updated: January 2026*
