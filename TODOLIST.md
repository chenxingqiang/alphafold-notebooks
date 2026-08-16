# Project TODO List

## ✅ Completed

### Algorithm Notebooks (85+ Complete)

| Model | Notebooks | Status |
|-------|-----------|--------|
| AlphaFold2 | 32/32 algorithms | ✅ Complete |
| AlphaFold3 | 23/23 algorithms | ✅ Complete |
| Boltz-1 | 20/20 algorithms | ✅ Complete |
| Boltz-2 | 10/10 algorithms | ✅ Complete |

### Reference Papers (229 Curated)

| Model | Papers | Status |
|-------|--------|--------|
| AlphaFold2 | 83 papers | ✅ [AF2REFPAPERS.md](alphafold2/AF2REFPAPERS.md) |
| AlphaFold3 | 50 papers | ✅ [AF3REFPAPERS.md](alphafold3/AF3REFPAPERS.md) |
| Boltz-1 | 46 papers | ✅ [BOLTZREFPAPERS.md](boltz/BOLTZREFPAPERS.md) |
| Boltz-2 | 50 papers | ✅ [BOLTZ2REFPAPERS.md](boltz2/BOLTZ2REFPAPERS.md) |

### Project Organization

- [x] Reorganize directory structure (lowercase, consistent naming)
- [x] Create comprehensive README.md
- [x] Update .gitmodules for new paths
- [x] Create BLOG_POST.md
- [x] Add finetuning framework

---

## 🔄 In Progress

### Fine-tuning Framework Enhancement

- [x] AlphaFold 3 weight download/validation tooling (`finetuning/af3/`)
- [x] AF3 parameter schema vendored (405 entries, metadata only)
- [x] AF3 Haiku-parameter-space LoRA + `AlphaFold3FineTuner`
- [ ] Add pre-trained LoRA weights for common tasks
- [ ] Create Colab notebooks for cloud training
- [ ] Build benchmark datasets for each task type
- [ ] Implement model zoo with pre-fine-tuned models

### Documentation

- [ ] Add Chinese documentation
- [ ] Create video tutorials
- [ ] Write detailed API documentation for finetuning module

---

## 📋 Planned

### New Model Support

- [ ] ESMFold notebooks (protein language model approach)
- [ ] Chai-1 notebooks (next-gen competitor)
- [ ] OpenFold notebooks (PyTorch AlphaFold2)
- [ ] RoseTTAFold notebooks

### Advanced Features

- [ ] Interactive 3D visualization in notebooks
- [ ] Streamlit/Gradio demo apps
- [ ] Docker containers for easy deployment
- [ ] CI/CD pipeline for notebook testing

### Fine-tuning Extensions

- [ ] Domain-specific adapters (antibodies, enzymes, GPCRs)
- [ ] Multi-task learning examples
- [ ] Active learning integration
- [ ] Uncertainty quantification

---

## 📊 AlphaFold2 Source Code Review

Location: `alphafold2/source/`

### common/
- [x] confidence.py
- [x] protein.py
- [x] protein_test.py
- [x] residue_constants.py
- [x] residue_constants_test.py

### data/
- [x] mmcif_parsing.py
- [x] parsers.py
- [x] pipeline.py
- [x] templates.py

### data/tools/
- [x] hhblits.py
- [x] hhsearch.py
- [x] hmmbuild.py
- [x] hmmsearch.py
- [x] jackhmmer.py
- [x] kalign.py
- [x] utils.py

### model/
- [x] all_atom.py
- [x] all_atom_test.py
- [x] common_modules.py
- [x] config.py
- [x] data.py
- [x] features.py
- [x] folding.py
- [x] layer_stack.py
- [x] layer_stack_test.py
- [x] lddt.py
- [x] lddt_test.py
- [x] mapping.py
- [x] model.py
- [x] modules.py
- [x] prng.py
- [x] prng_test.py
- [x] quat_affine.py
- [x] quat_affine_test.py
- [x] r3.py
- [x] utils.py

### model/tf/
- [x] data_transforms.py
- [x] input_pipeline.py
- [x] protein_features.py
- [x] protein_features_test.py
- [x] proteins_dataset.py
- [x] shape_helpers.py
- [x] shape_helpers_test.py
- [x] shape_placeholders.py
- [x] utils.py

### relax/
- [x] amber_minimize.py
- [x] amber_minimize_test.py
- [x] cleanup.py
- [x] cleanup_test.py
- [x] relax.py
- [x] relax_test.py
- [x] utils.py
- [x] utils_test.py

---

## ✅ AlphaFold2 Algorithm Notebooks (32/32 Complete)

| # | Algorithm | Notebook | Status |
|---|-----------|----------|--------|
| 1 | MSABlockDeletion | [algorithm-1](alphafold2/notebooks/algorithm-1-MSABlockDeletion.ipynb) | ✅ |
| 2 | Inference | [algorithm-2](alphafold2/notebooks/algorithm-2-Inference.ipynb) | ✅ |
| 3 | InputEmbedder | [algorithm-3](alphafold2/notebooks/algorithm-3-InputEmbedder.ipynb) | ✅ |
| 4 | relpos | [algorithm-4](alphafold2/notebooks/algorithm-4-relpos.ipynb) | ✅ |
| 5 | one_hot | [algorithm-5](alphafold2/notebooks/algorithm-5-one_hot.ipynb) | ✅ |
| 6 | EvoformerStack | [algorithm-6](alphafold2/notebooks/algorithm-6-EvoformerStack.ipynb) | ✅ |
| 7 | MSARowAttentionWithPairBias | [algorithm-7](alphafold2/notebooks/algorithm-7-MSARowAttentionWithPairBias.ipynb) | ✅ |
| 8 | MSAColumnAttention | [algorithm-8](alphafold2/notebooks/algorithm-8-MSAColumnAttention.ipynb) | ✅ |
| 9 | MSATransition | [algorithm-9](alphafold2/notebooks/algorithm-9-MSATransition.ipynb) | ✅ |
| 10 | OuterProductMean | [algorithm-10](alphafold2/notebooks/algorithm-10-OuterProductMean.ipynb) | ✅ |
| 11 | TriangleMultiplicationOutgoing | [algorithm-11](alphafold2/notebooks/algorithm-11-TriangleMultiplicationOutgoing.ipynb) | ✅ |
| 12 | TriangleMultiplicationIncoming | [algorithm-12](alphafold2/notebooks/algorithm-12-TriangleMultiplicationIncoming.ipynb) | ✅ |
| 13 | TriangleAttentionStartingNode | [algorithm-13](alphafold2/notebooks/algorithm-13-TriangleAttentionStartingNode.ipynb) | ✅ |
| 14 | TriangleAttentionEndingNode | [algorithm-14](alphafold2/notebooks/algorithm-14-TriangleAttentionEndingNode.ipynb) | ✅ |
| 15 | PairTransition | [algorithm-15](alphafold2/notebooks/algorithm-15-PairTransition.ipynb) | ✅ |
| 16 | TemplatePairStack | [algorithm-16](alphafold2/notebooks/algorithm-16-TemplatePairStack.ipynb) | ✅ |
| 17 | TemplatePointwiseAttention | [algorithm-17](alphafold2/notebooks/algorithm-17-TemplatePointwiseAttention.ipynb) | ✅ |
| 18 | ExtraMsaStack | [algorithm-18](alphafold2/notebooks/algorithm-18-ExtraMsaStack.ipynb) | ✅ |
| 19 | MSAColumnGlobalAttention | [algorithm-19](alphafold2/notebooks/algorithm-19-MSAColumnGlobalAttention.ipynb) | ✅ |
| 20 | StructureModule | [algorithm-20](alphafold2/notebooks/algorithm-20-StructureModule.ipynb) | ✅ |
| 21 | rigidFrom3Points | [algorithm-21](alphafold2/notebooks/algorithm-21-rigidFrom3Points.ipynb) | ✅ |
| 22 | InvariantPointAttention | [algorithm-22](alphafold2/notebooks/algorithm-22-InvariantPointAttention.ipynb) | ✅ |
| 23 | BackboneUpdate | [algorithm-23](alphafold2/notebooks/algorithm-23-BackboneUpdate.ipynb) | ✅ |
| 24 | computeAllAtomCoordinates | [algorithm-24](alphafold2/notebooks/algorithm-24-computeAllAtomCoordinates.ipynb) | ✅ |
| 25 | makeRotX | [algorithm-25](alphafold2/notebooks/algorithm-25-makeRotX.ipynb) | ✅ |
| 26 | renameSymmetricGroundTruthAtoms | [algorithm-26](alphafold2/notebooks/algorithm-26-renameSymmetricGroundTruthAtoms.ipynb) | ✅ |
| 27 | torsionAngleLoss | [algorithm-27](alphafold2/notebooks/algorithm-27-torsionAngleLoss.ipynb) | ✅ |
| 28 | computeFAPE | [algorithm-28](alphafold2/notebooks/algorithm-28-computeFAPE.ipynb) | ✅ |
| 29 | predictPerResidueLDDT | [algorithm-29](alphafold2/notebooks/algorithm-29-predictPerResidueLDDT.ipynb) | ✅ |
| 30 | RecyclingInference | [algorithm-30](alphafold2/notebooks/algorithm-30-RecyclingInference.ipynb) | ✅ |
| 31 | RecyclingTraining | [algorithm-31](alphafold2/notebooks/algorithm-31-RecyclingTraining.ipynb) | ✅ |
| 32 | RecyclingEmbedder | [algorithm-32](alphafold2/notebooks/algorithm-32-RecyclingEmbedder.ipynb) | ✅ |

---

## ✅ AlphaFold3 Algorithm Notebooks (23/23 Complete)

| # | Algorithm | Status |
|---|-----------|--------|
| 1-4 | Input Preparation (MSA, Template, Atom, RelPos) | ✅ |
| 5-7 | MSA Module (OPM, Attention, Transition) | ✅ |
| 8-14 | Pairformer (Triangle Ops, Single Attn) | ✅ |
| 15-19 | Diffusion (Module, AdaLN, Transformer, CrossAttn) | ✅ |
| 20-23 | Confidence & Loss (Distogram, Confidence, Loss, LDDT) | ✅ |

---

## ✅ Boltz Algorithm Notebooks (30/30 Complete)

### Boltz-1 (20 algorithms)
| # | Category | Status |
|---|----------|--------|
| 1-3 | Input Processing | ✅ |
| 4-6 | MSA Processing | ✅ |
| 7-11 | Pairformer Stack | ✅ |
| 12-15 | Diffusion Module | ✅ |
| 16-20 | Confidence & Loss | ✅ |

### Boltz-2 (10 new algorithms)
| # | Category | Status |
|---|----------|--------|
| 1-4 | Affinity Module (NEW) | ✅ |
| 5-10 | Enhanced v2 Modules | ✅ |

---

## 📈 Project Statistics

| Metric | Count |
|--------|-------|
| Total Notebooks | 85+ |
| Reference Papers | 229 |
| Fine-tuning Task Types | 50+ |
| Git Submodules | 14 |
| Source Code Files | 60+ |

---

*Last updated: August 2026*
