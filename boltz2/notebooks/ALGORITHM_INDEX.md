# Boltz-2 Algorithm Index

This index covers the **new and enhanced algorithms in Boltz-2** compared to Boltz-1.

## Key Innovations in Boltz-2

| Feature | Description |
|---------|-------------|
| **Binding Affinity Prediction** | First DL model approaching FEP accuracy, 1000x faster |
| **Dual Output** | `affinity_pred_value` (log10 IC50) + `affinity_probability_binary` |
| **Contact Conditioning** | Guide predictions with experimental contacts |
| **Template v2** | Enhanced template processing |
| **Improved Confidence** | Better uncertainty estimation |

## Architecture Comparison

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Boltz-2 vs Boltz-1                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Boltz-1:  Input → MSA → Pairformer → Diffusion → Structure         │
│                                                    ↓                 │
│                                               Confidence             │
│                                                                      │
│  Boltz-2:  Input → MSA → Pairformer → Diffusion → Structure         │
│              ↓                                     ↓                 │
│         Templates v2                          Confidence v2          │
│         Contact Cond.                              ↓                 │
│                                             ┌──────────────┐         │
│                                             │ Affinity     │ ← NEW!  │
│                                             │ Module       │         │
│                                             └──────────────┘         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Algorithms

### New in Boltz-2

| # | Algorithm | Notebook | Source File | Status |
|---|-----------|----------|-------------|--------|
| 1 | Affinity Module | [algorithm-01-AffinityModule.ipynb](algorithm-01-AffinityModule.ipynb) | `model/modules/affinity.py` | ✅ |
| 2 | Gaussian Smearing | [algorithm-02-GaussianSmearing.ipynb](algorithm-02-GaussianSmearing.ipynb) | `model/modules/affinity.py` | ✅ |
| 3 | Contact Conditioning | [algorithm-03-ContactConditioning.ipynb](algorithm-03-ContactConditioning.ipynb) | `model/modules/trunkv2.py` | ✅ |
| 4 | Affinity Heads Transformer | [algorithm-04-AffinityHeadsTransformer.ipynb](algorithm-04-AffinityHeadsTransformer.ipynb) | `model/modules/affinity.py` | ✅ |

### Enhanced in Boltz-2 (v2 modules)

| # | Algorithm | Notebook | Source File | Status |
|---|-----------|----------|-------------|--------|
| 5 | Input Embedder v2 | [algorithm-05-InputEmbedderV2.ipynb](algorithm-05-InputEmbedderV2.ipynb) | `model/modules/trunkv2.py` | ✅ |
| 6 | Template Module v2 | [algorithm-06-TemplateModuleV2.ipynb](algorithm-06-TemplateModuleV2.ipynb) | `model/modules/trunkv2.py` | ✅ |
| 7 | Diffusion v2 | [algorithm-07-DiffusionV2.ipynb](algorithm-07-DiffusionV2.ipynb) | `model/modules/diffusionv2.py` | ✅ |
| 8 | Confidence v2 | [algorithm-08-ConfidenceV2.ipynb](algorithm-08-ConfidenceV2.ipynb) | `model/modules/confidencev2.py` | ✅ |
| 9 | Distogram v2 | [algorithm-09-DistogramV2.ipynb](algorithm-09-DistogramV2.ipynb) | `model/loss/distogramv2.py` | ✅ |
| 10 | B-Factor Prediction | [algorithm-10-BFactorPrediction.ipynb](algorithm-10-BFactorPrediction.ipynb) | `model/modules/trunkv2.py` | ✅ |

## Source Code References

- **Official Repository**: [jwohlwend/boltz](https://github.com/jwohlwend/boltz) (contains both Boltz-1 and Boltz-2)
- **Boltzina (Virtual Screening)**: [ohuelab/boltzina](https://github.com/ohuelab/boltzina)
- **Boltz-2 Paper**: [bioRxiv 2025.06.14.659707](https://doi.org/10.1101/2025.06.14.659707)

## Usage

Boltz-2 is the default model when running:
```bash
boltz predict input.yaml --use_msa_server
```

For affinity prediction, use a YAML with affinity specifications:
```yaml
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
