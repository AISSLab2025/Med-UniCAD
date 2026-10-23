# Source provenance

The release was reorganized from the finalized local training, architecture,
and inference files. Their SHA-256 hashes before the repository copy was made
are recorded below so the source snapshot can be audited independently.

| Source file | SHA-256 |
|---|---|
| `MedCLIP_Multi_Task_Training_Script_Unet_27_dense_freqxpert_moe_updated.py` | `3BFA9EA2004E370EAA7ACA25D1B8ED75F5653B808C0A8C5F6595BEE5B332407E` |
| `SwinTransformer27_Unified_MoE_Ablations.py` | `F3986FF86C9D75643473B02D1B27D850F0D95A4CE0E49602248F84949FB42D57` |
| `inference_config.example.json` | `C0827703C1CD2C830C245AD3053DB5559E155F6BD72F6A067E2A33A20824049E` |
| `unified_medUniCAD_api.py` | `539FEA63980961EF9C21FDAD229BA69C93571D773878D67523365421B9A1A22E` |
| `unified_medUniCAD_API_usage.md` | `9B060F50EFC868B44154FCCB824CFC4BCF15D451385A3D4E964E350A8993706C` |
| `unified_medUniCAD_inference.py` | `410A50856A26E7E07737FA59B61203179890511B51F227F12DAE68BE8057A8CB` |

Release changes preserve checkpoint-compatible model parameter names while
making the manuscript recipe authoritative: package imports and paths were
externalized; the final MoFX-Dense constructor was fixed; optimizer values now
come from the validated JSON; encoder and LoRA optimization starts at epoch 0;
report subsampling and non-paper metrics were removed; image/mask geometry was
synchronized; and non-final architecture, optimizer, pilot, and diagnostic
branches were removed. The original
source files listed above were not modified.
