"""Quick: list valid gemma-scope-2b-pt-res SAE IDs for layer 12."""
import modal

app = modal.App("e12-list-saes")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "sae_lens==6.39.0", "torch==2.6.0", "transformers==4.56.2", "numpy<2",
)


@app.function(image=image, timeout=300)
def ls():
    from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
    d = get_pretrained_saes_directory()
    for release_name in sorted(d.keys()):
        if "gemma-scope-2b-pt-res" in release_name:
            info = d[release_name]
            print(f"=== {release_name} ===")
            if hasattr(info, "saes_map"):
                for k in sorted(info.saes_map.keys()):
                    if "layer_12" in k and "16k" in k:
                        print(f"  {k}")


@app.local_entrypoint()
def main():
    ls.remote()
