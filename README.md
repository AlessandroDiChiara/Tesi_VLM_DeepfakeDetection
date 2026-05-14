## Posizione del dataset

Il dataset RRDataset si trova sul server in:


/mnt/ssd1/teglia/rrdataset/RRDataset_final


## I modelli sono salvati in:


/mnt/ssd1/teglia/dichiara/models


## Esecuzione

```bash
podman run \
  -v /home/teglia/dichiara:/work/project/ \
  -v /mnt/ssd1/teglia/rrdataset/RRDataset_final:/work/dataset/ \
  -v /mnt/ssd1/teglia/dichiara/models:/work/models/ \
  --device nvidia.com/gpu=all \
  --ipc host \
  localhost/alessandro240/internvl:latest \
  /opt/conda/bin/python3 /work/project/test_Qwen2.5VL.py \
  --model_size <7B|32B|72B> \
  --prompt <structured|unstructured>
```
