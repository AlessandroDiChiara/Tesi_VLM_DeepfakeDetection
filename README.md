## Posizione del dataset

Il dataset RRDataset è sul server nel seguente percorso:


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
    --model_size 7B \
    --prompt unstructured
```
