## Bidirectional CPC Model for Speech Representations 

The bidirectional CPC model proposed in the paper:  
["On Scaling Contrastive Representations for Low-Resource Speech Recognition"](https://arxiv.org/abs/2102.00850).

See `example.py` for a simple example of how to load the model and extract the representation used in the paper.

File format used for training is 16-bit PCM sampled at 16kHz. See `test.wav`.

To test, run:
```
conda create -n scaling python==3.6.8
conda activate scaling
pip install -r requirements.txt
python extract.py
```

Tested with `CUDA 9.0`.
