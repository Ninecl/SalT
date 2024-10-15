# README
The source code of "Self-adaptive Relational Transformer for IT Knowledge Graph Completion"

# Run Example

Take the dataset *FB-IT* as an example, you can run the following code to train the model:

```python
python train.py -d FB -e SaRT --num_epochs 10000
```

Then, to eavluate the model on Inductive KGC task, run:

```python
python test_rank.py -d FB -e SaRT -m ind
```

To evaluate the model on *IT* KGC task, run:

```python
python test_rank.py -d FB -e SaRT -m IT
```

