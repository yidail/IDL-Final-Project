# IDL-Final-Project
11785 F20 Final Project for Team 34  
Question answering is a popular research topic in natural language processing, and a widely used application in real world scenarios nowadays.
In this paper we propose a method based on BERT, which is a recurrent neural network composed of an encoder-decoder architecture, along with multihead self-attention mechanism, which is known as the "Transformer".
Several modifications will be added to the initial proposal in order to improve the question answering performance on the SQuAD dataset. All implementations are based on PyTorch.  
[[Video]](https://youtu.be/mBTJcJhdrjs)
## Branches
 - Main branch runs the LockedDropout and BLSTM version of the model
 - cnn branch runs the CNN bersion of the model
 - BERT-PBLSTM runs the PBLSTM version of the model

## Requirements
 - pip install transformers
 - pip install datasets

 ## Run
 - Use the notebook file to execute run_qa.py with our parameters  
 (the development was on Colab so the notebook is a little bit messy)

## Acknowledgements
 Thanks for huggingface's script to run SQUAD training.