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
 - Python3
 - `pip install torch`
 - `pip install transformers`
 - `pip install datasets`

 ## Run
  - For a custom model, define it in `models.py`
  - Import the model you want to run to `run_qa.py` by adding code `from models import YourModel` and replace `model` in `main()` with `model = YourModel.from_pretrained(model_args.model_name_or_path, from_tf=flase, config=config, cache_dir=model_args.cache_dir)`
 - Use the notebook file to execute run_qa.py with our parameters  
 (the development was on Colab so the notebook is a little bit messy)

## Acknowledgements
 Thanks for [huggingface's scripts](https://github.com/huggingface/transformers/tree/master/examples/question-answering) to run SQUAD training.
 
