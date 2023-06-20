# Notes as I train.

## General Notes
- microsoft/deberta-v3-small is good for prototpying ideas. batch size 64 works
- will swith over to microsoft/deberta-v3-base for final training. batch size 16 works.
- microsoft/deberta-v3-large is crashing weirdly (The metrics suddenly go downwards and the loss upwards) -> Debug this issue
- anferico/bert-for-patents also works well. Problem is its only one size. batch size 64 works.

## Things to try
- cosine scheduler
- change prediction to 4 classes instead (classification instead of regression) - Did not work
- composer algorithms?
- pretraining a model on patent big query dataset before final task
- Replace the patent context field with the description of that context provided by the patent office

## Things to do
- setup wandb logging
-  

## Useful Links
- https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332243
- https://wandb.ai/darek/fbck/reports/How-To-Build-an-Efficient-NLP-Model--VmlldzoyNTE5MDEx#adversarial-weight-perturbation-(awp)
- https://www.kaggle.com/code/nbroad/token-classification-approach-fpe
- https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332707
- 