# Method

## Teacher

- TF-IDF features (5,000 terms) on 20 newsgroups text  
- Multiclass **logistic regression**  

## Student (distillation)

- Compact **64-dimensional** text vectors via TF-IDF → TruncatedSVD → L2 normalize  
- **Ridge regression** trained to match teacher **class probabilities** on the training split  
- At inference, student picks `argmax` of its probability vector  

## Evaluation

Hold-out **20%** stratified split. Report **accuracy** and **macro-F1** for teacher and student on the same test labels.

This is classic **response distillation** on tabular features—not neural KD—but the workflow mirrors production distill pipelines: train teacher → soft targets → smaller student → compare metrics.
