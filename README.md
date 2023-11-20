# chatbot
## Answer 1
The attention mechanism used by transformers is key to process sequences effectively, and to keep focus to specific parts of the input. When we use transformers, different weights are assigned to each part of the sequence, which allows the model to focus on key aspects of the input.
## Answer 2
RLHF is a way to optimize models by human feedback, by doing evaluations in order to refine the model performance. This differs from fine-tuning instruction, which mainly consists on tagging data
## Answer 3
Methods for efficient parameters fine-tuning includes techniques such as weight pruning, quantization, and knowledge distillation.
For weight pruning we have algorithms like Optimal Brain Damage, that removes less relevant weights, but it may compromise precision. 
Quantization methods, like BERT quantization, reduce the amount of used bits to represent weights, improving efficiency, at the expense of quality. 
Knowledge distillation, as seen in DistilBERT, it's a way to transfer knowledge from a larger model to a smaller one.the smaller model, referred to as the "student," learns not only from the original data but also from the outputs and intermediate representations of the larger or "teacher" model. This approach critically depends on the guidance provided by the larger model, enabling the efficient synthesis of valuable information for smaller model sizes.
