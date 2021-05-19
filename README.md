# Metric-Learning-Losses-Exploration


## Probem1 – Sampling method comparison
- Triplet loss: implemented using the PairwiseDistance method of pytorch as distance measure, and a custom method to extract a random negative sample from the batch for each sample.
- Topk loss: I used the strategy we saw in class to sample the top 3 negatives. I used a batch size of 12 samples, and I tuned k=3. With higher values of k the model was selecting values not relevant to the loss (“not enough violation”). I sampled topk values both row wise and column wise, in order to consider violations in both views perspective.
- Semi hard loss: Instead of selecting the topk values as before I select a range of values define by a lower bound and a margin.

## Problem2 – Selecting a loss function
Constrastive loss: using a mask that maps the labels that are equal in the rows and columns of the similarity matrix, this method computes the constrastive loss using the mask as a binary label indicator (0,1) for each samples pairs image-phrase and the similarity matrix values as a distance measure between features.
Temperature-scaled cross entropy loss: I tuned tau as equal to 0.07 after multiple runs (no relevant changes if I consider tau lower than 0.07). To separate positive and negative correspondence from the similarity matrix I used 2 masks, one that maps the positives (labels match), and one for the negatives (labels don’t match).

## Problem3 – Memory module
I had some issues to implement correctly the memory module. My last version of the memory module (built as a python class) is implemented in the following way:
- For the first batch, when the memory is empty, the memory class module call the method “loss” to compute the cross entropy loss normally within the batch.
- After computing the loss the “add_to_memory” method select the top violations for each sample in the batch, and add the corresponding features vector (image features) in memory.
- From the second epoch until the end, the model will first use the “read_memory” method to extract all the features samples stored and then compute the loss considering both the distances within the batch and the distances between batch samples (phrases features only) and the saved image features stored in memory.

## Problem4 – Incorporating depth
For the last problem of the homework I tried 3 different implementations to include the 3rd view in the model:
- Average values between the 2 image features vectors (normal and depth) and the resulting vector used to compute the constrastive loss with the phrase features vector.
- Combination of the 2 images before the projection into the feature space; in this way the output feature vector will represent both the images.
- Using depth image as an augmented version of the original image and computing the constrastive loss between the 2. In this Case I have combined the loss computation for the normal batch (images + phrases) with the 3rd view loss computation (images+depth). The first part of the loss wants to optimize the match between images and phrases, the second wants to optimize the projection of the images into a features space where similar images stay close together.

## Model comparison

![Alt text](/git-docs/results.JPG ) 
