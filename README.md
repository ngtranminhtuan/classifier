# Summarize:
Below is performance chart:
![image alt text](<assets/confusion_matrix_epoch_80.png>)
![image alt text](<assets/metrics_plot.png>)

I don't have GPU. We can speed up training and inference by parallel compute with accelerator(CUDA, TPU, DSP...) by using torch to device, and increase batch size(16, 32, 64, 128, 256... based on your VRAM). 

If your VRAM is not enough, let's decrease batch size.


```bash
model.to(device)
image.to(device)
```
## Install git lfs

```bash
sudo apt-get update && sudo apt-get install git-lfs
git clone https://github.com/ngtranminhtuan/classifier
cd classifier
git lfs install
git lfs pull
```

## Train/Eval/Inference
### Inference
In container bash
``` bash
python inference.py --image_url=checkbox_state_v2/data/val/other/fde4d694c0fdff8e7f4c7e99b34678ec.png
```

### Train/Eval
``` bash
docker build -t image_name:tag .
docker run -it --name your-container-name -v $PWD:/usr/src/app your-image-name:tag
python train.py
```

## Training & Inference runtime analysis and optimization
### Benchmark:
Accuracy: 91.35%\n
Average Inference Time: 0.007960 seconds per sample
``` bash
python benchmark.py
python quantize_prune.py
```

After quantize and pruning, if accuracy is same as best_model, we can speed up model!

## Data Analysis, Preprocessing and Augmentations
There is multi-class classification problem (not multi-label), so we can use CrossEntropyLoss for output class name and confident score of every class.

After explore dataset, we see that have some images can be confused:
+ checked/0aabda81e5aa3cccbae391d6231238d8.png
+ checked/4c735d69860ff3c5b089ce64bd7ce846.png
+ checked/f081c3ca4cef8dc89c87e1b0b3464d89.png
+ unchecked/31d017d3bb3a3c4ead0f63bf53221ae0.png
+ unchecked/70daa7dbc6c317fc9c656d0db26a084b.png
+ unchecked/98c19fd0227f5b20810eb28cfab18dba.png
+ unchecked/be39967bbaf41b4e6617da73e5ebccad.png

If I remove this image, Accuracy will higher, but in this phase, I'll hold to training. We can add more data, or use another backbone model
to learning this features.

Preprocessing and Augmentation: I split datasets to train/val folder for train and evaluation.
``` bash
python data_preprocessing.py
```
In training, for avoid over-fitting, I'm using shuffle images, and validation is not shuffle.

Because symbol is "Symmetry" and Design carefully, so I don't use FLIP or complex Augmentation and keep transformation easiest. I resize images to 224x224 and transform to mean,std zone 
mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].

We can tuning in this point!

## Model design
![image alt text](<assets/ResNet50.png>)
I'm using backbone ResNet50 for easy building pipeline. We can use YoloV8 for best extractor.
Because this problem is multi-class, so I'm using nn.CrossEntropyLoss. (If multi-label, need to use Binary Cross Entropy and Sigmoid).

## The training code itself, i.e., the optimizer, default hyperparameters, epochs, etc.
+ We can load best model and continue to training.
+ Optimizer: In small datasets, I choose SGD, we can try Adam or switch based to convergence rate.
+ optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
+ patience = 200 (For early stopping avoid overfiting)
+ num_epochs=500

## Evaluation code i.e. how did you split the data, when do you evaluate the model.
+ After train, we loop to evaluate. We can use cross-validation, but in small dataset, it can be over-fitting.
+ I use validation loss, Top_1 Accuracy, Top_K_Accuracy and Confusion matrix for update performance of model.
+ If val_loss < best_val_loss and val_acc1 >= best_val_acc1, I'll save best model.

## Hyperparameter tuning
+ Optimizer: We tune optimizer types: SGD, Adam, AdamW...
+ Tuning momentum, learning rate, patience, mean/std of data distribution.

## Suggestions on how can you improve this system in the future.
+ Can modify loss function for class weighting bias (because unbalanced dataset), or 
using over-sampling (can be over-fitting)
+ Real-life Application: we need to use Triton/TensorRT for dynamic batching inference -> maximize GPU resource.
+ Add more data every class, increase quality of dataset.
+ To improve speed, we can deep-pruning cluster of graph by Torch Pruning and Quantize to Int8.
+ Try to change model size of ResNet, or other architecture likely Transformer: We
train embedding space (unsupervised) and downstream task classification(supervised).
+ Abstract to class for scalable source code.
