# A Demo of tensorflow record, slim dataset, and slim data provider
This is a small demo to illustrate how to use tensorflow record, slim dataset, and slim data provider. It also illustrates how to combine the three components so they can work together. For more info about slim dataset, plz check [here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim/python/slim/data/)

Codes contained in this demo did the following:
* Step 1: Create a tensorflow record file given a folder containing images
* Step 2: Create a slim dataset which loads and decode a tensorflow record file
* Step 3: Create a data provider based on a slim dataset
* Step 4: Iterate the dataset over one epoch

## how to run the code
* run load_and_covert_mydata.py. It will generate a TF record file in a folder called data_target. you can open the folder to varify it
* run my_dataset.py. It will read the record file as input, and iterate over the whole dataset and print image, label pairs. There are some important params in the provider API, feel free to tune it.

## Environments
support tensorflow 1.4 

## Notes
* this demo is useful when training a deep network (inception v3 or something like that) from scratch using a big dataset (Imagenet, MS-Celeb-1M). If traning data is small enough to fit into memory, using feed_dict would be more convenient.
* Slim data provider use several queues to maintain intermediate results. Each queue has one or more producers and consumers. This is producer-consumer prototype. For more details, plz read [this note](https://www.tensorflow.org/api_guides/python/reading_data#Preloaded_data)