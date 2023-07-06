# ner_system
The system for Named Entity Recognition with the option of fine-tuning on custom datasets.

### Launch the service

```shell
docker-compose up --build ner
```

### Set GPU number

In ner\_system/docker-compose.yml file set the `CUDA_VISIBLE_DEVICES` variable for `ner` service.

### Launch on GPU or CPU

In [configuration file](https://github.com/dmitrijeuseew/ner_system/blob/4a33589a7968fc38253f2deeeceae7b0b4612d04/services/ner/ner_rured.json#L37) set the parameter `device` cpu or gpu.

### Examples of the request and output data structure

In `[this file] <https://github.com/dmitrijeuseew/ner_system/blob/main/services/ner/example.py>`__ you can find description of output data elements and example of a query to the service.
