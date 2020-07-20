### LTR Modeling

The pipeline is preset to train model in offline mode. However, if you have a Solr server running you can pull data from it, save it to use later. (Although it might require a few tweaks). Tested approach though is with offline mode.

#### Step 1

Pull data from Solr server
```python3 pull_data.py --base_url http://solrhost.com/solr/ --rows 1000 --q fancy bed --write_name sample_1sb```
This will pull data and store it in the file sample_1sb.csv

#### Step 2

Run training with LTRSimpleModel adjusting some hyperparameters
```python3 train.py --epochs 50 -lr 0.0001 -wd 0.01```

#### Step 3

Go drink coffee

### Disclamer

Even though model is not ready to be deployed in production I hope I helped to do some heavy lifting in terms of organizing it. There are two models I tried to build (one simple containing linear layers only, another doing more sophisticated transformations and here is my concern: weights of linear layers are easily transformed to Solr format, other layers might not be or at least I do not know how :) )