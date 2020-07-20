import numpy as np
import requests
import simplejson

# Number of documents to be re-ranked.
RERANK = 50
with open("RERANK.int", "w") as f:
    f.write(str(RERANK))

# Build query URL.
q_id = row["id"]
title = row["title"].strip().lower()
query = row["query"].strip().lower()
text = " ".join([title, query])

url = "http://localhost:8983/solr/homedepot/query"
url += "?q={0}&df=text&rq={{!ltr model=my_efi_model ".format(text)
url += "efi.title='{0}' efi.query='{1}' efi.text='{2}'}}".format(title, query, text)
url += "&fl=id,score,[features]&rows={1}".format(text, RERANK)

response = requests.request("GET", url)
try:
    json = simplejson.loads(response.text)
except simplejson.JSONDecodeError:
    print(q_id)

if "error" in json:
    print(q_id)

# Extract the features.
results_features = []
results_targets = []
results_ranks = []
add_data = False

for (rank, document) in enumerate(json["response"]["docs"]):

    features = document["[features]"].split(",")
    feature_array = []
    for feature in features:
        feature_array.append(feature.split("=")[1])

    feature_array = np.array(feature_array, dtype = "float32")
    results_features.append(feature_array)

    doc_id = document["id"]
    # Check if document is relevant to query.
    if q_id in relevant.get(doc_id, {}):
        results_ranks.append(rank + 1)
        results_targets.append(1)
        add_data = True
    else:
        results_targets.append(0)

if add_data:
    np.save("{0}_X.npy".format(q_id), np.array(results_features))
    np.save("{0}_y.npy".format(q_id), np.array(results_targets))
    np.save("{0}_rank.npy".format(q_id), np.array(results_ranks))