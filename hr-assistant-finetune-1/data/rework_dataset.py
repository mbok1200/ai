import json

with open("data/merged_with_links.jsonl", "r") as fin, open("data/merged_with_links_is.jsonl", "w") as fout:
    for line in fin:
        ex = json.loads(line)
        # metadata = ex.get("metadata", {})
        # output = ex.get("output", "")
        ex["instruction"] = "Проаналізуй цей запит"
        # ex["output"] = {
        #     "short_text": output,
        #     "metadata": metadata
        # }
        fout.write(json.dumps(ex, ensure_ascii=False) + "\n")