import requests
from tqdm import tqdm


def get_go_term_info(go_term_list):
    base_url = "https://api.geneontology.org/api/ontology/term/"
    results = []

    for go_id in tqdm(go_term_list):
        go_id = go_id.strip()
        if not go_id.startswith("GO:"):
            go_id = "GO:" + go_id
        try:
            response = requests.get(f"{base_url}{go_id}")
            response.raise_for_status()  # Raise exception for HTTP errors
            data = response.json()
            name = data.get("label", "Name not found")
            definitions = data.get("definition", {})
            definition = definitions if definitions else "Definition not found"
            result = f"name:{name} , definition:{definition}"
            print(go_id, ":", result)
            results.append(result)

        except requests.exceptions.RequestException as e:
            results.append(f"Error fetching information for {go_id}: {str(e)}")
        except (KeyError, IndexError) as e:
            results.append(f"Error parsing information for {go_id}: {str(e)}")

    return results


def prep_go():
    with open("data/mf/go.txt", "r") as f:
        go_terms = f.read().splitlines()
    results = get_go_term_info(go_terms)
    with open("data/mf/go_info.txt", "w") as f:
        for result in results:
            f.write(result + "\n")

    return results


def prep_protein(split):
    with open(f"data/mf/{split}_1.txt", "r") as f:
        lines = f.read().splitlines()
    lines = [l.replace(".", "") for l in lines]
    with open(f"data/mf/{split}.txt", "w") as f:
        for line in lines:
            f.write(line + "\n")


def prep_pairs(split, go_terms):
    with open(f"data/mf/{split}.txt", "r") as f:
        proteins = f.read().splitlines()
    with open(f"data/mf/{split}_labels.txt", "r") as f:
        labels_lines = f.read().splitlines()
    labels = []
    for line in labels_lines:
        line = line.split()
        line = [int(float(x)) for x in line]
        labels.append(line)
    assert len(labels) == len(proteins), "Labels and proteins length mismatch"
    print("Split:", split,"Number of proteins:", len(proteins), "Number of proteins:", len(labels))

    pos_pairs = []
    neg_pairs = []
    for i in range(len(proteins)):
        protein = proteins[i]
        label = labels[i]
        for i, l in enumerate(label):
            pair = (protein, go_terms[i])
            if l == 0:
                neg_pairs.append(pair)
            else:
                pos_pairs.append(pair)

    print("Split:", split, "Number of positive pairs:", len(pos_pairs), "Number of negative pairs:", len(neg_pairs))

    with open(f"data/mf/{split}_reaction.txt", "w") as f_src:
        with open(f"data/mf/{split}_enzyme.txt", "w") as f_tgt:
            for pair in pos_pairs:
                f_src.write(pair[0] + "\n")
                f_tgt.write(pair[1] + "\n")
    with open(f"data/mf/{split}_reaction_neg.txt", "w") as f_src:
        with open(f"data/mf/{split}_enzyme_neg.txt", "w") as f_tgt:
            for pair in neg_pairs:
                f_src.write(pair[0] + "\n")
                f_tgt.write(pair[1] + "\n")


if __name__ == "__main__":
    # prep_go()
    prep_protein("train")
    prep_protein("test")
    prep_protein("valid")

    with open("data/mf/go_info.txt", "r") as f:
        go_terms = f.read().splitlines()

    prep_pairs("train", go_terms)
    prep_pairs("test", go_terms)
    prep_pairs("valid", go_terms)
