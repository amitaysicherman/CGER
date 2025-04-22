import requests
from tqdm import tqdm


def get_go_term_info(go_term_list):
    """
    Fetches information about GO terms using the Gene Ontology API.

    Args:
        go_term_list (list): List of GO term IDs (e.g., ['GO:0016681', 'GO:0005739'])

    Returns:
        list: List of formatted strings with name and definition for each GO term
    """
    base_url = "https://api.geneontology.org/api/ontology/term/"
    results = []

    for go_id in tqdm(go_term_list):
        # Clean the GO ID to ensure proper format
        go_id = go_id.strip()
        if not go_id.startswith("GO:"):
            go_id = "GO:" + go_id

        try:
            # Make API request to the Gene Ontology
            response = requests.get(f"{base_url}{go_id}")
            response.raise_for_status()  # Raise exception for HTTP errors

            data = response.json()

            # Extract name and definition
            name = data.get("label", "Name not found")

            # Definitions in GO are typically stored as a list with the first item containing the text
            definitions = data.get("definition", {})
            definition = definitions if definitions else "Definition not found"

            # Format the result string
            result = f"name:{name} , definition:{definition}"
            print(go_id, ":", result)
            results.append(result)

        except requests.exceptions.RequestException as e:
            results.append(f"Error fetching information for {go_id}: {str(e)}")
        except (KeyError, IndexError) as e:
            results.append(f"Error parsing information for {go_id}: {str(e)}")

    return results


def main():
    """
    Main function to demonstrate usage.
    """
    # Example usage
    with open("data/mf/go.txt", "r") as f:
        go_terms = f.read().splitlines()
    results = get_go_term_info(go_terms)
    with open("data/mf/go_info.txt", "w") as f:
        for result in results:
            f.write(result + "\n")

    return results


if __name__ == "__main__":
    main()
