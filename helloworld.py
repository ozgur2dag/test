import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def compare_pr_issues(json_str1, json_str2, threshold=0.85):
    """
    Compare 'issue' fields from comments of two AI-generated PR review JSONs.
    Only compares comments referring to the same file and line.

    Parameters:
        json_str1 (str): Raw JSON string from LLM 1
        json_str2 (str): Raw JSON string from LLM 2
        threshold (float): Cosine similarity threshold for matching issues

    Returns:
        List[Tuple[int, int, float]]: Tuples of (index_in_json1, index_in_json2, similarity_score)
    """
    try:
        data1 = json.loads(json_str1)
        data2 = json.loads(json_str2)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON input: {e}")

    comments1 = data1.get("comments", [])
    comments2 = data2.get("comments", [])

    # Find comment pairs matching on file and line
    matched_pairs = [
        ((i, j), c1["issue"], c2["issue"])
        for i, c1 in enumerate(comments1)
        for j, c2 in enumerate(comments2)
        if c1.get("file") == c2.get("file") and c1.get("line") == c2.get("line")
    ]

    if not matched_pairs:
        return []

    model = SentenceTransformer("all-MiniLM-L6-v2")
    results = []

    for (i, j), issue1, issue2 in matched_pairs:
        emb1, emb2 = model.encode([issue1, issue2])
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        if similarity > threshold:
            results.append((i, j, similarity))

    return results

# Example use
if __name__ == "__main__":
    print("Comparing PR issues...")