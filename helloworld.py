print("hello worlds")

print("hello worlds")

def compare_pr_issues(json_str1, json_str2, threshold=0.85):
    """
    Compare 'issue' fields from comments of two AI-generated PR review JSONs.
    Only compares comments referring to the same file and line.

    Parameters:
    - json_str1, json_str2: Raw JSON strings from LLMs
    - threshold: cosine similarity threshold

    Returns:
    - List of tuples: (index_in_json1, index_in_json2, similarity_score)
    """
    # Load and validate JSON
    try:
        data1 = json.loads(json_str1)
        data2 = json.loads(json_str2)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON input: {e}")

    comments1 = data1.get("comments", [])
    comments2 = data2.get("comments", [])

    # Build pairs only with matching file+line
    matched_pairs = []
    for i, c1 in enumerate(comments1):
        for j, c2 in enumerate(comments2):
            if c1.get("file") == c2.get("file") and c1.get("line") == c2.get("line"):
                matched_pairs.append(((i, j), c1["issue"], c2["issue"]))

    if not matched_pairs:
        return []

    # Compute embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    results = []

    for (i, j), issue1, issue2 in matched_pairs:
        embeddings = model.encode([issue1, issue2])
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        if sim > threshold:
            results.append((i, j, sim))

    return results