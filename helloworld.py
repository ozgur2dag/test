import json
from typing import List, Tuple, Optional, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load model globally to avoid reloading every time
_model = SentenceTransformer("all-MiniLM-L6-v2")


def parse_json_comments(json_str: str) -> List[Dict[str, Any]]:
    """Parse and extract comments from a JSON string."""
    try:
        data = json.loads(json_str)
        return data.get("comments", [])
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON input: {e}")


def find_matching_pairs(comments1: List[Dict[str, Any]], comments2: List[Dict[str, Any]]) -> List[Tuple[int, int, str, str]]:
    """Find comment pairs with matching file and line numbers."""
    matches = []
    for i, c1 in enumerate(comments1):
        for j, c2 in enumerate(comments2):
            if c1.get("file") == c2.get("file") and c1.get("line") == c2.get("line"):
                issue1 = c1.get("issue", "").strip()
                issue2 = c2.get("issue", "").strip()
                if issue1 and issue2:
                    matches.append((i, j, issue1, issue2))
    return matches


def compute_similarity(issue1: str, issue2: str) -> float:
    """Compute cosine similarity between two issue descriptions."""
    emb1, emb2 = _model.encode([issue1, issue2])
    return float(cosine_similarity([emb1], [emb2])[0][0])


def compare_pr_issues(json_str1: str, json_str2: str, threshold: float = 0.85, verbose: bool = False) -> List[Tuple[int, int, float]]:
    """
    Compare 'issue' fields from two PR review JSON strings.

    Args:
        json_str1: JSON string from first model.
        json_str2: JSON string from second model.
        threshold: Similarity threshold to consider a match.
        verbose: If True, prints matching issues and their scores.

    Returns:
        A list of (index_in_json1, index_in_json2, similarity_score) for matched comments.
    """
    comments1 = parse_json_comments(json_str1)
    comments2 = parse_json_comments(json_str2)
    matched_pairs = find_matching_pairs(comments1, comments2)

    if not matched_pairs:
        return []

    results = []
    for i, j, issue1, issue2 in matched_pairs:
        score = compute_similarity(issue1, issue2)
        if verbose:
            print(f"[{i}, {j}] → Score: {score:.3f} | Issue1: {issue1[:60]} | Issue2: {issue2[:60]}")
        if score >= threshold:
            results.append((i, j, score))

    return results


# Example test usage
if __name__ == "__main__":
    print("PR Issue Comparator Ready. Run with test JSON strings.")
