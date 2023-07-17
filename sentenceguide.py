from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def find_similar_names(garbage_name, name_list, topn=10):
    vectorizer = TfidfVectorizer()
    name_vectors = vectorizer.fit_transform(name_list + [garbage_name])
    similarity_scores = cosine_similarity(name_vectors[:-1], name_vectors[-1])
    similar_names_indices = similarity_scores.argsort(axis=0)[-topn:].flatten()[::-1]
    similar_names = [(name_list[i], similarity_scores[i]) for i in similar_names_indices]
    return similar_names

def find_best_match(predicted_names, first_two_letters):
    best_match = None
    best_score = 0
    for name, score in predicted_names:
        if name[:2].lower() == first_two_letters.lower() and len(name) > 2 and score > best_score:
            best_match = name
            best_score = score
    return best_match

def thisistheway(word):
    # Example usage
    file_path = 'words_alpha.txt'
    garbage_name = word.lower()
    name_list=[]
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if(len(line.strip())>2 and line.strip()[:2]==garbage_name[:2]):
                name_list.append(line.strip())
        
    predicted_names = find_similar_names(garbage_name, name_list)

    if predicted_names:
        print(f"Similar names to '{garbage_name}':")
        #for name, score in predicted_names:
        #   print(f"{name} (Similarity Score: {score})")
        print(predicted_names[0][0])
    else:
        print(f"No similar names found for '{garbage_name}'.")
