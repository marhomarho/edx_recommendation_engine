from flask import Flask, request, jsonify

import random
from sklearn.feature_extraction.text import TfidfVectorizer
import json

app = Flask(__name__)

@app.route('/get_recom_courses', methods=['GET', 'POST'])
def get_recom_courses():
    """ Get recommended list of courses to follow based on the courses that user already took 
    """
    content = request.get_json(silent=True)
    js_req = json.loads(content)
    print(content)
    # Payload example
    # {'uuid':"stringaacaso", 'sample_course':{corsoA}, 'catalog':[{corso1},{corso2,...}]}
    
    uuid = js_req['uuid']
    print("sample_course", len(js_req['sample_course']))
    print("catalog", len(js_req['catalog']))
    sample_course = js_req['sample_course']
    print("sample_course", sample_course['short_description'])
    available_courses = js_req['catalog']
    similar_courses = get_similar_courses(sample_course, available_courses)
    print(similar_courses)

    return jsonify({'uuid':uuid, 'recom':similar_courses})
    
def get_similararities(query, input_corpus):
    corpus=[]
    corpus.extend([query])
    corpus.extend(input_corpus)
    vect = TfidfVectorizer(min_df=1, stop_words="english")
    tfidf = vect.fit_transform(corpus)
    pairwise_similarity = tfidf * tfidf.T
    sorted_similar = sort_sparse_matrix(pairwise_similarity)
    print(pairwise_similarity.toarray())
    similar_corpus_ids = [el-1 for el in sorted_similar[0][1:]]
    return similar_corpus_ids

def sort_sparse_matrix(m, rev=True, only_indices=True):
    """ Sort a sparse matrix and return column index dictionary
    """
    col_dict = dict()
    for i in range(m.shape[0]): # assume m is square matrix.
        d = m.getrow(i)
        s = zip(d.indices, d.data)
        sorted_s = sorted(s, key=lambda v: v[1], reverse=True)
        if only_indices:
            col_dict[i] = [element[0] for element in sorted_s]
        else:
            col_dict[i] = sorted_s
    return col_dict

def get_similar_courses(sample_course, available_courses):
    sample_descript = sample_course['short_description'] if sample_course['short_description'] else ""

    # Remove courses with empty description
    for el in available_courses:
        if el['short_description'] == None:
            available_courses.remove(el)
    corpus_courses = [el['short_description'] for el in available_courses]

    print("Sample course", sample_descript)
    print("corpus_courses", corpus_courses)

    idx_similar_courses = get_similararities(sample_descript, corpus_courses)
    similar_courses = [available_courses[i] for i in idx_similar_courses] 
    print("Similar_courses len",len(similar_courses))
    return similar_courses


if __name__ == '__main__':
    app.run(host= '0.0.0.0', port=5000, debug=False)

