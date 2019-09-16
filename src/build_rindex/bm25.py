from math import log

k1 = 1.2
k2 = 100
b = 0.75
R = 0.0


def score_BM25(n, f, qf, r, N, dl, avdl):
    K = compute_K(dl, avdl)
    first = log(((r + 0.5) / (R - r + 0.5)) / ((n - r + 0.5) / (N - n - R + r + 0.5)))
    second = ((k1 + 1) * f) / (K + f)
    third = ((k2 + 1) * qf) / (k2 + qf)
    return first * second * third


def compute_K(dl, avdl):
    return k1 * ((1 - b) + b * (float(dl) / float(avdl)))


def get_query_result(terms, idx, dlt, dlt_total_length, dlt_avdl):
    # dlt_total_length = len(dlt)
    # dlt_avdl = dlt.get_average_length()
    query_result = dict()
    for term in terms:
        if term in idx.index:
            doc_dict = idx.index[term]  # retrieve index entry
            for docid, freq in doc_dict.iteritems():  # for each document and its word frequency
                score = score_BM25(n=len(doc_dict), f=freq, qf=1, r=0, N=dlt_total_length,
                                   dl=dlt.get_length(docid), avdl=dlt_avdl)  # calculate score
                if docid in query_result:  # this document has already been scored once
                    query_result[docid] += score
                else:
                    query_result[docid] = score
    return query_result
