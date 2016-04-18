__author__ = 'Deyang'


def get_lcs_length(word_id_seq1, word_id_seq2):
    """
    Should take two list of word ids instead of raw stings.
    Other wise white spaces etc. taken into account

    :param word_id_seq1:
    :param word_id_seq2:
    :return:
    """
    m = len(word_id_seq1)
    n = len(word_id_seq2)
    # An (m+1) times (n+1) matrix
    C = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if word_id_seq1[i-1] == word_id_seq2[j-1]:
                C[i][j] = C[i-1][j-1] + 1
            else:
                C[i][j] = max(C[i][j-1], C[i-1][j])
    return C[-1][-1]

print get_lcs_length('12343010', '01010101')