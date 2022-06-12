import json

from shor_discrete_log import shor_discrete_log


def classical_discrete_log(a, b, N):
    for answer in range(1, N):
        if a ** answer % N == b:
            return answer


with open('input.json') as fin:
    input_data = json.load(fin)

    a = input_data['a']
    b = input_data['b']
    N = input_data['N']

    s = shor_discrete_log(a, b, N)

    print(s)

with open('output.json', 'w') as fout:
    json.dump({'s': s}, fout)
