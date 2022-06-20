import json

with open('input.json') as fin:
    input_data = json.load(fin)

    a = input_data['a']
    b = input_data['b']
    N = input_data['N']

with open('output.json') as fout:
    output_data = json.load(fout)
    s = output_data['s']


if a ** s % N == b:
    print(f'{a} ** {s} = {b} (mod {N})')
    print('PVerification result: PASS')
else:
    print('Verification result: FAIL')
