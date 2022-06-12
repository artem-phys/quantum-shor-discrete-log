from math import ceil
import numpy as np

from fractions import Fraction

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, transpile
from qiskit.circuit.library import QFT
from qiskit.extensions import UnitaryGate


def calculate_period(a, N):
    product = a
    i = 1
    while product != 1 and i != N:
        product = product * a % N
        i += 1
    return i


def modinv(a, N):
    r = calculate_period(a, N)
    return a ** (r - 1) % N


def shor_discrete_log(a, b, N):

    r = calculate_period(a, N)

    eval_qubits = ceil(np.log2(r))
    state_qubits = ceil(np.log2(N))
    n_qubits = 2 * eval_qubits + state_qubits

    def f(x1, x2):
        return (b ** x1 * a ** x2) % N

    def discrete_log_oracle():

        outputs = np.zeros(2 ** n_qubits, dtype=int)

        for state in range(2 ** n_qubits):

            state_bin = format(state, f"0{n_qubits}b")

            y_bin = state_bin[:state_qubits]
            x2_bin = state_bin[state_qubits: state_qubits + eval_qubits]
            x1_bin = state_bin[state_qubits + eval_qubits:]

            x1 = int(x1_bin, 2)
            x2 = int(x2_bin, 2)
            y = int(y_bin, 2)

            y_xor_f_bin = format(f(x1, x2) ^ y, f"0{state_qubits}b")

            output_bin = y_xor_f_bin + x2_bin + x1_bin

            outputs[state] = int(output_bin, 2)

        # Unitary matrix building by columns
        U = np.zeros((2 ** n_qubits, 2 ** n_qubits), dtype=int)

        for y in range(2 ** n_qubits):
            for i in range(2 ** n_qubits):
                U[i][y] = 1 if outputs[y] == i else 0

        return UnitaryGate(U, label=f'Oracle for {a}^x1 * {b}^x2 mod {N}')

    qc = QuantumCircuit(QuantumRegister(eval_qubits, 'x1'), QuantumRegister(eval_qubits, 'x2'), QuantumRegister(state_qubits, 'y'), ClassicalRegister(eval_qubits, 'cl_x1'), ClassicalRegister(eval_qubits, 'cl_x2'))

    # Initialize evaluation qubits
    for q in range(2 * eval_qubits):
        qc.h(q)

    qc.barrier()

    # Oracle
    qc.append(discrete_log_oracle(), range(n_qubits))

    # Do inverse-QFT
    qc.barrier()
    inverse_QFT_gate = QFT(eval_qubits, inverse=True, name='  IQFT').to_gate()
    qc.append(inverse_QFT_gate, range(eval_qubits))
    qc.append(inverse_QFT_gate, range(eval_qubits, 2 * eval_qubits))

    # Measure circuit
    qc.measure(range(eval_qubits * 2), range(eval_qubits * 2))

    # Simulate Results
    backend = Aer.get_backend('aer_simulator')
    n_shots = 32
    job = backend.run(transpile(qc, backend), shots=n_shots, memory=True)
    memory = job.result().get_memory()

    for readout in memory:
        l2r, sl2r = [int(r, 2) / (2 ** eval_qubits) for r in readout.split()]

        f1 = Fraction(sl2r).limit_denominator(r)
        f2 = Fraction(l2r).limit_denominator(r)

        s_candidate = f1.numerator * modinv(f2.numerator, r) * f2.denominator // f1.denominator % r

        if a ** s_candidate % N == b:
            return s_candidate
