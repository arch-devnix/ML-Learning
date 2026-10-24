"""
A single neuron that learns to predict pass/fail based on
hours slept (x1) and hours studied (x2). Uses sigmoid + cross-entropy.

Inputs is the info given
Weights are esentially the learned numbers. It starts random and then it's multiplied by the input to produce that inputs contribution to z
Bias is the bar shifting; how easily the neuron fires.
"""

import random
import math

# Helpers

def neuron(x1, x2, w1, w2, b):
    return (x1 * w1 + x2 * w2) + b

def sigmoid(z):
    z = max(min(z, 500), -500)
    return 1 / (1 + math.exp(-z))

def loss(out, label):
    out = max(min(out, 0.9999), 0.0001) 
    return -(label * math.log(out) + (1 - label) * math.log(1 - out))

# Logic
def main(learning_rate):

    w1 = random.uniform(-1, 1)
    w2 = random.uniform(-1, 1)
    b  = random.uniform(-1, 1)
    data = [
        (7, 3, 0),
        (8, 6, 1),
        (6, 2, 0),
        (9, 8, 1),
        (5, 1, 0),
        (7, 5, 1),
        (6, 4, 0),
        (8, 7, 1),
        (10, 2, 0),
        (5, 9, 1),
        (7, 6, 1),
        (8, 1, 0),
        (6, 5, 1),
        (9, 3, 0),
        (4, 4, 0),
        (8, 8, 1),
        (7, 2, 0),
        (6, 7, 1),
        (9, 6, 1),
        (5, 3, 0)
    ]

    for epoch in range(1000):
        total_loss = 0
        for x1, x2, label in data:
            z   = neuron(x1, x2, w1, w2, b)
            out = sigmoid(z)

            total_loss += loss(out, label)

            err = out - label
            w1 -= learning_rate * err * x1
            w2 -= learning_rate * err * x2
            b  -= learning_rate * err
        if epoch % 100 == 0:
            print(f"epoch {epoch}: w1={w1:.3f} w2={w2:.3f} b={b:.3f}")

    print("\nAFTER TRAINING")
    correct = 0
    for x1, x2, label in data:
        z   = neuron(x1, x2, w1, w2, b)
        out = sigmoid(z)
        
        prediction = 1 if out > 0.5 else 0
        print(f"[{x1},{x2}] z={z:.2f} | prob={out:.3f} | pred={prediction} | label={label}")

        if prediction == label:
            correct += 1

    print(f"\nFinal: {correct}/{len(data)} correct | Acc {(correct/len(data)):.0%}")


if __name__ == "__main__":
    main(0.01)
