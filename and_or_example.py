from assets.Neuron import NeuronCol
import matplotlib.pyplot as plt
import argparse

training_data = {
    # given inputs at index 1, 2, result is 3 index 0 is bias.
    "AND": [
        (0, 1, 1, 1),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 0)
    ],
    "OR": [
        (0, 1, 1, 1),
        (0, 1, 0, 1),
        (0, 0, 1, 1),
        (0, 0, 0, 0)
    ]
}


def main(args):
    if args.AND:
        data = training_data["AND"]
    else:
        data = training_data["OR"]

    input_layer = NeuronCol(3)
    output_layer = NeuronCol(1)

    input_layer.insert_bias()
    input_layer.connect(output_layer.neurons)

    errors = []
    avg = []
    for i in range(500):
        for input_data in data:
            input_layer.set_values(input_data[:-1])
            input_layer.fire()
            output_layer.compute_held()

            output_layer.neurons[0].compute_error_variance(input_data[-1])

            # calculate output error
            output_layer.output_error([input_data[-1]])

            # save the error
            errors.append(output_layer.neurons[0].j)

            print("I think {}-{}-{} should result in {} with {} confidence!".format(
                input_data[0], input_data[1], input_data[2], output_layer.neurons[0].held, 1 - output_layer.neurons[0].j
            ))

            input_layer.adjust_weights()

            output_layer.clean()

        avg.append(sum(errors)/len(errors))
        errors.clear()

    # display results
    plt.plot(avg)
    plt.ylabel("Error Rate")
    plt.xlabel("Iterations")
    plt.title("Error Rate over Time")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a neural net to simulate an AND / OR gate.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-A', '--AND', action="store_true", help="Trains an AND gate."
    )
    group.add_argument(
        '-O', '--OR', action="store_true", help="Trains an OR gate."
    )
    main(parser.parse_args())

