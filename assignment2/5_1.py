from naive_bayes_classifier.runner import nbc


def main():
    train_acc, test_acc = nbc(1)

    print(f'Training Accuracy: {train_acc:.2f}')
    print(f'Testing Accuracy: {test_acc:.2f}')


if __name__ == '__main__':
    main()
