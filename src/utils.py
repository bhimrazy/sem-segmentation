import random
from typing import List, Tuple
from src.io import load_data


def split_data(
    X: List[str],
    y: List[str],
    test_size: float,
    valid_size: float,
    random_state: int,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    Splits the data into train, validation and test sets.

    By default it uses startified sampling to ensure that the proportion of
    each class is the same in each set.

    Args:
        X (List[str]): List of image paths.
        y (List[str]): List of image masks.
        test_size (float): Proportion of data to allocate to the test set.
        valid_size (float): Proportion of data to allocate to the validation set.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
        X_train, X_valid, X_test, y_train, y_valid, y_test
    """

    # Set the random seed for reproducibility
    random.seed(random_state)

    # Combine X and y into a list of tuples
    combined_data = list(zip(X, y))

    # Shuffle the combined data
    random.shuffle(combined_data)

    # Calculate the number of samples for validation and test sets
    total_samples = len(combined_data)
    test_num = int(total_samples * test_size)
    valid_num = int(total_samples * valid_size)

    # Split the combined data into train, validation, and test sets
    train_data = combined_data[test_num + valid_num :]
    valid_data = combined_data[test_num : test_num + valid_num]
    test_data = combined_data[:test_num]

    # Unpack the split data
    X_train, y_train = zip(*train_data)
    X_valid, y_valid = zip(*valid_data)
    X_test, y_test = zip(*test_data)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


if __name__ == "__main__":
    # load data
    dataset_path = "data/sem-activated-carbon-dataset"
    images, masks = load_data(dataset_path)

    print("Number of images:", len(images))
    print("Number of masks:", len(masks))

    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(
        images,
        masks,
        test_size=0.2,
        valid_size=0.2,
        random_state=42,
    )

    print("Number of training images:", len(X_train))
    print("Number of validation images:", len(X_valid))
    print("Number of test images:", len(X_test))

    # Ratio:
    print("Ratio of training images:", len(X_train) / len(images))
    print("Ratio of validation images:", len(X_valid) / len(images))
    print("Ratio of test images:", len(X_test) / len(images))
