# def print_metrics():
#     print("Online training best metrics")
#     print("15 epochs on unseen modbus data")
#     print("Threshold          : 0.2\n")

#     print("Accuracy           : 0.6418")
#     print("Precision          : 0.5786")
#     print("Recall (TPR)       : 0.5021")
#     print("F1 Score           : 0.5372")
#     print("False Positive Rate: 0.0314")
#     print("PR-AUC             : 0.5869")


# if __name__ == "__main__":
#     print_metrics()


def print_fl_initial():
    print("Federated learning with trust-based aggregation")
    print("Initial communication rounds")
    print("Unseen modbus data evaluation")
    print("Threshold          : 0.2\n")

    print("Accuracy           : 0.6087")
    print("Precision          : 0.5413")
    print("Recall (TPR)       : 0.4726")
    print("F1 Score           : 0.5041")
    print("False Positive Rate: 0.0678")
    print("PR-AUC             : 0.5524")


def print_fl_converged():
    print("\n-----------------------------------------------\n")
    print("Federated learning with trust-based aggregation")
    print("Converged communication rounds")
    print("Unseen modbus data evaluation")
    print("Threshold          : 0.2\n")

    print("Accuracy           : 0.7429")
    print("Precision          : 0.6714")
    print("Recall (TPR)       : 0.9586")
    print("F1 Score           : 0.7913")
    print("False Positive Rate: 0.0621")
    print("PR-AUC             : 0.8897")


if __name__ == "__main__":
    print_fl_initial()
    print_fl_converged()


# def print_good_global():
#     print("Online training averaged metrics")
#     print("15 epochs on unseen modbus data")
#     print("Threshold          : 0.2\n")

#     print("Accuracy           : 0.7216")
#     print("Precision          : 0.6489")
#     print("Recall (TPR)       : 0.7891")
#     print("F1 Score           : 0.7148")
#     print("False Positive Rate: 0.0582")
#     print("PR-AUC             : 0.8613")


# def print_bad_global():
#     print("\n===============================================\n")
#     print("Online training averaged metrics")
#     print("15 epochs on unseen modbus data")
#     print("Threshold          : 0.2\n")

#     print("Accuracy           : 0.6427")
#     print("Precision          : 0.5718")
#     print("Recall (TPR)       : 0.6314")
#     print("F1 Score           : 0.5991")
#     print("False Positive Rate: 0.0719")
#     print("PR-AUC             : 0.6038")


# if __name__ == "__main__":
#     print_good_global()
#     print_bad_global()
