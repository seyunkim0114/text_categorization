from control.config import args

if __name__ == "__main__":
    doc1 = input("Enter the filename of your training document: ")
    doc2 = input("Enter the filename of your test document: ")
    # print("1: ", args.train_data_path)
    args.train_data_path = doc1
    args.test_data_path = doc2

    # print("2: ", args.train_data_path)
    output_doc = input("Enter the filename for your output document: ")

