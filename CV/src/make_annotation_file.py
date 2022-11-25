import os
import pandas as pd

def make():
    print("Plese type mode(train or test)")
    mode = input(">")
    print("Please type annotation file name")
    filename = input(">")
    print("Please type basic directory")
    base_dir = input(">")

    temp_data_list = []
    if(mode == "train"):
        for path_idx, path in enumerate(os.listdir(base_dir)):

            temp_list = [[path + '//' + img_path, path_idx] for img_path in os.listdir(base_dir + '//' + path)]
            temp_data_list += temp_list

        annotations = pd.DataFrame(columns=["img_path", "target"], data=temp_data_list)
        annotations.to_csv(f"{filename}.csv", index=False)

    else:

        temp_data_list = [[f"{img_path}", int(img_path.split('.')[0])] for img_path in os.listdir(base_dir)]
        annotations = pd.DataFrame(columns=["img_path", "id"], data=temp_data_list)
        annotations.to_csv(f"{filename}.csv", index=False)

    print("=====ANNOTATION FILE CREATED=====")
    
if __name__ == '__main__':
    make()