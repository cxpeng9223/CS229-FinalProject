import random

#Read the Raw Data File
RawData = open("Data.txt","r")
Data = RawData.readlines()

RawData.close()

#Inputing features
variablename = ["age", "team", "league", "position", "game_played", "game_started", "min_pg", "fg_pg", "fga_pg", \
                "fg_pct", "fg3_pg", "fg3a_pg","fg3_pct", "fg2_pg", "fg2a_pg", "fg2_pct", "efg_pct", "ft_pg", "fta_pg",\
                "ft_pct", "orb_pg", "drb_pg", "trb_pg", "ast_pg","stl_pg", "blk_pg", "tov_pg", "pf_pg", "pts_pg", \
                "draft_order", "shoot_hand", "player_height", "player_weight"]

#Initializing training and test sets
train_data_dict = {}
test_data_dict = {}

#Initializing features in the training and test sets
for name in variablename:
    train_data_dict[name] = []
    test_data_dict[name] = []

#Assign features to the training and test sets
for i in range(0, len(Data)):
    decision = random.random() #assgin score to each sample
    d = Data[i].split("  ")
    # Bad point constrains

    if d[1] != "TOT" and float(d[4]) >= 15 and float(d[6]) >= 8 and d[30] != "":
        if decision >= 0.2:
            for j in range(0, len(variablename)):
                train_data_dict[variablename[j]].append(d[j])
        else:
            for j in range(0, len(variablename)):
                test_data_dict[variablename[j]].append(d[j])
    else:
        print(d)


variable_list =["age", "position", "game_played", "game_started", "min_pg", "fg_pg", "fga_pg", "fg_pct", "pts_pg", \
                "draft_order", "shoot_hand", "player_height", "player_weight"]
float_var_list = ["age", "game_played", "game_started", "min_pg", "fg_pg", "fga_pg", "fg_pct", "pts_pg", "draft_order",\
                  "player_weight"]

dummy_var_list = ["position", "shoot_hand", "player_height"]

#clean up some of missing values in the data set
for vari in float_var_list:
    if vari == "draft_order":
        for i in range(0, len(train_data_dict[vari])):
            v = train_data_dict[vari][i]
            if v == "Undrafted":
                train_data_dict[vari][i] = "61"
            else:
                v = v[0:len(v)-2]
                train_data_dict[vari][i] = str(int(v))
        for j in range(0, len(test_data_dict[vari])):
            v = test_data_dict[vari][j]
            if v == "Undrafted":
                test_data_dict[vari][j] = "61"
            else:
                v = v[0:len(v)-2]
                test_data_dict[vari][j] = str(int(v))

    if vari == "shoot_hand":
        for i in range(0, len(train_data_dict[vari])):
            v = train_data_dict[vari][i]
            if v == " ":
                train_data_dict[vari][i] = "Right"

        for i in range(0, len(test_data_dict[vari])):
            v = test_data_dict[vari][j]
            if v is '':
                test_data_dict[vari][j] = "Right"


# convert categorical variables into dummy variables

def make_dummy(name:str, data_dict:dict, dummy_num:int, var_name_list: list):
    uni_list = set(data_dict[name])
    print(uni_list)

    for value in uni_list:
        data_dict[value] = []
        if value not in var_name_list:
            var_name_list.append(value)

    if len(uni_list) == dummy_num:
        for i in range(0, len(data_dict[name])):
            for values in uni_list:
                if data_dict[name][i] == values:
                    data_dict[values].append(1)
                else:
                    data_dict[values].append(0)
    else:
        print("Warning: Dummy Variable does match input number!")

    return data_dict, var_name_list

train_data_dict, variablename = make_dummy("position", train_data_dict, 5, variablename)
test_data_dict, variablename = make_dummy("position", test_data_dict, 5, variablename)
train_data_dict, variablename = make_dummy("shoot_hand", train_data_dict, 2, variablename)
test_data_dict, variablename = make_dummy("shoot_hand", test_data_dict, 2, variablename)
train_data_dict, variablename = make_dummy("player_height", train_data_dict, 19, variablename)
test_data_dict, variablename = make_dummy("player_height", test_data_dict, 19, variablename)

#separating the sets into txt files for further process

text_file = open("Training_Data.txt", "w")

for name in variablename:
    text_file.write(name + "  ")
text_file.write("\n")

for i in range (0, len(train_data_dict["age"])):
    for name in variablename:
        variable = train_data_dict[name][i]
        text_file.write(str(variable) + "  ")

    text_file.write("\n")

text_file.close()

text_file = open("Test_Data.txt", "w")

for name in variablename:
    text_file.write(name + "  ")
text_file.write("\n")

for i in range (0, len(test_data_dict["age"])):
    for name in variablename:
        variable = test_data_dict[name][i]
        text_file.write(str(variable) + "  ")

    text_file.write("\n")

text_file.close()