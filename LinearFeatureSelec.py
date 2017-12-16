from sklearn.linear_model import LinearRegression
import itertools
import math
import numpy as np
import matplotlib.pyplot as plt

TrainData = open("Training_Data.txt","r")
Train_Data = TrainData.readlines()
TrainData.close()

TestData = open("Test_Data.txt","r")
Test_Data = TestData.readlines()
TestData.close()

variablename = Train_Data[0].split("  ")
variablename.remove("\n")

train_data_dict = {}
test_data_dict = {}

for name in variablename:
    train_data_dict[name] = []
    test_data_dict[name] = []

for i in range(1, len(Train_Data)):
    d = Train_Data[i].split("  ")
    for j in range(0, len(variablename)):
        if j != 1 and j != 2 and j != 3 and j!= 30 and j!= 31:
            train_data_dict[variablename[j]].append(float(d[j]))
        else:
            train_data_dict[variablename[j]].append(d[j])

for i in range(1, len(Test_Data)):
    d = Test_Data[i].split("  ")
    for j in range(0, len(variablename)):
        if j != 1 and j != 2 and j != 3 and j!= 30 and j!= 31:
            test_data_dict[variablename[j]].append(float(d[j]))
        else:
            test_data_dict[variablename[j]].append(d[j])

float_var_list = ['age','game_played', 'game_started' ,'min_pg', 'fg3a_pg', 'fg2a_pg','draft_order',\
                   'player_weight',]

dummy_var_list = ['C', 'SG', 'SF', 'PF', 'PG' ,'Right','Left','6-11','7-1', '6-8', '7-3', '6-0', '5-10', '6-5', '6-6',\
                  '5-11','6-1', '6-4', '5-9', '6-7','6-3', '6-10', '7-2', '6-9', '7-0', '6-2']

train_y = train_data_dict['fg2_pct']
train_y_3 = train_data_dict['fg3_pct']

test_y = test_data_dict['fg2_pct']
test_y_3 = test_data_dict['fg3_pct']

#inilizing best subset
optimal_list = []
optimal_list2 = []
optimal_error = []
optimal_error2 = []
init_error_hold = len(train_data_dict["fg2_pct"])
init_error_hold2 = len(train_data_dict["fg3_pct"])

#feature selection algorithm, loop through all possible combinations

for num_features in range(1, len(float_var_list)+1):
    total_var_list = itertools.combinations(float_var_list, num_features)
    temp = list(total_var_list)

    for it in range(0,len(temp)):
        var_list = list(temp[it])
        train_dict = {}

        train_x = np.zeros((len(train_y), len(var_list)))
        for d in var_list:
            train_dict[d] = train_data_dict[d]

        for list_num in range(0, len(var_list)):
            for j in range(0, len(train_y)):
                train_x[j][list_num] = train_data_dict[var_list[list_num]][j]

        test_x = np.zeros((len(test_y), len(train_dict)))

        for list_num in range(0, len(var_list)):
            for j in range(0, len(test_y)):
                test_x[j][list_num] = test_data_dict[var_list[list_num]][j]

        lireg2 = LinearRegression()
        lireg3 = LinearRegression()

        train_y_2 = np.array(train_y)
        train_y_3 = np.array(train_y_3)

        li_reg_obj_2 = lireg2.fit(train_x, train_y_2)
        li_reg_obj_3 = lireg3.fit(train_x, train_y_3)

        pred_y_2_li = li_reg_obj_2.predict(train_x)
        pred_y_3_li = li_reg_obj_3.predict(train_x)

        pred_test_y_2_li = li_reg_obj_2.predict(test_x)
        pred_test_y_3_li = li_reg_obj_3.predict(test_x)

    #initlizating and calculating errors
        error1 = [0]*10
        error2 = [0]*10

        for i in range(0, len(test_y)):
            delta = abs(pred_test_y_2_li[i] - test_y[i])/0.01
            if math.floor(delta) < 9:
                n = math.floor(delta)
                error1[n] = error1[n] + 1
            else:
                error1[9] = error1[9] + 1

        for i in range(0, len(test_y_3)):
            delta = abs(pred_test_y_3_li[i] - test_y_3[i]) / 0.01
            if math.floor(delta) < 9:
                n = math.floor(delta)
                error2[n] = error2[n] + 1
            else:
                error2[9] = error2[9] + 1

        if error1[9] < init_error_hold:
            init_error_hold = error1[9]
            optimal_error = error1
            optimal_list = var_list


        if error2[9] < init_error_hold2:
            init_error_hold2 = error2[9]
            optimal_error2 = error2
            optimal_list2 = var_list


print("\n")
print("The optimal feature for fg2_pct is:")
print(optimal_list)
print("Error Distribution is:")
print(optimal_error)
print("\n")
print("The optimal feature for fg3_pct is:")
print(optimal_list2)
print("Error Distribution is:")
print(optimal_error2)



new_train_list_2 = []
new_train_list_3 = []
for item in optimal_list:
    new_train_list_2.append(item)

for item in optimal_list2:
    new_train_list_3.append(item)

for item in dummy_var_list:
    new_train_list_2.append(item)
    new_train_list_3.append(item)

new_train_x_2 = np.zeros((len(train_y), len(new_train_list_2)))

for list_num in range(0, len(new_train_list_2)):
    for j in range(0, len(train_y)):
        new_train_x_2[j][list_num] = train_data_dict[new_train_list_2[list_num]][j]

new_test_x_2 = np.zeros((len(test_y), len(new_train_list_2)))

for list_num in range(0, len(new_train_list_2)):
    for j in range(0, len(test_y)):
        new_test_x_2[j][list_num] = test_data_dict[new_train_list_2[list_num]][j]

new_train_x_3 = np.zeros((len(train_y), len(new_train_list_3)))

for list_num in range(0, len(new_train_list_3)):
    for j in range(0, len(train_y)):
        new_train_x_3[j][list_num] = train_data_dict[new_train_list_3[list_num]][j]

new_test_x_3 = np.zeros((len(test_y), len(new_train_list_3)))

for list_num in range(0, len(new_train_list_3)):
    for j in range(0, len(test_y)):
        new_test_x_3[j][list_num] = test_data_dict[new_train_list_3[list_num]][j]

#create actual models using library with predictions

lireg2 = LinearRegression()
lireg3 = LinearRegression()

train_y_2 = np.array(train_y)
train_y_3 = np.array(train_y_3)

li_reg_obj_2 = lireg2.fit(new_train_x_2, train_y_2)
li_reg_obj_3 = lireg3.fit(new_train_x_3, train_y_3)

pred_y_2_li = li_reg_obj_2.predict(new_train_x_2)
pred_y_3_li = li_reg_obj_3.predict(new_train_x_3)

pred_test_y_2_li = li_reg_obj_2.predict(new_test_x_2)
pred_test_y_3_li = li_reg_obj_3.predict(new_test_x_3)

#calculating the model performances

def correct_cal(y_test, y_cal):
    dist = [0]*10
    perc_dist = [0]*3
    for i in range(0, len(y_test)):
        diff = abs(y_cal[i] - y_test[i])/0.01
        if y_test[i] != 0:
            per = abs(y_cal[i] - y_test[i])/y_test[i]
        else:
            per = abs(y_cal[i] - y_test[i])
        #print(y_cal[i], y_test[i])
        if diff < 9:
            dist[math.floor(diff)] += 1
        else:
            dist[9] += 1

        if per <= 0.1:
            perc_dist[0] += 1
        elif per <= 0.2:
            perc_dist[1] += 1
        else:
            perc_dist[2] += 1

    return dist, perc_dist

dist_y_2_li, per_dist_y_2_li = correct_cal(train_y_2, pred_y_2_li)
print(dist_y_2_li, per_dist_y_2_li)

dist_y_2_li_test, per_dist_y_2_li_test = correct_cal(test_y, pred_test_y_2_li)
print(dist_y_2_li_test,per_dist_y_2_li_test)

dist_y_3_li, per_dist_y_3_li = correct_cal(train_y_3, pred_y_3_li)
print(dist_y_3_li,per_dist_y_3_li)

dist_y_3_li_test, per_dist_y_3_li_test = correct_cal(test_y_3, pred_test_y_3_li)
print(dist_y_3_li_test,per_dist_y_3_li_test)

# plt.hist(train_y_2, bins = 20)
# plt.title("2pt Percentage Distribution")
# plt.show()
#
# plt.hist(train_y_3, bins = 20)
# plt.title("3pt Percentage Distribution")
# plt.show()