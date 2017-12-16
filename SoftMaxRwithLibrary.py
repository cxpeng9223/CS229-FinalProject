from sklearn.linear_model import LogisticRegression
import numpy as np

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

def round_up(ylist:list, dig: int):
    for i in range(0, len(ylist)):
        ylist[i] = round(ylist[i], dig)
    return ylist

train_data_dict["const"] = [1.0] * len(train_data_dict['fg2_pct'])
test_data_dict["const"] = [1.0] * len(test_data_dict['fg2_pct'])

train_data_dict['fg2_pct'] = round_up(train_data_dict['fg2_pct'], 2)
train_data_dict['fg3_pct'] = round_up(train_data_dict['fg3_pct'], 2)
test_data_dict['fg2_pct'] = round_up(test_data_dict['fg2_pct'], 2)
test_data_dict['fg3_pct'] = round_up(test_data_dict['fg3_pct'], 2)

float_var_list = ['age','game_played', 'game_started' ,'min_pg', 'fg3a_pg', 'fg2a_pg','tov_pg', 'pf_pg','draft_order',\
                   'player_weight','orb_pg', 'drb_pg','ast_pg','stl_pg','pts_pg']

train_list_2 = ['age','game_played','min_pg', 'draft_order','player_weight', 'drb_pg','C', 'SG', 'SF', 'PF', 'PG' ,\
              'Right','Left','6-11','7-1', '6-8', '7-3', '6-0', '5-10', '6-5', '6-6','5-11','6-1', '6-4', '5-9', '6-7',\
              '6-3', '6-10', '7-2', '6-9', '7-0', '6-2']

train_list_3 = ['age','game_started','fg3a_pg','draft_order','player_weight', 'drb_pg', 'orb_pg','C', 'SG', 'SF', 'PF', 'PG' ,\
              'Right','Left','6-11','7-1', '6-8', '7-3', '6-0', '5-10', '6-5', '6-6','5-11','6-1', '6-4', '5-9', '6-7',\
              '6-3', '6-10', '7-2', '6-9', '7-0', '6-2']

dummy_var_list = ['C', 'SG', 'SF', 'PF', 'PG' ,'Right','Left','6-11','7-1', '6-8', '7-3', '6-0', '5-10', '6-5', '6-6',\
                  '5-11','6-1', '6-4', '5-9', '6-7','6-3', '6-10', '7-2', '6-9', '7-0', '6-2']


train_y = train_data_dict['fg2_pct']
train_y_3 = train_data_dict['fg3_pct']

test_y = test_data_dict['fg2_pct']
test_y_3 = test_data_dict['fg3_pct']

#round up all the outputs
for i in range(0, len(train_y)):
    train_y[i] = int(train_y[i] * 100)

for i in range(0, len(train_y_3)):
    train_y_3[i] = int(train_y_3[i] * 100)

for i in range(0, len(test_y)):
    test_y[i] = int(test_y[i] * 100)

for i in range(0, len(test_y_3)):
    test_y_3[i] = int(test_y_3[i] * 100)

#create a training X for the library softmax method
train_x = np.zeros((len(train_y), len(train_list_2)))

for list_num in range(0, len(train_list_2)):
    for j in range(0, len(train_y)):
        train_x[j][list_num] = train_data_dict[train_list_2[list_num]][j]

#create a test X for the library softmax method
test_x = np.zeros((len(test_y), len(train_list_2)))

for list_num in range(0, len(train_list_2)):
    for j in range(0, len(test_y)):
        test_x[j][list_num] = test_data_dict[train_list_2[list_num]][j]


train_x_3 = np.zeros((len(train_y), len(train_list_3)))

for list_num in range(0, len(train_list_3)):
    for j in range(0, len(train_y)):
        train_x_3[j][list_num] = train_data_dict[train_list_3[list_num]][j]

test_x_3 = np.zeros((len(test_y_3), len(train_list_3)))

for list_num in range(0, len(train_list_3)):
    for j in range(0, len(test_y)):
        test_x_3[j][list_num] = test_data_dict[train_list_3[list_num]][j]

#create actual models using library
logreg2 = LogisticRegression()
logreg3 = LogisticRegression()

train_y_2 = np.array(train_y)
log_reg_obj_2 = logreg2.fit(train_x, train_y)

train_y_3 = np.array(train_y_3)
log_reg_obj_3 = logreg3.fit(train_x_3, train_y_3)

#
pred_y_2_log = log_reg_obj_2.predict(train_x)
pred_y_3_log = log_reg_obj_3.predict(train_x_3)


pred_test_y_2_log = log_reg_obj_2.predict(test_x)
pred_test_y_3_log = log_reg_obj_3.predict(test_x_3)


#calculating the model performances

def correct_cal(y_test, y_cal):
    dist = [0]*10
    perc_dist = [0]*3
    for i in range(0, len(y_test)):
        diff = abs(int(y_cal[i] - y_test[i]))
        if y_test[i] != 0:
            per = diff/y_test[i]
        else:
            per = diff/100
        #print(y_cal[i], y_test[i])
        if diff < 9:
            dist[diff] += 1
        else:
            dist[9] += 1

        if per <= 0.1:
            perc_dist[0] += 1
        elif per <= 0.2:
            perc_dist[1] += 1
        else:
            perc_dist[2] += 1

    return dist, perc_dist



dist_y_2_log = correct_cal(train_y_2, pred_y_2_log)
print(dist_y_2_log)

dist_y_2_log_test = correct_cal(test_y, pred_test_y_2_log)
print(dist_y_2_log_test)

dist_y_3_log = correct_cal(train_y_3, pred_y_3_log)
print(dist_y_3_log)

dist_y_3_log_test = correct_cal(test_y_3, pred_test_y_3_log)
print(dist_y_3_log_test)

pred_score = log_reg_obj_2.score(train_x,train_y)
pred_score3 = log_reg_obj_3.score(train_x_3,train_y_3)


