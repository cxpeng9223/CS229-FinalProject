import math
import random

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

#round up function
def round_up(ylist:list, dig: int):
    for i in range(0, len(ylist)):
        ylist[i] = round(ylist[i], dig)
    return ylist


#Softmax training function

def softmax(y_list:list, x_dict:dict, alpha: float):
    #initialize theta
    theta = {}
    varname = list(x_dict.keys())
    uni_class_1 = list(set(y_list))

    for classes in uni_class_1:
        theta[classes] = []
        for i in range(0, len(x_dict)):
            theta[classes].append(random.random()/100)

    for it in range(0, 3): #number of iteration
        print("Iteration #" + str(it))
        print(theta)
        for k in range(0, len(uni_class_1)): #loop over all the possible k(classes)
            value = uni_class_1[k]
            l_theta = theta[value]

            for j in range(0, len(varname)): # for each j in the the class
                grad_sum = 0

                for i in range(0, len(y_list)): # for each sample
                    y_i = y_list[i]
                    theta_i = theta[y_i]
                    temp_sum = 0

                    for jj in range(0, len(varname)): # calculating the derivatives
                        x = x_dict[varname[jj]]
                        temp_sum += theta_i[jj]* x[i]
                    numerator = math.exp(temp_sum)
                    denominator = 0

                    for kk in range(0, len(uni_class_1)):
                        temp_sum_2 = 0
                        theta_temp = theta[uni_class_1[kk]]

                        for jjj in range(0, len(varname)):
                            x = x_dict[varname[jjj]]
                            temp_sum_2 += theta_temp[jj] * x[i]
                        denominator += math.exp(temp_sum_2)
                    pyi = numerator/denominator

                    if y_i == value:
                        diff = 1 - pyi
                    else:
                        diff = -pyi
                    grad_x = x_dict[varname[j]]
                    grad_sum += grad_x[i]*diff

                l_theta[j] += l_theta[j] - grad_sum * alpha

            # theta[value] = l_theta

        # theta_norm = [0]*len(uni_class_1)  # calculating the total norm as a threshold to the iteration
        # for i in range(0, len(uni_class_1)):
        #     sum_1 = 0
        #     for j in range(0, len(varname)):
        #         delta = old_theta[i][j] - theta[i][j]
        #         sum_1 += delta * delta
        #     theta_norm[i] = math.sqrt(sum_1)
        #
        # print (theta_norm)

    return theta


 #round up the output features to train the softmax
train_data_dict['fg2_pct'] = round_up(train_data_dict['fg2_pct'], 2)
train_data_dict['fg3_pct'] = round_up(train_data_dict['fg3_pct'], 2)
test_data_dict['fg2_pct'] = round_up(test_data_dict['fg2_pct'], 2)
test_data_dict['fg3_pct'] = round_up(test_data_dict['fg3_pct'], 2)



float_var_list = ['age','game_played', 'game_started' ,'min_pg', 'fg3a_pg', 'fg2a_pg','tov_pg', 'pf_pg','draft_order',\
                   'player_weight','orb_pg', 'drb_pg','ast_pg','stl_pg','pts_pg']

#changing features that put into the model, this was kept small for the purposes of fast tuning the model
train_list = ['age']

dummy_var_list = ['C', 'SG', 'SF', 'PF', 'PG' ,'Right','Left','6-11','7-1', '6-8', '7-3', '6-0', '5-10', '6-5', '6-6',\
                  '5-11','6-1', '6-4', '5-9', '6-7','6-3', '6-10', '7-2', '6-9', '7-0', '6-2']

train_x = {}
train_y = train_data_dict['fg2_pct']
for l in train_list:
    train_x[l] = train_data_dict[l]
    
#the actual softmax training
theta_test = softmax(train_y, train_x, 0.000000001)

y_test = [0,0]
var_list = list(train_x.keys())
value_list = list(theta_test.keys())

#actual function to performance of the softmax
for i in range(0, len(train_y)):
    p = 0
    y_cal = 0
    denominator = 0
    for v in value_list:
        theta_i = theta_test[v]
        sum_1 = 0
        for j in range(0, len(var_list)):
            sum_1 += theta_i[j]*train_x[var_list[j]][i]
        denominator += math.exp(sum_1)

    for v in value_list:
        theta_i = theta_test[v]
        sum_1 = 0
        for j in range(0, len(var_list)):
            sum_1 += theta_i[j]*train_x[var_list[j]][i]
        numerator = math.exp(sum_1)
        p_test = numerator/denominator
        if p_test > p:
            p = p_test
            y_cal = v

    print(y_cal)

    if y_cal == train_y[i]:
        y_test[0] += 1
    else:
        y_test[1] += 1


print(y_test)