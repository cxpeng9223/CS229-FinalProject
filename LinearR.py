import math

#read the training set
TrainData = open("Training_Data.txt","r")
Train_Data = TrainData.readlines()
TrainData.close()

#read test set
TestData = open("Test_Data.txt","r")
Test_Data = TestData.readlines()
TestData.close()

#read feature names
variablename = Train_Data[0].split("  ")
variablename.remove("\n")

train_data_dict = {}
test_data_dict = {}

#re-creating a dictionary to train the model
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

#the hand-written gradient decent linear regression model using a dictionary of training sets

def grandient_decent(y_list:list, x_dict:dict, alpha:float):
    x_dict["const"] = [1.0] * len(x_dict[list(x_dict.keys())[0]])
    varname = list(x_dict.keys())
# initialize all the parameters
    theta = [0.0]*(len(x_dict))
    old_theta = [0.0]*len(x_dict)
    delta_theta = [0.0]*len(x_dict)

#iterate over gradient decent
    for ite in range(0, 500):
        delta = 0
        # if ite%100 == 0:
        #     print("iteration = " + str(ite))
        for j in range(0, len(varname)):
            grad_sum = 0
            for i in range(0, len(y_list)):
                h = 0
                for x in range (0, len(x_dict)):
                    n = varname[x]
                    h = h + (x_dict[n][i] * theta[x])
                diff = (y_list[i] - h)*x_dict[varname[j]][i]
                grad_sum = grad_sum + diff

            theta[j] = theta[j] + alpha * grad_sum
    # calculating the stopping thershold
            delta_theta[j] = theta[j] - old_theta[j]
            old_theta[j] = theta[j]
            delta = delta + delta_theta[j]*delta_theta[j]

        norm_theta = math.sqrt(delta)
        if norm_theta < 0.000001:
            print("Gradient decent method converges in " + str(ite) + " iterations.")
            break
    return theta, varname

float_var_list = ['age','game_played', 'game_started' ,'min_pg', 'fg3a_pg', 'fg2a_pg','tov_pg', 'pf_pg','draft_order',\
                   'player_weight','orb_pg', 'drb_pg','ast_pg','stl_pg','pts_pg']

dummy_var_list = ['C', 'SG', 'SF', 'PF', 'PG' ,'Right','Left','6-11','7-1', '6-8', '7-3', '6-0', '5-10', '6-5', '6-6',\
                  '5-11','6-1', '6-4', '5-9', '6-7','6-3', '6-10', '7-2', '6-9', '7-0', '6-2']

#preform the actual regression with the written algorithm

theta_1, vlist = grandient_decent(train_data_dict["fg2_pct"],train_data_dict, 0.00000001)
theta_2, vlist2 = grandient_decent(train_data_dict["fg3_pct"],train_data_dict, 0.00000001)
