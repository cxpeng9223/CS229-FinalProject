import urllib.request as request
from bs4 import BeautifulSoup
from string import ascii_lowercase as lower

#initialize vectors, 33 total

age = []
team = []
league = []
position = []
game_played = []
game_started = []
min_pg = []
fg_pg = []
fga_pg = []
fg_pct = []
fg3_pg = []
fg3a_pg = []
fg3_pct = []
fg2_pg = []
fg2a_pg = []
fg2_pct = []
efg_pct = []
ft_pg = []
fta_pg = []
ft_pct = []
orb_pg = []
drb_pg = []
trb_pg = []
ast_pg = []
stl_pg = []
blk_pg = []
tov_pg = []
pf_pg = []
pts_pg = []
draft_order = []
shoot_hand = []
player_height = []
player_weight = []

variablelist = [age, team, league, position, game_played, game_started, min_pg, fg_pg, fga_pg, fg_pct, fg3_pg, fg3a_pg,\
                fg3_pct, fg2_pg, fg2a_pg, fg2_pct, efg_pct, ft_pg, fta_pg, ft_pct, orb_pg, drb_pg, trb_pg, ast_pg, \
                stl_pg, blk_pg, tov_pg, pf_pg, pts_pg, draft_order, shoot_hand, player_height, player_weight]


for letter in lower:
    if letter != 'x': # the letter 'X' is not on the list of baksetball-reference.com
    #loop over all possible players listed
        soupdic = BeautifulSoup(request.urlopen('https://www.basketball-reference.com/players/'+ letter+'/'), "lxml")
        playerTable = soupdic.find('table', id="players")
        activePlayerList = playerTable.find_all('strong')
        for p in range (0, len(activePlayerList)):
            player = activePlayerList[p].find('a')
            playerAddress = player['href']
            print(playerAddress)
            playersoup = BeautifulSoup(request.urlopen('https://www.basketball-reference.com'+playerAddress),"lxml")

            basicInfo = playersoup.find('div', id="info")

            shtHand = []
            draft = []
            height = []
            weight = []

            #code to handle some of the special cases that encourted during the trials

            for i in range(0, len(basicInfo.find_all('p'))):
                attribute = basicInfo.find_all('p')[i]
                tagList = attribute.find_all('strong')

                if tagList == []:
                    continue
                else:
                    TestText = tagList[0].text
                    TestText = TestText.replace('\n', '').replace('\r', '').replace(' ', '')

                    if TestText == "Position:":
                        if len(tagList) == 2:
                            shtHand = tagList[1].next_sibling
                        else:
                            shtHand = "Right"

                        shtHand = shtHand.replace('\n', '').replace('\r', '').replace(' ', '')
                        heightAndWeight = basicInfo.find_all('p')[i + 1]
                        height = heightAndWeight.find_all('span')[0].find(text=True)
                        weight = heightAndWeight.find_all('span')[1].find(text=True)
                        weight = weight.replace("lb", "")

                    elif TestText == "Draft:":
                        draftTag = basicInfo.find_all('p')[i].find_all('a')
                        draft = draftTag[0].next_sibling
                        draft = draft.split(" ")
                        draft = draft[5]

            if draft == []:
                draft = "Undrafted"

            #separation of data into features

            division = playersoup.find('div', {'class', 'table_outer_container'})
            table = division.find('table', id="per_game")
            data = table.find('tbody')
            stat = data.find_all('td')
            if len(stat) % 29 == 0:
                for i in range(0, len(stat)):
                    variablelist[i % 29].append(stat[i].find(text=True))
                    if i%29 == 28:
                        draft_order.append(draft)
                        shoot_hand.append(shtHand)
                        player_height.append(height)
                        player_weight.append(weight)

                        if len(draft_order) != len(age):
                            print('Error Message')
                            exit(0)


# Writre into the total big raw data file
text_file = open("Data.txt", "w")
for i in range (0, len(draft_order)):
    for j in range (0, len(variablelist)):
        variable = variablelist[j]
        if variable[i] is None:
            text_file.write("0.0" + "  ")
        else:
            text_file.write(variable[i] + "  ")
    text_file.write("\n")

text_file.close()

