
import csv
actions = []
ids = []
with open("/home/taxen/Downloads/Charades/Charades_v1_train.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        action = row['actions'].split(';')
        id = row['id']
        actions.append(action)
        ids.append(id)

row_choice = 3

print("id: ", ids[row_choice], "actions: ", actions[row_choice])
