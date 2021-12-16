import pandas
from utils import project_list

with open('../../data/travistorrent_8_2_2017.csv') as f:
    df = pandas.read_csv(f, low_memory=False)

# print(len(df)) # 3702595

for gh_project_name in project_list:
    out = df[df['gh_project_name'] == gh_project_name]
    print(len(out))
    project_name_on_disk = gh_project_name.split('/')[1]
    out.to_csv('../../data/{}.csv'.format(project_name_on_disk), index=False)
