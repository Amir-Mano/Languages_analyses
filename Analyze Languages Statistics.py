#### Analyze Languages Statistics ####

# read daquitta from csv file
import pandas as pd
data = pd.read_excel('BB_answers.xlsx')
set_data = data['L2'].str.replace(',',' ')
set_data = set_data.str.replace('  ',' ')
set_data = set_data.str.split(' ')

dict_languages = {'אנגלית':0, 'עברית':1}
for key in dict_languages.keys():
    set_data = set_data.replace(key, dict_languages[key])


